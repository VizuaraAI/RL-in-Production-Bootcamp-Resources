"""Math invariants that must hold or every number this project reports is wrong.

Runs on CPU in seconds, no GPU and no model download. Run it before any paid run:

    python3 tests/test_math.py

Why these two specifically: both are memory optimisations that change HOW a quantity is
computed without being allowed to change WHAT it is. Each was introduced to fix a real OOM
on real hardware, and each could silently alter the loss instead of just its memory profile
— which would look like a training difference rather than a bug.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from race.grpo import Batch, grpo_loss, token_logprobs  # noqa: E402

V, H = 97, 16


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.head = nn.Linear(H, V)

    def forward(self, input_ids, attention_mask=None):
        class O:
            pass
        o = O()
        o.logits = self.head(self.emb(input_ids))
        return o


def _batch(B, T):
    ids = torch.randint(0, V, (B, T))
    comp = torch.zeros(B, T, dtype=torch.long)
    comp[:, T // 2:] = 1
    # ragged completions, so micro-batches have UNEQUAL token counts — that is what makes
    # the re-weighting non-trivial. With equal counts a plain mean would also pass and the
    # test would prove nothing.
    for i in range(B):
        comp[i, T - (i % 5):] = 0
    return Batch(input_ids=ids, attention_mask=torch.ones(B, T, dtype=torch.long),
                 completion_mask=comp, advantages=torch.randn(B),
                 logp_behavior=torch.randn(B, T) * 0.1,
                 lag=torch.zeros(B, dtype=torch.long))


def _grads(model):
    return torch.cat([p.grad.flatten().clone() for p in model.parameters()])


def test_microbatch_gradient_equivalence(B=12, T=20, mb=4) -> bool:
    """Accumulating over micro-batches must give the SAME gradient as one full batch.

    Guards the fix for: "Tried to allocate 44.99 GiB" — a [B, T, 151936] logits tensor.
    """
    torch.manual_seed(0)
    model, b = Tiny(), _batch(B, T)

    model.zero_grad()
    loss, _ = grpo_loss(model(b.input_ids).logits, b)
    loss.backward()
    full = _grads(model)

    model.zero_grad()
    total = b.completion_mask.sum().clamp(min=1.0)
    for s in range(0, B, mb):
        e = min(s + mb, B)
        sub = Batch(b.input_ids[s:e], b.attention_mask[s:e], b.completion_mask[s:e],
                    b.advantages[s:e], b.logp_behavior[s:e], b.lag[s:e])
        l, _ = grpo_loss(model(sub.input_ids).logits, sub)
        (l * (sub.completion_mask.sum() / total)).backward()
    micro = _grads(model)

    rel = (full - micro).abs().max().item() / max(full.abs().max().item(), 1e-12)
    cos = F.cosine_similarity(full, micro, dim=0).item()
    ok = rel < 1e-5 and cos > 1 - 1e-9
    print(f"  microbatch gradient: relative={rel:.3e} cosine={cos:.10f} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_chunked_logprobs() -> bool:
    """Chunked logsumexp must equal the naive fp32 log_softmax.

    Guards the fix for the 6.8 GiB [B, T, V] fp32 tensor that OOM'd an L4.
    """
    torch.manual_seed(1)
    logits = torch.randn(3, 30, 51)
    ids = torch.randint(0, 51, (3, 30))
    naive = F.pad(torch.log_softmax(logits[:, :-1].float(), -1)
                  .gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1), (1, 0))
    ok = True
    for c in (1, 4, 7, 1024):
        d = (token_logprobs(logits, ids, chunk=c) - naive).abs().max().item()
        ok &= d < 1e-5
        print(f"  chunked logprobs (chunk={c:4d}): max|delta|={d:.3e} "
              f"{'PASS' if d < 1e-5 else 'FAIL'}")
    return ok


def test_group_advantages_zero_mean() -> bool:
    """Within a group the advantages must be mean-zero — that IS the GRPO baseline.
    If this drifts, every gradient carries a constant bias."""
    from race.grpo import group_advantages
    torch.manual_seed(2)
    G = 8
    r = torch.rand(5 * G)
    adv = group_advantages(r, G).view(-1, G)
    m = adv.mean(dim=1).abs().max().item()
    ok = m < 1e-5
    print(f"  group advantage mean: max|mean|={m:.3e} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("math invariants:")
    results = [
        test_microbatch_gradient_equivalence(),
        test_chunked_logprobs(),
        test_group_advantages_zero_mean(),
    ]
    print("\nALL PASS" if all(results) else "\nFAILURES PRESENT — do not run a paid job")
    raise SystemExit(0 if all(results) else 1)
