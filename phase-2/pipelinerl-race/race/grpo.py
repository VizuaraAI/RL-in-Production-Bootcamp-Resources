"""GRPO — imported by BOTH arms. There is exactly one copy of this on purpose.

If each arm had its own loss, any measured difference could be an artefact of the loss
rather than of the system architecture, and the whole experiment would be worthless.
Both arms call `grpo_loss` with tensors assembled the same way; the only difference between
them is *when* the data was generated relative to the current weights (`lag`), which is
precisely the thing under study.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Batch:
    """A padded training batch. `logp_behavior` is the log-prob under the policy that
    ACTUALLY generated the tokens — for conventional RL that is the current policy
    (ratio == 1), for PipelineRL it is a slightly older one, which is why the clipped
    importance ratio below is not decorative."""
    input_ids: torch.Tensor       # [B, T]
    attention_mask: torch.Tensor  # [B, T]
    completion_mask: torch.Tensor # [B, T] 1 on generated tokens only (never on the prompt)
    advantages: torch.Tensor      # [B]
    logp_behavior: torch.Tensor   # [B, T] log-probs from the generator
    lag: torch.Tensor             # [B] optimizer steps of staleness


def group_advantages(rewards: torch.Tensor, group_size: int,
                     eps: float = 1e-4) -> torch.Tensor:
    """Group-normalised advantage: A = (r - mean_g) / (std_g + eps).

    Reshapes [n_groups * G] -> [n_groups, G] so each prompt is its own baseline. A group
    where every rollout got the same reward yields A == 0 and contributes no gradient —
    this is the 'zero advantage' case that silently empties batches when the policy is too
    weak (or too strong) for the task. We return it as-is and let the caller count them,
    rather than filtering here, so both arms account for it identically.
    """
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor,
                   chunk: int = 1024) -> torch.Tensor:
    """log p(x_t | x_<t) aligned so index t holds the log-prob OF token t.

    logits[:, t] predicts token t+1, so we shift by one and pad position 0 with zero.
    Getting this off by one is the classic silent RL bug: the loss still decreases, but it
    is training on the wrong tokens.

    MEMORY: the obvious implementation, `log_softmax(logits.float())`, materialises a
    [B, T, V] fp32 tensor. With Qwen2.5's 151,936-token vocabulary that is 6.8 GiB for a
    16x700 batch — on its own enough to OOM an L4, which is exactly what the first smoke
    test hit. Instead we compute logp = x_gathered - logsumexp(x) in slices along the time
    axis, so peak memory is bounded by `chunk` timesteps rather than the full sequence.
    Mathematically identical, and it never builds the full fp32 distribution.
    """
    B, Tm1, _ = logits[:, :-1].shape
    tgt = input_ids[:, 1:]
    out = torch.empty((B, Tm1), dtype=torch.float32, device=logits.device)
    for s in range(0, Tm1, chunk):
        e = min(s + chunk, Tm1)
        sl = logits[:, s:e].float()
        gathered = sl.gather(-1, tgt[:, s:e].unsqueeze(-1)).squeeze(-1)
        out[:, s:e] = gathered - torch.logsumexp(sl, dim=-1)
        del sl, gathered
    return F.pad(out, (1, 0), value=0.0)


def grpo_loss(logits: torch.Tensor, batch: Batch, clip_eps: float = 0.2,
              kl_coef: float = 0.0,
              ref_logp: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Token-level clipped policy-gradient loss with group-normalised advantages.

    Returns (loss, stats). Normalisation is by total completion tokens in the batch, not
    per-sequence-then-mean, so long completions are not down-weighted — matching the
    'token-level' convention the PipelineRL/GRPO line of work uses.
    """
    logp = token_logprobs(logits, batch.input_ids)
    mask = batch.completion_mask.float()
    ntok = mask.sum().clamp(min=1.0)

    ratio = torch.exp(logp - batch.logp_behavior)
    adv = batch.advantages.unsqueeze(1)

    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg = -torch.min(unclipped, clipped)
    loss = (pg * mask).sum() / ntok

    stats = {
        "pg_loss": loss.detach(),
        "ratio_mean": ((ratio * mask).sum() / ntok).detach(),
        "clip_frac": (((ratio - 1.0).abs() > clip_eps).float() * mask).sum().detach() / ntok,
        "adv_abs_mean": batch.advantages.abs().mean().detach(),
        "zero_adv_frac": (batch.advantages.abs() < 1e-6).float().mean().detach(),
        "lag_mean": batch.lag.float().mean().detach(),
        "tokens": ntok.detach(),
    }

    if kl_coef > 0.0 and ref_logp is not None:
        # k3 estimator — non-negative and lower variance than (logp_ref - logp).
        d = ref_logp - logp
        kl = (torch.exp(d) - d - 1.0)
        kl_term = (kl * mask).sum() / ntok
        loss = loss + kl_coef * kl_term
        stats["kl"] = kl_term.detach()

    return loss, stats
