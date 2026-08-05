"""Rollouts -> padded training Batch. Shared by both arms (fairness contract).

If the arms padded, masked, or normalised differently, the loss would see different tensors
for the same data and the comparison would be meaningless. So this lives in exactly one place.
"""
from __future__ import annotations

import torch

from .grpo import Batch, group_advantages


def rollouts_to_batch(rollouts, tokenizer, group_size: int, device,
                      pad_id: int | None = None) -> tuple[Batch, dict]:
    """Pack rollouts into a right-padded batch.

    Layout per row:  [prompt tokens ..., completion tokens ..., PAD ...]
    completion_mask is 1 ONLY on completion tokens — the loss must never be applied to the
    prompt, which the policy did not choose.

    Rollouts must arrive grouped: group_size consecutive entries share a prompt, because
    group_advantages() reshapes to [n_groups, group_size]. Caller guarantees this; we assert
    it rather than silently mis-pairing advantages with rollouts.
    """
    assert len(rollouts) % group_size == 0, (
        f"{len(rollouts)} rollouts is not a multiple of group_size={group_size}")

    # STRONGER: every group must be rollouts of the SAME prompt. The modulo check above is
    # necessary but nowhere near sufficient — an arm that generates one rollout per prompt
    # also satisfies it, and then group_advantages() silently computes the GRPO baseline
    # across unrelated questions. The loss still decreases, so nothing looks wrong. This
    # exact bug shipped in the async arm and was caught only by a sample-count mismatch.
    for gi in range(0, len(rollouts), group_size):
        pids = {r.pid for r in rollouts[gi:gi + group_size]}
        if len(pids) != 1:
            raise ValueError(
                f"group at index {gi} spans {len(pids)} distinct prompts ({sorted(pids)!r}) "
                f"— GRPO's advantage baseline must be computed WITHIN one prompt. The "
                f"generator is not returning group_size rollouts per prompt.")
    pad_id = pad_id if pad_id is not None else (tokenizer.pad_token_id or tokenizer.eos_token_id)

    seqs = [r.prompt_ids + r.completion_ids for r in rollouts]
    T = max(len(s) for s in seqs)
    B = len(seqs)

    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    comp = torch.zeros((B, T), dtype=torch.long)
    logp_b = torch.zeros((B, T), dtype=torch.float32)

    for i, (r, s) in enumerate(zip(rollouts, seqs)):
        n, p = len(s), len(r.prompt_ids)
        input_ids[i, :n] = torch.tensor(s, dtype=torch.long)
        attn[i, :n] = 1
        comp[i, p:n] = 1
        # r.logp[j] is the log-prob of completion token j, which sits at absolute index p+j.
        if r.logp:
            k = min(len(r.logp), n - p)
            logp_b[i, p:p + k] = torch.tensor(r.logp[:k], dtype=torch.float32)

    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
    adv = group_advantages(rewards, group_size)
    lag = torch.tensor([r.lag for r in rollouts], dtype=torch.long)

    batch = Batch(
        input_ids=input_ids.to(device),
        attention_mask=attn.to(device),
        completion_mask=comp.to(device),
        advantages=adv.to(device),
        logp_behavior=logp_b.to(device),
        lag=lag.to(device),
    )
    info = {
        "n": B, "pad_frac": float(1 - attn.float().mean()),
        "mean_reward": float(rewards.mean()),
        "mean_completion_len": float(comp.sum(1).float().mean()),
        "zero_adv_groups": float((adv.abs() < 1e-6).float().mean()),
        "mean_lag": float(lag.float().mean()),
    }
    return batch, info


def vllm_output_to_rollouts(outs, problems, reward_fn, lag: int, tokenizer):
    """Flatten vLLM RequestOutputs into grouped Rollout records.

    Ordering is critical: vLLM returns one RequestOutput per prompt with n completions
    inside it, so iterating prompt-major then completion-minor gives exactly the grouping
    rollouts_to_batch() expects.
    """
    from .engine import Rollout

    rolls = []
    for out, prob in zip(outs, problems):
        pids = list(out.prompt_token_ids)
        for c in out.outputs:
            lp = []
            if c.logprobs:
                for tok, d in zip(c.token_ids, c.logprobs):
                    e = d.get(tok) if isinstance(d, dict) else None
                    lp.append(float(getattr(e, "logprob", 0.0)) if e is not None else 0.0)
            rolls.append(Rollout(
                pid=prob.pid, prompt_ids=pids, completion_ids=list(c.token_ids),
                logp=lp, reward=reward_fn(c.text, prob.answer), lag=lag))
    return rolls
