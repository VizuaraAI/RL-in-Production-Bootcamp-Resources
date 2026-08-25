"""ECHO Environment-Prediction Loss (the core contribution of arxiv 2605.24517).

L_ECHO(θ) = L_GRPO(θ; A) + λ · L_Env(θ; O')                                (eq 1)

    L_Env(θ; O') = -(1/Z) Σ_{t∈O'} log p_θ(x_t | x_<t),   Z = |O|          (eq 3)

Key subtleties from the paper we must get exactly right:
  * O'  = the terminal-output ("env") tokens ONLY — the <command_output>…</command_output>
          span — EXCLUDING the harness "warning-prefix" tokens (§3.2). Warnings are
          low-entropy and get memorized in ~60 steps, so they lose useful gradient.
  * Z   = |O| = TOTAL observation length (env + warning tokens), NOT |O'|. The paper
          normalizes by |O| "so runs with different target subsets remain comparable
          on a per-observation scale" (§3.1).
  * per-sequence normalization: each sequence i is divided by its own |O_i|, then we
          average across sequences (sequence-level aggregation, matching the GRPO term).
  * λ = 0.05 (base init) / 0.02 (SFT init), constant — self-annealing because L_Env
          falls rapidly as the model learns terminal-output statistics (§3.3).
  * SAME forward pass as GRPO: `logprobs` are the already-computed per-token log-probs
          of the realized tokens; ECHO just gathers them at O' instead of A (Alg 1).

This module is framework-agnostic: give it per-token logprobs and the two masks.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch


def compute_env_prediction_loss(
    logprobs: torch.Tensor,          # [B, T]  log p_θ(x_t | x_<t) of the realized token at t
    env_target_mask: torch.Tensor,   # [B, T]  1 at O' positions (env output tokens, warnings EXCLUDED)
    obs_total_mask: torch.Tensor,    # [B, T]  1 at ALL observation positions O (env + warnings)
    reduction: str = "sequence_mean",  # match GRPO's sequence-level aggregation
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return (L_Env, metrics). L_Env is the *unweighted* env-prediction loss; caller
    scales by λ and adds to the GRPO policy loss. `metrics` includes the per-token env
    cross-entropy in nats (the quantity plotted in the paper's Fig 3 / Fig 6).
    """
    if logprobs.dim() != 2:
        raise ValueError(f"expected [B,T] logprobs, got {tuple(logprobs.shape)}")
    env_target_mask = env_target_mask.to(logprobs.dtype)
    obs_total_mask = obs_total_mask.to(logprobs.dtype)

    # numerator: sum of NEGATIVE logprob over the O' (env-target) tokens, per sequence
    nll_per_tok = -logprobs
    seq_target_nll = (nll_per_tok * env_target_mask).sum(dim=-1)          # [B]
    seq_obs_len = obs_total_mask.sum(dim=-1)                              # [B]  = |O_i|
    seq_target_cnt = env_target_mask.sum(dim=-1)                          # [B]  = |O'_i|

    # eq 3: per-sequence CE normalized by TOTAL observation length |O_i|
    valid = seq_obs_len > 0
    seq_env_loss = torch.where(valid, seq_target_nll / (seq_obs_len + eps),
                               torch.zeros_like(seq_target_nll))          # [B]

    if reduction == "sequence_mean":
        # mean over sequences that actually have observations
        denom = valid.sum().clamp(min=1)
        env_loss = seq_env_loss.sum() / denom
    elif reduction == "token_mean":
        # global token-level: total env NLL / total env-target tokens
        env_loss = seq_target_nll.sum() / (seq_target_cnt.sum() + eps)
    else:
        raise ValueError(f"unknown reduction {reduction!r}")

    # diagnostics
    with torch.no_grad():
        tot_target = seq_target_cnt.sum()
        per_token_ce = (seq_target_nll.sum() / (tot_target + eps)).item()  # nats/token on O'
        metrics = {
            "env_loss": env_loss.detach().item(),
            "env_ce_per_token_nats": per_token_ce,
            "env_target_tokens": int(tot_target.item()),
            "obs_total_tokens": int(seq_obs_len.sum().item()),
            "frac_seqs_with_obs": float(valid.float().mean().item()),
        }
    return env_loss, metrics


def echo_total_loss(
    grpo_loss: torch.Tensor,
    logprobs: torch.Tensor,
    env_target_mask: torch.Tensor,
    obs_total_mask: torch.Tensor,
    lam: float = 0.05,
    reduction: str = "sequence_mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_ECHO = L_GRPO + λ · L_Env  (eq 1)."""
    env_loss, metrics = compute_env_prediction_loss(
        logprobs, env_target_mask, obs_total_mask, reduction=reduction
    )
    total = grpo_loss + lam * env_loss
    metrics["lambda"] = lam
    metrics["grpo_loss"] = grpo_loss.detach().item()
    metrics["echo_total_loss"] = total.detach().item()
    return total, metrics
