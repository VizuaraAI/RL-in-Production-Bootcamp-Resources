# ECHO × SkyRL patch (the core scientific contribution)

`echo_skyrl.patch` — a 2-file, ~62-line diff on SkyRL that adds the ECHO auxiliary loss.
Reproduce: `cd upstream/SkyRL && git apply ../../echo_skyrl.patch`.

## What it does
Implements `L_ECHO = L_GRPO(θ;A) + λ·L_Env(θ;O')` (arxiv 2605.24517, eq 1) by adding a
length-normalized cross-entropy on environment-observation tokens to the RL loss, reusing
the **same forward pass** as GRPO (the paper's "world model for free" property).

## Why this injection point (verified against the code)
- SkyRL's actor computes per-token logprobs over the **full response span** including
  observation tokens (`model_wrapper.py:337,393` → `action_log_probs = log_probs[:, -num_actions-1:-1]`,
  where `num_actions` = full response length). So obs-token logprobs are already present and
  differentiable at the loss site — nothing is thrown away before the loss.
- Two masks travel together to the worker: `loss_mask` (1=assistant action) and
  `action_mask`/`response_mask` (1=every response token). Therefore
  **`O = (action_mask>0) & (loss_mask==0)`** = env-observation tokens, derivable locally.
- A registered custom policy loss (Option A) can't see the obs mask (fixed signature), so we
  add the term in `worker._forward_backward_micro` RL branch alongside the existing KL/entropy
  auxiliary terms (Option B) — the minimal, idiomatic change.

## Files changed
1. `skyrl/train/config/config.py` — new `EchoConfig{enabled, lambda_, exclude_warning_prefix}`,
   added as `AlgorithmConfig.echo`. CLI: `trainer.algorithm.echo.enabled=true trainer.algorithm.echo.lambda_=0.05`.
2. `skyrl/backends/skyrl_train/workers/worker.py` — RL branch:
   - `obs_total_mask = (action_mask>0)&(loss_mask==0)`; `obs_target_mask = obs_total_mask`
     (× `env_output_mask` if `exclude_warning_prefix` and the generator supplies it, §3.2).
   - `env_ce = sequence_mean over seqs of  [ Σ_{O'} -logp / |O| ]`  (per-seq |O|-normalized, eq 3).
   - `loss += λ · env_ce · microbatch_weight` (treated like KL/entropy: not pre-scaled).
   - metrics `echo/{env_ce,env_ce_term,obs_tokens,lambda}`.

## Correctness tests (all pass, `scripts/`)
- `test_echo_loss.py` — hand-computed eq-3 math, §3.2 warning-exclusion effect, |O|-vs-|O'|
  normalization, batch/skip-empty, gradient localized to env tokens only.
- `test_worker_equiv.py` — the in-worker inline math ≡ the reference kernel (with/without
  warning-exclusion) + step-wise-mode inert detection.

## CRITICAL operational note (from the integration spec)
ECHO needs an **interleaved multi-turn rollout** where observations sit INSIDE the response
span (`response_mask=1, loss_mask=0`). Harbor's **step-wise** mode (`step_wise_trajectories=true`)
puts observations in the *next* prompt (fully masked) → `obs_mask` empty → **ECHO silently inert**.
Guard: watch `echo/obs_tokens` in the first training steps — it MUST be > 0. Use the
concatenated/merged multi-turn path (or `SkyRLGymGenerator` multi-turn) so obs tokens are in-span.

## Warning-exclusion (O' faithful vs approximate)
- v1 (approx): `exclude_warning_prefix=false` → O' = all observation tokens (warnings included).
  Fine as a first cut (warnings only appear on parse-failure turns; λ=0.05).
- Faithful: terminal env wraps real output in a sentinel span and the generator emits
  `env_output_mask` (1 on terminal-output tokens only); set `exclude_warning_prefix=true`.
  Plumbing the `env_output_mask` through Experience is the only remaining generator-side work.
