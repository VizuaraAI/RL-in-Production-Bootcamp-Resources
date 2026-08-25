# Project 3 — ECHO: a world model, for free

Replication of **ECHO: Terminal Agents Learn World Models for Free** (arXiv 2605.24517, MSR)
on SkyRL + Modal. A terminal agent is trained with GRPO plus **one extra job**: predict what
the terminal will say back. That auxiliary loss reuses the same forward pass — a world model
for almost no extra compute:

```
L_ECHO = L_GRPO(θ; A) + λ · L_Env(θ; O′)        λ = 0.05
```

- `A`  — assistant-action tokens (standard GRPO on these)
- `O′` — terminal-output ("environment") tokens, excluding the warning prefix
- `L_Env` — length-normalized cross-entropy on `O′`, computed from the **same forward pass**
  as GRPO: just a second loss mask.

## The core contribution: `echo_skyrl.patch`

The whole method is a **2-file, ~62-line diff on SkyRL** — see [PATCH.md](PATCH.md) for the
line-by-line rationale, verified against SkyRL's actor code.

```bash
git clone https://github.com/NovaSky-AI/SkyRL upstream/SkyRL
cd upstream/SkyRL && git apply ../../echo_skyrl.patch
# enable at train time:
#   trainer.algorithm.echo.enabled=true trainer.algorithm.echo.lambda_=0.05
```

Why the loss is injected in `worker._forward_backward_micro` (and not as a registered custom
policy loss): SkyRL already computes differentiable logprobs over the **full response span**
including observation tokens, and the two masks that travel to the worker
(`loss_mask` = assistant actions, `response_mask` = all response tokens) let the observation
mask be derived locally as `O = (response_mask > 0) & (loss_mask == 0)`. A registered policy
loss has a fixed signature and can't see it.

> **The gotcha that costs people a week** (see [notes/02](notes/02_rollout_architecture.md)):
> Harbor's *step-wise* mode puts observations in the **next prompt**, never inside a response
> span — so `O` is empty and ECHO is **silently inert** while training "succeeds". Use the
> concatenated multi-turn path, and watch `echo/obs_tokens` in the first steps: it MUST be > 0.

## Layout

```
echo_skyrl.patch     the ECHO loss as a diff on SkyRL (the scientific payload)
PATCH.md             what the patch does, why that injection point, the step-wise trap
PLAN.md              the locked replication plan: recipe, phases, honest success criteria
scripts/             echo_loss.py (reference kernel) + test_echo_loss.py, test_worker_equiv.py
                     — hand-computed eq-3 math, warning-exclusion, gradient-localization (all pass)
terminal_env/        the rollout side: modal.Sandbox (gVisor) terminal sessions, Harbor task
                     runner, reward = unit-tests-pass; test_terminal_session.py
modal/               training entrypoints + images for Modal (train.py, echo_image.py, probe.py)
prime/               alternative Prime Intellect launcher (modal_prime.py)
configs/             matched rl-grpo.toml vs rl-echo.toml — identical except the ECHO switch
data/build_corpus.py builds the training corpus from endless-terminals + OpenThoughts-Agent-v1-RL
sample_tasks/        one full Harbor task (environment, tests, reference solution) to see the format
notes/               the replication journal: feasibility, verified ground truth (every model/dataset
                     checked against live HF/GitHub APIs), the rollout-architecture decision, and the
                     validated rollout+verifier cycle (reference solve.sh → 1.0, empty agent → 0.0)
```

## Reproduce

1. **Verify the loss math on CPU first** (free, seconds):
   `python scripts/test_echo_loss.py && python scripts/test_worker_equiv.py`
2. **Prove the rollout/verifier cycle** on one task (Modal, cents):
   `python terminal_env/sandbox_task.py` — reference solution must score 1.0, empty agent 0.0.
3. **Smoke** (H100:1–2, ~$5–15): a few GRPO steps end-to-end; check `echo/obs_tokens > 0`.
4. **The A/B** (H100:8): matched 500-step GRPO vs ECHO on Qwen3-8B — identical base, seed,
   and harness; only the extra loss differs. Recipe in [PLAN.md](PLAN.md) (n=16 rollouts/prompt,
   lr 1e-6, ε_lo/ε_hi 0.2/0.28, no KL, reward = tests pass). Eval: Harbor + Terminus-2 on
   terminal-bench-2.0 (89 tasks), 5 rollouts, temp 0.6.

**Honest scope** (from PLAN.md): the primary claim is the *relative* ECHO > GRPO doubling under
identical conditions. The paper's absolute numbers (2.70→5.17 pass@1 at 8B) depend on ~6k private
training tasks that were never released — our corpus is reconstructed from the public sets, so
absolutes are reported best-effort with that gap stated plainly.

*Vizuara AI Labs · RL in Production · Phase 2, Project 3.*
