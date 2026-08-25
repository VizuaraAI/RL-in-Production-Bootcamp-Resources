# ECHO replication — feasibility probe results (2026-07-21)

Paper: **ECHO: Terminal Agents Learn World Models for Free** (arxiv 2605.24517, MSR).
Goal: replicate TerminalBench-2.0 accuracy (8B: 2.70→5.17 pass@1; 14B: 5.17→10.79).

## Core method (the whole scientific contribution)
`L_ECHO = L_GRPO(θ; A) + λ·L_Env(θ; O')`
- `A` = assistant action tokens (standard GRPO on these).
- `O'` = terminal-output ("env") tokens ONLY, excluding harness warning-prefix tokens.
- `L_Env = -(1/|O|) Σ_{t∈O'} log p_θ(x_t|x_<t)`, normalized by TOTAL obs length |O|.
- `λ = 0.05` (base init) / `0.02` (SFT init). Constant, self-annealing.
- Shares ONE forward pass with GRPO — just an extra loss mask. No extra rollouts/teacher.

## Training recipe (paper §4, App B)
GRPO: n=16 rollouts/prompt, batch 16, lr 1e-6 constant (AdamW β=0.9/0.95, wd 0.01),
grad clip 0.2, ε_lo=0.2 / ε_hi=0.28 (clip-higher, DAPO), no KL, prompt-level adv norm,
sequence-level loss agg, rollout temp 0.8. Reward = 1 if final unit tests pass else 0.
500 GRPO steps on 8×B200 (~24–48h). Episodes ≤16 turns, 16k ctx, ≤2048 gen tok/turn.

## Eval
- TB2 (TerminalBench-2.0, 89 tasks): Terminus-2 harness, 5 attempts, temp 0.6, 32k ctx,
  1200s agent+verifier timeouts, seed 42. SE on pass@1 at n=5 ≈ 1.5pp.
- Internal: val100 (100), ITD (71), TBLite (100) — minimal harness, 8 rollouts, temp 0.6.

## Modal feasibility — BOTH CRITICAL GATES GREEN ✅
Account: `teamvizuara` (authenticated via ~/.modal.toml). modal 1.5.2, uv venv py3.12.

1. **GPU** — `@app.function(gpu="H100")` ran: NVIDIA **H100 80GB HBM3**, torch 2.13.0+cu130,
   CUDA 13.0, matmul OK. (Need to confirm 8-GPU single-node availability + B200/H200.)
2. **Rollout execution** — `modal.Sandbox.create(image=from_registry("ubuntu:22.04"))`
   works: exec agent bash turn-by-turn, gVisor-isolated, **STATE PERSISTS across exec calls**
   (wrote /tmp/state.txt, read it back). This IS the terminal-task rollout model. Parallelizable.

## Env gaps to fill
- No HF token stored (Qwen3 is public/ungated; may need token for some datasets).
- No OpenAI/GPT-5 key (paper uses GPT-5 to filter tasks — deferrable for public-data repro).

## Verdict
The two hardest infra risks (GPU + isolated stateful bash rollouts) are cleared on the
actual account. Remaining work is engineering: RL trainer + ECHO loss + data + eval harness.
