# ECHO Replication — Locked Plan

**Paper:** ECHO: Terminal Agents Learn World Models for Free (arxiv 2605.24517, MSR).
**Goal:** reproduce TerminalBench-2.0 accuracy — 8B: GRPO 2.70 → ECHO 5.17 pass@1; 14B: 5.17 → 10.79.
**Platform:** Modal (`teamvizuara`), full-scale real GPU runs. Cost is not a constraint.

## The method (one isolable change)
`L_ECHO = L_GRPO(θ;A) + λ·L_Env(θ;O')`, λ=0.05.
- A = assistant-action tokens (standard GRPO). O' = terminal-output ("env") tokens, excluding warning-prefix.
- `L_Env = -(1/|O|) Σ_{t∈O'} log p_θ(x_t|x_<t)` — same forward pass as GRPO, just a second loss mask.

## Stack (all verified real + public)
- **Trainer:** SkyRL (`skyrl/backends/skyrl_train`, FSDP) — GRPO native; ECHO = custom registered policy loss.
- **Rollouts:** `modal.Sandbox` (gVisor, stateful bash, verified) driven by a SkyRL-agent terminal env; Harbor task format.
- **Policy serving:** vLLM (OpenAI-compatible), co-located on the 8-GPU node.
- **Data:** endless-terminals + OpenThoughts-Agent-v1-RL (public Harbor tasks); base = OpenThinker-Agent-v1-SFT / Qwen3-8B / Qwen3-14B.
- **Eval:** Harbor + Terminus-2 on terminal-bench-2.0 (89 tasks), 5 rollouts, temp 0.6, seed 42, 1200s timeouts.

## GRPO recipe (paper §4 / App B)
n=16 rollouts/prompt, batch 16, lr 1e-6 const (AdamW β 0.9/0.95, wd 0.01), grad clip 0.2,
ε_lo=0.2 / ε_hi=0.28, no KL, prompt-level adv norm, sequence-level loss agg, rollout temp 0.8,
500 steps, ≤16 turns, 16k ctx, ≤2048 tok/turn, reward = unit-tests-pass.

## Phases
- **P0 — Smoke** (H100:1–2, ~$5–15): SkyRL trainer + vLLM + modal.Sandbox rollout + Volume checkpoint/resume; a few GRPO steps on ~20 tasks; one Terminus-2 TB2 task scores.
- **P1 — 8B A/B** (H100:8, ~$500–900 incl. debug): matched 500-step GRPO vs ECHO on Qwen3-8B (identical base/seed/harness). Reduced λ-sweep (3 values) as a wiring-correctness check. Eval both on TB2. **Delivers the ECHO>GRPO doubling.**
- **P2 — 14B + write-up** (B200:8 for faithful SKU, ~$600–1000): matched GRPO vs ECHO on Qwen3-14B (policy served tensor-parallel). Results report vs paper Table 1/4.

## Success criteria (honest)
- **PRIMARY (high confidence):** statistically-supported ECHO > GRPO on TB2 under identical conditions — target ~2× relative, the actual scientific claim. Robust to train-set reconstruction differences (both arms shift together).
- **SECONDARY (data-dependent):** land near the absolute numbers (5.17 / 10.79). These are 4–10 tasks out of 89 → statistically fragile and sensitive to the ~6170 private tasks we can't exactly reproduce. Reported best-effort, with the gap stated plainly.

## Known dependencies / risks
- ECHO released no code/weights/task-set → loss, O' subset, and 8770-task mix reconstructed from prose. Mitigate: validate GRPO baseline first; λ-sweep as correctness check; freeze eval harness independently.
- Chasing exact absolute numbers → full 6170-task regeneration needs an LLM gen+filter (paper used GPT-5). Would need an OpenAI/GPT-5 API key (not currently available). Deferred; not needed for the P1 A/B.
- 24h/function Modal timeout vs 24–48h runs → checkpoint to Volume + `--detach` + resume.
- 8×B200 scarcity → iterate on 8×H100, use B200 for faithful final runs.
