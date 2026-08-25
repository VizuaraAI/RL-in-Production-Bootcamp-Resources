# Rollout architecture decision (2026-07-21) — course-correction

## The trap we avoided
SkyRL's **`HarborGenerator` hard-requires step-wise mode** (`harbor_generator.py:244`:
`raise "HarborGenerator only supports step-wise training"`). In step-wise mode
(`build_step_wise_generator_output`), each training step = one assistant completion with
`step_loss_mask = [1]*len(comp_ids)`, and **observations become the NEXT step's prompt —
never inside any response span**. So `obs_mask = (response_mask & ~loss_mask)` would be
EMPTY → **ECHO's env-CE term is silently zero → ECHO ≈ GRPO.** Running the stock
`run_codecontest.sh` recipe with ECHO enabled would have produced a null result.

## The correct path: SkyRLGymGenerator + a custom `TerminalEnv`
`SkyRLGymGenerator` with `generator.use_conversation_multi_turn=true` builds the interleaved
`[action][obs][action][obs]...` sequence with observation tokens IN the response span
(`response_mask=1, loss_mask=0`) — exactly the structure the paper describes (§2) and exactly
what ECHO needs. This is the path the integration spec recommended.

Harbor + Terminus-2 is still used, but ONLY for **evaluation** on TerminalBench-2.0 (a separate,
ECHO-free harness). Training rollouts use our own `TerminalEnv` executing bash in `modal.Sandbox`.

## TerminalEnv spec (subclass `skyrl_gym.envs.base_text_env.BaseTextEnv`)
Template = `skyrl-gym/skyrl_gym/envs/search/env.py` (a tool-executing multi-turn env).
- `__init__(env_config, extras)` — `extras` carries per-task data: `{task_dir | docker_image,
  instruction, tests_cmd, max_turns}`. Register id `terminal` → `TerminalEnv`.
- `init(prompt)` — boot a `modal.Sandbox.create(image=from_registry(task_image))` (or build the
  task Dockerfile), return the initial prompt (system + instruction.md).
- `step(action)` — parse the bash command from the action (paper: Qwen XML bash block or a
  `task-done` signal); `sandbox.exec("bash","-c",cmd)`; return
  `observations=[{"role":"user","content":"<command_output>{stdout+stderr, exit}</command_output>"}]`,
  `reward=0`, `done=(task-done or turns>=16)`.
- on `done` — run the verifier (`tests/test.sh` / pytest) in the sandbox; `reward = 1.0` iff pass.
- `close()` — `sandbox.terminate()`.
- observation wrapped in `<command_output>…</command_output>` so the faithful O' mask
  (exclude warning prefixes, §3.2) can be added later via `env_output_mask` plumbing; v1 uses O'=O.

## Generator/config keys (dotted)
- `generator.use_conversation_multi_turn=true`  ← REQUIRED for ECHO (obs in-span)
- `generator.max_turns=16`, `generator.sampling_params.max_generate_length=2048`
- `environment.env_class=terminal` (per-row `env_class` in the dataset)
- per-task data → dataset row `env_extras` → `skyrl_gym.make(env_class, env_config, extras=env_extras)`
- ECHO: `trainer.algorithm.echo.enabled=true trainer.algorithm.echo.lambda_=0.05`
- GRPO recipe (from run_codecontest.sh + paper): `advantage_estimator=grpo`, `use_kl_loss=false`,
  `eps_clip_low=0.2 eps_clip_high=0.28`, `lr=1.0e-6`, `n_samples_per_prompt=16`,
  `loss_reduction=sequence_mean`, temp 0.8. Placement: `colocate_all=true`, fsdp, 8 policy GPUs,
  N inference engines (vLLM) with TP.

## Data → env mapping
Each training row: `{prompt: [system, {role:user, content:instruction}], env_class:"terminal",
env_extras:{task_dir, tests, docker_image, ...}}`. Built from the Harbor task dirs
(endless-terminals + OpenThoughts-Agent-v1-RL) verified in 01_ground_truth.md.
