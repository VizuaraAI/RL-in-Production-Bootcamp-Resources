# ECHO replication — verified ground truth (2026-07-21)

Every external dependency below was verified via the live HF/GitHub APIs (not inferred).

## Models (all ungated, Apache-2.0 / public)
| Model | HF id | Notes |
|---|---|---|
| SFT base (OT-SFT) | `open-thoughts/OpenThinker-Agent-v1-SFT` | Qwen3-8B finetune on ~15k GLM demos; 4 safetensors shards; ships chat_template.jinja |
| 8B base | `Qwen/Qwen3-8B` | 16.4M downloads |
| 14B base | `Qwen/Qwen3-14B` | 4.5M downloads |

## Training-task datasets (public, Harbor task format)
| Dataset | HF id | Count | Format |
|---|---|---|---|
| Endless Terminals | `obiwan96/endless-terminals` | ≥1000 (card: 3255; paper used 1977) | per-task DIR: `environment/Dockerfile`, `environment/container.def`, `instruction.md`, `solution/solve.sh`, `task.toml`, `tests/test.sh`+`tests/test_final_state.py` |
| OpenThoughts RL | `open-thoughts/OpenThoughts-Agent-v1-RL` | **728** (paper used 723) | parquet: rows = `{path, task_binary}`; `task_binary` = TARBALL of a Harbor task dir. Unpack via bundled `extract_parquet_tasks.py` |

## Eval benchmark
- `harborframework/terminal-bench-2.0` — **exactly 89 tasks** ✅ (matches paper). Each task = Harbor dir: `environment/Dockerfile`, `instruction.md`, `solution/solve.sh`, `task.toml`, `tests/test.sh`+`tests/test_outputs.py`. Some tasks ship `protected.tar.gz.enc` (encrypted test outputs, anti-gaming). Run via **Harbor + Terminus-2** agent.

## Harbor task schema (task.toml)
```toml
version = "0.1"
[metadata] author_name, difficulty (easy/med/hard), category, tags
[verifier] timeout_sec           # e.g. 300 (paper uses 120 for train verifier)
[agent]    timeout_sec           # e.g. 300 (paper uses 600 for train agent)
[environment] build_timeout_sec, cpus, memory_mb, storage_mb
```
`instruction.md` = the natural-language task the agent sees. `tests/` = the verifier (reward=1 iff pass).

## Chosen stack (verified real)
- **RL framework:** SkyRL (`NovaSky-AI/SkyRL`) — monorepo `skyrl/` unified lib. Native `AdvantageEstimator.GRPO` + `PolicyLossRegistry` with `@register_policy_loss` / `@register_advantage_estimator` decorators → **ECHO can be a registered custom loss, no core fork.** `skyrl-agent` advertises "Train your terminal-use agent!" via Harbor. FSDP/Megatron backends. Uses Ray + vLLM.
- **Rollout env:** modal.Sandbox (gVisor, stateful, verified) OR Harbor `--env modal`.
- **Reference integration to mirror:** `kanishkg/endless-terminals` (SkyRL + Harbor + Ray, same OT-SFT base).
- **Eval:** Harbor + Terminus-2 on terminal-bench-2.0.

## ECHO injection point (SkyRL)
`skyrl/backends/skyrl_train/utils/ppo_utils.py`:
- `PolicyLossType` enum already includes `CROSS_ENTROPY`; registry maps names→fns.
- Custom loss signature: `f(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None) -> (loss, metrics)`.
- OPEN QUESTION (integration agent resolving): are OBSERVATION-token logprobs available at the loss site, or only action-token logprobs? Determines whether ECHO is a pure registered loss (option A) or needs a trainer-step tweak + observation-mask plumbing (option B).
