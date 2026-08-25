# Rollout + verifier cycle — VALIDATED on modal.Sandbox (2026-07-21)

`terminal_env/sandbox_task.py` proved the full terminal-task mechanic works, independent of SkyRL.

## Result (sample task et_0033979a, security-hardening)
- reference solve.sh → **reward 1.0** (5/5 pytest PASSED), empty agent → **reward 0.0**. setup_warnings [].

## The scalable design (chosen over per-task Docker builds)
`modal.Image.from_dockerfile` reads its build context at image-LOAD time, which happens inside
the remote worker where the local Dockerfile no longer exists → dynamic per-task builds fail; and
pre-building thousands of task images doesn't scale. SOLUTION:
- ONE static `BASE_IMAGE` (ubuntu:22.04 + python3/pip/pytest/curl/git/build tools), built once.
- At rollout: `modal.Sandbox.create(image=BASE_IMAGE)` then **replay the task's Dockerfile setup**
  (`parse_dockerfile_setup` handles RUN with `\`-continuations + heredocs, COPY/ADD by staging
  bytes, ENV, WORKDIR; skips FROM/LABEL/CMD). Reproduces initial state, zero per-task builds, fast.
- Verify: stage `tests/*` into `/tests/`, `mkdir -p /logs/verifier`, run `bash /tests/test.sh`
  (writes 1|0 to `/logs/verifier/reward.txt`), read it → binary reward.

## Caveats / TODO
- Tasks with non-ubuntu bases (FROM python:3.11 / node) may need base-image handling; the ubuntu
  base + apt/pip replay covers the majority. Record filtered-out tasks in provenance.gaps.
- `Sandbox.open()`/`FileIO.write()` are deprecated → migrate to `Sandbox.filesystem.write_bytes()`
  for reliability at scale when building the full TerminalEnv (works today, just noisy).
- Concurrency at scale: n=16 × batch 16 = up to 256 concurrent sandboxes (Modal supports 50k).

## Milestone status — all three CORE mechanics proven
1. ECHO loss: patch + unit tests + **verified loading inside the real training image**
   (`ECHO_CONFIG_OK EchoConfig(enabled=False, lambda_=0.05)`). ✅
2. Training image: SkyRL FSDP + Harbor + torch 2.11/ray 2.56/vllm 0.23/flash_attn 2.8.3. ✅
3. Rollout + verifier on modal.Sandbox (replay approach, reward 1/0 correct). ✅

## Next
- Wrap sandbox_task into `TerminalEnv(BaseTextEnv)` (action parsing + 16-turn loop + reward on done).
- Materialize corpus (endless-terminals + OpenThoughts-Agent-v1-RL) → SkyRL dataset (+val100).
- Wire SkyRLGymGenerator (use_conversation_multi_turn=true) + register `terminal` env.
- P0 smoke (confirm echo/obs_tokens>0) → P1 matched GRPO-vs-ECHO 500-step on Qwen3-8B.
