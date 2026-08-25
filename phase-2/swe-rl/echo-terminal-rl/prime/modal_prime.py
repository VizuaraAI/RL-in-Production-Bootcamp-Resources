"""prime-rl on Modal — image build + debug ECHO validation, then the terminal-agent A/B.

Strategy (de-risk order):
  1. Build a lean prime-rl image (CUDA 12.8 devel + `uv sync --all-packages --extra flash-attn`,
     skipping the single-node-irrelevant extras: disagg/NIXL/UCX/gpt-oss/quack/mamba/flash-attn-3).
  2. `debug` — run prime-rl's OWN debug ECHO config (0.6B alphabet-sort, subprocess harness) on
     2xH100 to validate the whole async stack (inference+orchestrator+trainer) + ECHO works on Modal
     and SURVIVES the between-step gap (the exact phase that killed every SkyRL run). No corpus needed.
  3. (next file/fn) terminal-agent A/B on 8xH100 with HarborTaskset + BashHarness + Qwen3-8B.

    modal run --detach prime/modal_prime.py::debug
"""
import modal
from pathlib import Path

PRIME = Path(__file__).parent.parent / "upstream" / "prime-rl"
APP = "echo-prime"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "curl", "ca-certificates", "libnuma-dev", "clang",
                 "tmux", "pkg-config")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh",
        "/usr/local/bin/uv --version",
    )
    .env({"UV_LINK_MODE": "copy", "UV_PROJECT_ENVIRONMENT": "/app/.venv",
          "UV_PYTHON_PREFERENCE": "only-system", "UV_COMPILE_BYTECODE": "1",
          "CUDA_HOME": "/usr/local/cuda", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # bring in the prime-rl tree WITH submodules already checked out (deps/verifiers etc.)
    .add_local_dir(str(PRIME), "/app", copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/__pycache__/**", "**/*.pyc", "**/outputs/**"])
    .run_commands(
        # lean sync: base (torch/vllm) + flash-attn wheel + ALL workspace packages (envs/tasksets).
        # --frozen installs exactly from uv.lock (submodule editable pkgs use their fallback-version
        # 0.0.0 when there's no git metadata) and, unlike --locked, does NOT error on pyproject drift.
        # No tail mask -> the full error streams to the build log. No GPU needed (prebuilt wheels).
        "cd /app && /usr/local/bin/uv sync --all-packages --extra flash-attn --frozen --no-dev",
        # `harbor` is a verifiers extra (not a prime-rl root extra) -> install the CLI into the venv.
        "/usr/local/bin/uv pip install --python /app/.venv/bin/python 'harbor==0.14.0'",
        "/app/.venv/bin/python -c \"import vllm, torch, verifiers; print('vllm', vllm.__version__, 'torch', torch.__version__)\"",
    )
    .env({"WANDB_MODE": "offline"})  # no wandb creds needed for validation
)

app = modal.App(APP)
out_vol = modal.Volume.from_name("echo-prime-out", create_if_missing=True)
hf_vol = modal.Volume.from_name("echo-models", create_if_missing=True)


@app.function(image=image, gpu="H100:2", timeout=2 * 3600,
              volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol},
              cpu=16, memory=96 * 1024)
def debug():
    """Validate prime-rl + ECHO end-to-end on Modal with the shipped debug config (2xH100)."""
    import subprocess, os
    # `rl` spawns `inference`/`orchestrator`/`trainer` console-scripts via PATH — must include the venv bin
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": "/root/.cache/huggingface",
           "WANDB_MODE": "offline",
           "PATH": f"/app/.venv/bin:{os.environ.get('PATH', '')}"}
    cmd = ["/app/.venv/bin/rl", "@", "configs/debug/algo/echo.toml",
           "--output-dir", "/out/echo-debug-v2",
           # H100=SM90 auto-picks flash_attention_3 which we didn't install; force FA2 (we have the wheel)
           "--trainer.model.attn", "flash_attention_2",
           "--max-steps", "3"]  # short: validate ECHO (nll metric) within one connected window
    print("LAUNCH:", " ".join(cmd))
    p = subprocess.run(cmd, cwd="/app", env=env)
    out_vol.commit()
    print("debug rl exit code:", p.returncode)
    return p.returncode


TERMINAL_CFG = """
max_steps = 3
seq_len = 24576
[model]
name = "Qwen/Qwen3-1.7B"
[orchestrator]
batch_size = 4
group_size = 2
[orchestrator.algo]
type = "echo"
[orchestrator.algo.roles.tool]
alpha = 0.05
# KEEP zero-advantage rollouts: when the small model solves nothing (all reward 0), GRPO has no
# signal, but ECHO still trains on the env-observation tokens — this is ECHO's whole point, and it
# lets the run proceed (non-empty batches) instead of aborting.
[[orchestrator.post_batch_filters]]
type = "zero_advantage"
enforce = false
[[orchestrator.train.env]]
name = "terminal"
[orchestrator.train.env.env.taskset]
id = "echoterm"
parquet = "/data/corpus/full/train.parquet"
limit = 4
[orchestrator.train.env.env.agent.harness]
id = "bash"
[orchestrator.train.env.env.agent.harness.runtime]
type = "modal"
image = "ubuntu:22.04"
# Qwen3 emits a <think> block first; at 1024 tokens it was truncated (finish_reason=length)
# MID-REASONING, before ever emitting the bash tool call -> no tool observation -> ECHO CE=0.
# Give it room to finish thinking AND emit the call; lower temp for cleaner tool-call JSON.
[orchestrator.train.sampling]
temperature = 0.6
max_completion_tokens = 6144
[orchestrator.renderer]
name = "prime-qwen3"
[trainer.optim]
lr = 1.0e-6
[inference]
gpu_memory_utilization = 0.5
"""


data_vol = modal.Volume.from_name("echo-data", create_if_missing=True)


@app.function(image=image, gpu="H100:2", timeout=2 * 3600,
              volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("prime-api")],  # PRIME_API_KEY for the sandbox->host tunnel
              cpu=16, memory=96 * 1024)
def terminal():
    """First terminal-agent ECHO test: harbor taskset (hello-world) + bash harness + modal runtime."""
    import subprocess, os
    open("/tmp/term.toml", "w").write(TERMINAL_CFG)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": "/root/.cache/huggingface",
           "WANDB_MODE": "offline", "PATH": f"/app/.venv/bin:{os.environ.get('PATH', '')}"}
    cmd = ["/app/.venv/bin/rl", "@", "/tmp/term.toml", "--output-dir", "/out/term-debug",
           "--trainer.model.attn", "flash_attention_2"]
    print("LAUNCH:", " ".join(cmd))
    p = subprocess.run(cmd, cwd="/app", env=env)
    out_vol.commit()
    print("terminal rl exit code:", p.returncode)
    return p.returncode


def _ab_config(echo: bool, steps: int, batch: int, group: int, limit: int, ckpt: int,
               num_infer: int = 4, num_train: int = 4) -> str:
    algo = ('[orchestrator.algo]\ntype = "echo"\n[orchestrator.algo.roles.tool]\nalpha = 0.05\n'
            if echo else '[orchestrator.algo]\ntype = "grpo"\n')
    lim = f"limit = {limit}\n" if limit else ""
    return f"""
max_steps = {steps}
seq_len = 16384
[model]
name = "Qwen/Qwen3-8B"
[inference.model]
max_model_len = 16384
[deployment]
num_infer_gpus = {num_infer}
num_train_gpus = {num_train}
[ckpt]
interval = {ckpt}
[orchestrator]
batch_size = {batch}
group_size = {group}
max_inflight_rollouts = 256
{algo}
[[orchestrator.post_batch_filters]]
type = "zero_advantage"
enforce = false
[orchestrator.train.sampling]
temperature = 0.8
# Qwen3 thinking-block room: 1024 truncated mid-<think> before the tool call (ECHO CE=0). 8B
# reasons more concisely than 1.7B (which needed 6144), but keep generous headroom so the model
# always finishes thinking AND emits the bash tool call -> real tool observations for ECHO.
max_completion_tokens = 4096
[[orchestrator.train.env]]
name = "terminal"
[orchestrator.train.env.env.taskset]
id = "echoterm"
parquet = "/data/corpus/full/train.parquet"
{lim}[orchestrator.train.env.env.agent.harness]
id = "bash"
[orchestrator.train.env.env.agent.harness.runtime]
type = "modal"
image = "ubuntu:22.04"
[orchestrator.renderer]
name = "prime-qwen3"
[trainer.optim]
lr = 1.0e-6
[trainer.model]
attn = "flash_attention_2"
[inference]
gpu_memory_utilization = 0.7
"""


model_vol = modal.Volume.from_name("echo-models", create_if_missing=True)


@app.function(image=image, gpu="H100:8", timeout=24 * 3600,
              volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("prime-api")], cpu=32, memory=256 * 1024)
def train8ab(echo: bool = True, quick: bool = False):
    """Qwen3-8B matched GRPO (echo=false) / ECHO (echo=true) on my terminal corpus (echoterm)."""
    import subprocess, os
    name = ("echo8b" if echo else "grpo8b") + ("-q" if quick else "")
    if quick:
        cfg = _ab_config(echo, steps=3, batch=8, group=8, limit=24, ckpt=1000)
    else:
        cfg = _ab_config(echo, steps=500, batch=128, group=16, limit=0, ckpt=25)
    path = f"/tmp/{name}.toml"
    open(path, "w").write(cfg)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": "/root/.cache/huggingface",
           "WANDB_MODE": "offline", "PATH": f"/app/.venv/bin:{os.environ.get('PATH', '')}"}
    cmd = ["/app/.venv/bin/rl", "@", path, "--output-dir", f"/out/{name}"]
    print("LAUNCH:", " ".join(cmd), "| echo=", echo, "quick=", quick)
    p = subprocess.run(cmd, cwd="/app", env=env)
    out_vol.commit()
    print(f"{name} rl exit code:", p.returncode)
    return p.returncode


def _ab_small_config(echo: bool, steps: int, batch: int, group: int, train_limit: int,
                     eval_n: int, max_turns: int, lr: float, model: str = "Qwen/Qwen3-1.7B",
                     deploy: str = "", max_inflight: int = 0, gpu_mem: float = 0.5,
                     seq_len: int = 24576, eval_interval: int = 0, ckpt_interval: int = 0) -> str:
    """Matched A/B config: IDENTICAL for both arms except the algorithm block.
    Both keep zero-advantage rollouts (enforce=false) so the ONLY difference is whether the
    env-observation tokens receive ECHO's alpha*CE (echo) or not (grpo) — isolating ECHO's effect.
    Startup + final eval on the held-out validation split gives a pass@1 per arm.
    `deploy` optionally injects a [deployment] block. Keep num_infer_gpus=1 (the proven-good
    inference setup); scale num_train_gpus for a bigger model's optimizer memory. num_infer_gpus>1
    mismatches the default inference parallel size and hangs /resume (learned the hard way).
    `max_inflight` caps concurrent rollouts so a big eval can't saturate the single inference GPU."""
    algo = ('[orchestrator.algo]\ntype = "echo"\n[orchestrator.algo.roles.tool]\nalpha = 0.05\n'
            if echo else '[orchestrator.algo]\ntype = "grpo"\n')
    mif = f"max_inflight_rollouts = {max_inflight}\n" if max_inflight else ""
    ckpt = f"[ckpt]\ninterval = {ckpt_interval}\n" if ckpt_interval else ""
    return f"""
max_steps = {steps}
seq_len = {seq_len}
[model]
name = "{model}"
{deploy}{ckpt}[orchestrator]
batch_size = {batch}
group_size = {group}
{mif}{algo}[[orchestrator.post_batch_filters]]
type = "zero_advantage"
enforce = false
[orchestrator.train.sampling]
temperature = 0.6
max_completion_tokens = 6144
[[orchestrator.train.env]]
name = "terminal"
[orchestrator.train.env.env.taskset]
id = "echoterm"
parquet = "/data/corpus/full/train.parquet"
limit = {train_limit}
[orchestrator.train.env.env.agent]
max_turns = {max_turns}
[orchestrator.train.env.env.agent.harness]
id = "bash"
[orchestrator.train.env.env.agent.harness.runtime]
type = "modal"
image = "ubuntu:22.04"
[orchestrator.eval]
interval = {eval_interval or steps}
num_examples = {eval_n}
group_size = 4
[orchestrator.eval.sampling]
temperature = 0.3
max_completion_tokens = 6144
[[orchestrator.eval.env]]
name = "terminal_val"
[orchestrator.eval.env.env.taskset]
id = "echoterm"
parquet = "/data/corpus/full/validation.parquet"
limit = {eval_n}
[orchestrator.eval.env.env.agent]
max_turns = {max_turns}
[orchestrator.eval.env.env.agent.harness]
id = "bash"
[orchestrator.eval.env.env.agent.harness.runtime]
type = "modal"
image = "ubuntu:22.04"
[orchestrator.renderer]
name = "prime-qwen3"
[trainer.optim]
lr = {lr}
[inference]
gpu_memory_utilization = {gpu_mem}
"""


_AB_MODELS = {
    "1p7b": ("Qwen/Qwen3-1.7B", "H100:2", ""),
    # 4B: inference stays on 1 GPU (proven-good, avoids the /resume hang); training gets 3 GPUs
    # (FSDP) for the optimizer memory. Total 4 = H100:4.
    "4b":   ("Qwen/Qwen3-4B", "H100:4",
             "[deployment]\nnum_infer_gpus = 1\nnum_train_gpus = 3\n"),
}


@app.function(image=image, gpu="H100:2", timeout=8 * 3600,
              volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("prime-api")], cpu=16, memory=96 * 1024)
def train_ab_small(echo: bool = True, steps: int = 30, size: str = "1p7b"):
    """Matched GRPO (echo=false) / ECHO (echo=true) A/B on echoterm, with held-out eval."""
    import subprocess, os
    model, _gpu, deploy = _AB_MODELS[size]
    name = f"ab-{'echo' if echo else 'grpo'}-{size}"
    cfg = _ab_small_config(echo, steps=steps, batch=24, group=6, train_limit=128,
                           eval_n=16, max_turns=10, lr=1.0e-6, model=model, deploy=deploy)
    path = f"/tmp/{name}.toml"
    open(path, "w").write(cfg)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": "/root/.cache/huggingface",
           "WANDB_MODE": "offline", "PATH": f"/app/.venv/bin:{os.environ.get('PATH', '')}"}
    cmd = ["/app/.venv/bin/rl", "@", path, "--output-dir", f"/out/{name}",
           "--trainer.model.attn", "flash_attention_2"]
    print("LAUNCH:", " ".join(cmd), "| echo=", echo, "steps=", steps, "size=", size)
    p = subprocess.run(cmd, cwd="/app", env=env)
    out_vol.commit()
    print(f"{name} rl exit code:", p.returncode)
    return p.returncode


# gpu is fixed per-Function at decoration time, so bigger sizes get their own wrappers.
@app.function(image=image, gpu="H100:4", timeout=18 * 3600,
              volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol, "/data": data_vol},
              secrets=[modal.Secret.from_name("prime-api")], cpu=32, memory=192 * 1024)
def train_ab_4b(echo: bool = True, steps: int = 300):
    """Qwen3-4B matched A/B (4xH100: 1 infer + 3 train). Long run to see if the ECHO>GRPO accuracy
    gap emerges: intermediate evals every ~1/5 of the run + checkpoints as insurance for the ~6h run."""
    import subprocess, os
    model, _gpu, deploy = _AB_MODELS["4b"]
    name = f"ab-{'echo' if echo else 'grpo'}-4b"
    eval_every = max(1, steps // 5)  # ~5-6 held-out evals across the run to trace the trajectory
    # ckpt disabled: large 4B ckpts (weights+optimizer) risk filling the volume and crashing the
    # run they'd protect; .spawn() is robust enough. Relaunch on the rare failure.
    cfg = _ab_small_config(echo, steps=steps, batch=24, group=6, train_limit=128,
                           eval_n=48, max_turns=10, lr=1.0e-6, model=model, deploy=deploy,
                           max_inflight=32, gpu_mem=0.8, eval_interval=eval_every)
    path = f"/tmp/{name}.toml"
    open(path, "w").write(cfg)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": "/root/.cache/huggingface",
           "WANDB_MODE": "offline", "PATH": f"/app/.venv/bin:{os.environ.get('PATH', '')}"}
    cmd = ["/app/.venv/bin/rl", "@", path, "--output-dir", f"/out/{name}",
           "--trainer.model.attn", "flash_attention_2"]
    print("LAUNCH:", " ".join(cmd), "| echo=", echo, "steps=", steps, "size=4b")
    import threading
    # Incremental commits: prime-rl only writes results at the end, so an end-of-run hang (e.g. one
    # eval rollout blocking forever) previously lost ALL data. Committing every ~150s means a late
    # hang costs at most the final eval — and intermediate evals/logs become readable live mid-run.
    stop = threading.Event()
    def _committer():
        while not stop.wait(150):
            try:
                out_vol.commit()
            except Exception as e:
                print("periodic commit error:", e)
    th = threading.Thread(target=_committer, daemon=True)
    th.start()
    p = subprocess.Popen(cmd, cwd="/app", env=env)
    rc = p.wait()
    stop.set()
    out_vol.commit()
    print(f"{name} rl exit code:", rc)
    return rc


@app.local_entrypoint()
def ab_small_echo(steps: int = 30):
    print(train_ab_small.remote(echo=True, steps=steps))


@app.local_entrypoint()
def ab_small_grpo(steps: int = 30):
    print(train_ab_small.remote(echo=False, steps=steps))


@app.local_entrypoint()
def ab4b_echo(steps: int = 60):
    print(train_ab_4b.remote(echo=True, steps=steps))


@app.local_entrypoint()
def ab4b_grpo(steps: int = 60):
    print(train_ab_4b.remote(echo=False, steps=steps))


@app.local_entrypoint()
def ab4b_echo_spawn(steps: int = 300):
    """Re-run only the ECHO arm (GRPO already finished as the control) — fire-and-forget so it
    survives client death; the function now commits incrementally so a late hang can't lose data."""
    e = train_ab_4b.spawn(echo=True, steps=steps)
    print("SPAWNED_ECHO", e.object_id)


@app.local_entrypoint()
def ab4b_both(steps: int = 300):
    """Fire-and-forget BOTH 4B arms via .spawn() so they survive client death.
    Unlike .remote() (streams; an abnormal client death cancels the server-side run), .spawn()
    submits and returns immediately — the client exits cleanly and, with `modal run --detach`, the
    spawned functions keep running server-side. Poll `modal app logs <app-id>` for progress."""
    e = train_ab_4b.spawn(echo=True, steps=steps)
    g = train_ab_4b.spawn(echo=False, steps=steps)
    print("SPAWNED_ECHO", e.object_id)
    print("SPAWNED_GRPO", g.object_id)


@app.local_entrypoint()
def main():
    print(debug.remote())


@app.local_entrypoint()
def q8b_echo():
    print(train8ab.remote(echo=True, quick=True))


@app.local_entrypoint()
def run_terminal():
    print(terminal.remote())
