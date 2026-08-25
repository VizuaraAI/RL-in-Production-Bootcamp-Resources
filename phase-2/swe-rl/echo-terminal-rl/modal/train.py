"""ECHO training on Modal — smoke + full runs of the SkyRL GRPO/ECHO terminal-agent trainer.

P0 SMOKE (validate the whole loop + confirm echo/obs_tokens>0), 2xH100, tiny model, ~2 steps:
    modal run --detach modal/train.py::smoke

The trainer creates nested modal.Sandbox instances for rollouts (works from inside a Modal
function — verified). ECHO on: trainer.algorithm.echo.enabled=true lambda_=0.05.
"""
import modal
from pathlib import Path

# ---- training image (inlined so the entrypoint is self-contained for Modal's remote) ----
REPO = Path(__file__).parent.parent / "upstream" / "SkyRL"
TERMINAL_ENV = Path(__file__).parent.parent / "terminal_env"
REMOTE = "/workspace/SkyRL"
VENV_SITE = f"{REMOTE}/.venv/lib/python3.12/site-packages"
VENV_PY = f"{REMOTE}/.venv/bin/python"

_base = (
    modal.Image.from_registry("anyscale/ray:2.56.0-slim-py312-cu128", add_python="3.12")
    .apt_install("wget", "kmod", "libxml2", "build-essential", "libnuma-dev", "git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/0.9.4/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh",
        "/usr/local/bin/uv --version",
    )
    .env({"UV_LINK_MODE": "copy"})
    .add_local_dir(str(REPO), REMOTE, copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/__pycache__/**", "**/*.pyc"])
    .run_commands(
        f"bash -euo pipefail -c 'cd {REMOTE} && /usr/local/bin/uv sync --frozen --extra fsdp --extra harbor'",
        gpu=None,
    )
)
TRAIN_IMAGE = _base.add_local_dir(
    str(TERMINAL_ENV), f"{VENV_SITE}/terminal_env", copy=True,
    ignore=["**/__pycache__/**", "**/*.pyc", "test_*.py"],
).env({
    # CONTAINER-LEVEL disarm of the PyTorch NCCL heartbeat watchdog — reaches EVERY process
    # (training FSDP ranks, weight-sync group, AND vLLM engine workers), unlike driver-env or the
    # inference-only runtime_env. The watchdog SIGABRT-kills whichever NCCL process group sits idle
    # >600s during the long between-step rollout phase (256 sandbox episodes + grading). This is the
    # comprehensive fix after driver-env and engine-actor runtime_env both failed to propagate.
    "TORCH_NCCL_ENABLE_MONITORING": "0",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "14400",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
    "NCCL_TIMEOUT": "2400",
    "VLLM_RPC_TIMEOUT": "1800000",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "1800",
})

app = modal.App("echo-train-run")
data_vol = modal.Volume.from_name("echo-data", create_if_missing=True)
model_vol = modal.Volume.from_name("echo-models", create_if_missing=True)


def _download_model(hf_id: str, dest: str):
    import subprocess, os
    if os.path.exists(os.path.join(dest, "config.json")):
        print(f"model cached: {dest}")
        return
    print(f"downloading {hf_id} -> {dest}")
    subprocess.run(
        [VENV_PY, "-c",
         "import os; from huggingface_hub import snapshot_download; "
         f"snapshot_download('{hf_id}', local_dir='{dest}', "
         "ignore_patterns=['*.pt','*.bin','original/*'])"],
        check=True, env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )


def _run_training(overrides: list, run_name: str):
    import subprocess, os
    # call the venv python directly (Ray workers inherit sys.executable = this venv);
    # avoids `uv run` re-syncing and pruning the manually-installed terminal_env.
    cmd = ([VENV_PY, "-m", "terminal_env.main_terminal"] + overrides)
    # PRIMARY STABILITY FIX (root cause diagnosed from crash logs): vLLM V1 engine workers were
    # SIGABRT-killed by the PyTorch NCCL heartbeat watchdog during the LONG idle gap between rollout
    # steps (256 sandbox bash episodes + grading + weight-sync run with no generation for ~15+ min,
    # exceeding the 600s TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC default -> silent hard kill, no traceback;
    # matches vllm #42742/#35104). These env vars are set on the DRIVER process so Ray captures them
    # into the runtime_env inherited by the EngineCore/RayWorkerProc actors.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "TORCH_NCCL_ENABLE_MONITORING": "0",        # disarm the SIGABRT idle watchdog (the fix)
           "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "14400",  # 4h safety net if monitoring re-enables
           "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
           "NCCL_TIMEOUT": "2400",
           "VLLM_RPC_TIMEOUT": "1800000",              # 30min RPC timeout (covers long idle/weight-sync)
           "VLLM_ENGINE_ITERATION_TIMEOUT_S": "1800"}
    print("LAUNCH:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=REMOTE, env=env)
    print(f"training [{run_name}] exit code:", p.returncode)
    return p.returncode


SMOKE_MODEL = "Qwen/Qwen3-1.7B"


@app.function(image=TRAIN_IMAGE, gpu="H100:2", volumes={"/data": data_vol, "/models": model_vol},
              timeout=3 * 3600, cpu=8, memory=64 * 1024)
def smoke(echo: bool = True):
    model_dir = "/models/Qwen3-1.7B"
    _download_model(SMOKE_MODEL, model_dir)
    model_vol.commit()
    ov = [
        # data: use the tiny 8-task val split as train so the smoke is ~2 steps
        "data.train_data=['/data/corpus/smoke/validation.parquet']",
        "data.val_data=['/data/corpus/smoke/validation.parquet']",
        f"trainer.policy.model.path={model_dir}",
        "environment.env_class=terminal",
        # GRPO recipe (paper)
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.algorithm.use_kl_loss=false",
        "trainer.algorithm.eps_clip_low=0.2",
        "trainer.algorithm.eps_clip_high=0.28",
        "trainer.algorithm.loss_reduction=sequence_mean",
        "trainer.policy.optimizer_config.lr=1.0e-6",
        # ECHO
        f"trainer.algorithm.echo.enabled={'true' if echo else 'false'}",
        "trainer.algorithm.echo.lambda_=0.05",
        # placement / infra (2 GPUs, colocated policy+vLLM)
        "trainer.strategy=fsdp",
        "trainer.placement.colocate_all=true",
        "trainer.placement.policy_num_gpus_per_node=2",
        "trainer.placement.ref_num_gpus_per_node=2",
        # colocate_all requires num_rollout_gpus (num_engines*tp) == num_policy_gpus (2)
        "generator.inference_engine.num_engines=2",
        "generator.inference_engine.tensor_parallel_size=1",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.weight_sync_backend=nccl",
        "generator.inference_engine.gpu_memory_utilization=0.55",
        # multi-turn terminal rollouts (ECHO needs obs tokens in-span)
        "generator.use_conversation_multi_turn=true",
        "generator.batched=false",
        "generator.max_turns=6",
        "generator.sampling_params.max_generate_length=1024",
        "generator.sampling_params.temperature=0.8",
        "generator.n_samples_per_prompt=4",
        # tiny batch -> ~2 steps
        "trainer.train_batch_size=4",
        "trainer.policy_mini_batch_size=4",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.epochs=1",
        "trainer.eval_before_train=false",
        "trainer.eval_interval=1000",
        "trainer.ckpt_interval=1000",
        "trainer.max_prompt_length=4096",
        "trainer.algorithm.max_seq_len=16384",
        "trainer.logger=console",
        "trainer.project_name=echo",
        "trainer.run_name=smoke",
        "trainer.resume_mode=null",
        "trainer.ckpt_path=/data/ckpts/smoke",
        "trainer.log_path=/data/logs/smoke",
    ]
    rc = _run_training(ov, "smoke")
    data_vol.commit()
    return rc


# ============================ P1/P2: real 8-GPU runs ============================
@app.function(image=TRAIN_IMAGE, gpu="H100:8", volumes={"/data": data_vol, "/models": model_vol},
              timeout=24 * 3600, cpu=32, memory=320 * 1024)
def train8(echo: bool = True, run_name: str = "echo8b", model: str = "Qwen/Qwen3-8B",
           corpus: str = "full", n_samples: int = 16, batch: int = 16, epochs: int = 8,
           max_turns: int = 16, ckpt_interval: int = 10, resume: bool = True,
           num_engines: int = 4, tp: int = 1, gpu_mem: float = 0.85,
           lam: float = 0.05, max_gen: int = 2048, max_seq: int = 16384,
           quick: bool = False):
    # `quick`: fast 8B/8-GPU config validation (tiny data, a couple steps, no ckpt/eval)
    if quick:
        corpus, batch, n_samples, max_turns, epochs = "smoke", 4, 4, 6, 1
        ckpt_interval, resume = 1000, False
    """Matched GRPO (echo=false) / ECHO (echo=true) run on 8xH100. Checkpoints to the Volume;
    resume=true continues from the latest checkpoint (survives Modal's 24h function timeout —
    just re-invoke to continue). Mirrors the paper's GRPO recipe (n=16, batch=16, lr 1e-6,
    eps 0.2/0.28, no KL, temp 0.8, seq-level loss)."""
    model_dir = f"/models/{model.split('/')[-1]}"
    _download_model(model, model_dir)
    model_vol.commit()
    ov = [
        f"data.train_data=['/data/corpus/{corpus}/train.parquet']",
        f"data.val_data=['/data/corpus/{corpus}/validation.parquet']",
        f"trainer.policy.model.path={model_dir}",
        "environment.env_class=terminal",
        # --- GRPO recipe (paper §4 / App B) ---
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.algorithm.use_kl_loss=false",
        "trainer.algorithm.eps_clip_low=0.2",
        "trainer.algorithm.eps_clip_high=0.28",
        "trainer.algorithm.loss_reduction=sequence_mean",
        "trainer.policy.optimizer_config.lr=1.0e-6",
        # --- ECHO ---
        f"trainer.algorithm.echo.enabled={'true' if echo else 'false'}",
        f"trainer.algorithm.echo.lambda_={lam}",
        # --- infra: 8 GPUs, DISAGGREGATED (SkyRL-recommended for agentic RL) ---
        # policy FSDP on 4 GPUs + vLLM on the other 4. Non-colocated removes the weight-sync
        # memory contention (both weight sets on GPU) that was killing an engine ~step 3-4, and
        # enables async training (ideal for long-tail multi-turn rollouts). See inference.md.
        "trainer.strategy=fsdp",
        "trainer.placement.colocate_all=false",
        "trainer.placement.policy_num_gpus_per_node=4",
        "trainer.placement.ref_num_gpus_per_node=4",
        f"generator.inference_engine.num_engines={num_engines}",
        f"generator.inference_engine.tensor_parallel_size={tp}",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.weight_sync_backend=nccl",
        f"generator.inference_engine.gpu_memory_utilization={gpu_mem}",
        "generator.inference_engine.enforce_eager=true",  # avoid torch.compile/cudagraph engine deaths
        # TARGETED FIX: every crash traceback goes through ray_executor_v2.py "RayWorkerProc died".
        # Switch vLLM to its multiprocessing executor (not Ray) to remove that failing layer entirely.
        # KV usage was only 34% at crash -> NOT OOM, so this (not memory) is the through-line.
        "generator.inference_engine.distributed_executor_backend=mp",
        "generator.inference_engine.vllm_v1_disable_multiproc=false",  # required for the mp executor
        "generator.inference_engine.max_num_seqs=64",                  # mild concurrency cap
        "generator.inference_engine.use_expandable_segments=true",
        # (max_env_workers already defaults to 32; no override needed)
        # cap context to the paper's 16k training window (Qwen3-8B native is 40960 -> KV-cache OOM
        # under colocation with 500 concurrent 16-turn eval episodes; 16k slashes KV memory).
        f"generator.inference_engine.engine_init_kwargs.max_model_len={max_seq}",
        # --- multi-turn terminal rollouts (obs in-span for ECHO) ---
        "generator.use_conversation_multi_turn=true",
        "generator.batched=false",
        f"generator.max_turns={max_turns}",
        f"generator.sampling_params.max_generate_length={max_gen}",
        "generator.sampling_params.temperature=0.8",
        f"generator.n_samples_per_prompt={n_samples}",
        "generator.eval_n_samples_per_prompt=5",
        "generator.eval_sampling_params.temperature=0.6",
        "generator.apply_overlong_filtering=true",
        # --- batch / steps ---
        f"trainer.train_batch_size={batch}",
        f"trainer.policy_mini_batch_size={batch}",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.micro_train_batch_size_per_gpu=1",
        f"trainer.epochs={epochs}",
        "trainer.update_epochs_per_batch=1",
        "trainer.eval_before_train=false",  # skip heavy pre-train val100 eval; reach a checkpoint first
        "trainer.eval_interval=50",
        f"trainer.ckpt_interval={ckpt_interval}",
        "trainer.hf_save_interval=50",
        "trainer.max_ckpts_to_keep=5",
        "trainer.max_prompt_length=4096",
        f"trainer.algorithm.max_seq_len={max_seq}",
        "trainer.logger=console",
        "trainer.project_name=echo",
        f"trainer.run_name={run_name}",
        f"trainer.resume_mode={'latest' if resume else 'null'}",
        f"trainer.ckpt_path=/data/ckpts/{run_name}",
        f"trainer.export_path=/data/exports/{run_name}",
        f"trainer.log_path=/data/logs/{run_name}",
    ]
    rc = _run_training(ov, run_name)
    data_vol.commit()
    return rc
