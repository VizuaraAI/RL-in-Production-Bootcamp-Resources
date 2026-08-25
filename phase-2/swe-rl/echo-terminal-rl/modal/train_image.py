"""Modal image for ECHO×SkyRL training (FSDP backend) + a patch-verification entrypoint.

Mirrors SkyRL's docker/Dockerfile (anyscale ray 2.56 / py3.12 / cu128) and installs SkyRL
with the fsdp + harbor + vllm extras from the locally-patched repo (ECHO diff applied).

Build (CPU is enough — no GPU needed to install or to run the patch-verification):
    modal run --detach modal/train_image.py::verify
"""
import modal
from pathlib import Path

REPO = Path(__file__).parent.parent / "upstream" / "SkyRL"   # patched SkyRL clone (ECHO applied in-tree)
REMOTE = "/workspace/SkyRL"

app = modal.App("echo-train")

VENV_PY = f"{REMOTE}/.venv/bin/python"

image = (
    # add_python gives Modal a known standalone entrypoint python (the base anyscale image's
    # anaconda python isn't auto-detected -> "unable to determine Python version"). It coexists
    # with the base python and the uv-synced training venv (which the function shells out to).
    modal.Image.from_registry("anyscale/ray:2.56.0-slim-py312-cu128", add_python="3.12")
    .apt_install("wget", "kmod", "libxml2", "build-essential", "libnuma-dev", "git", "curl")
    # uv pinned to SkyRL's version, installed to a FIXED system path (build runs as user
    # `ray`; a per-user ~/.local/bin path is not stable). UV_INSTALL_DIR fixes location.
    .run_commands(
        "curl -LsSf https://astral.sh/uv/0.9.4/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh",
        "/usr/local/bin/uv --version",
    )
    .env({"UV_LINK_MODE": "copy"})
    # copy the ECHO-patched SkyRL tree into the image (exclude VCS + local venvs)
    .add_local_dir(str(REPO), REMOTE, copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/__pycache__/**", "**/*.pyc"])
    # resolve + install SkyRL with the fsdp + harbor extras against its lockfile.
    # pipefail so a uv failure fails the build (the earlier `| tail` silently swallowed it).
    # heavy deps are prebuilt wheels (flash-attn SKIP_CUDA_BUILD=TRUE, vllm/flashinfer) -> no nvcc.
    .run_commands(
        f"bash -euo pipefail -c 'cd {REMOTE} && /usr/local/bin/uv sync --frozen --extra fsdp --extra harbor'",
        gpu=None,
    )
)


@app.function(image=image, timeout=3600)
def verify():
    """Confirm the ECHO patch loads inside the real training image (uses the synced venv)."""
    import subprocess, json, os
    results = {}
    results["venv_python_exists"] = os.path.exists(VENV_PY)
    # 1) EchoConfig is importable and wired onto AlgorithmConfig with the paper's defaults
    check = subprocess.run(
        [VENV_PY, "-c",
         "from skyrl.train.config.config import AlgorithmConfig, EchoConfig; "
         "a=AlgorithmConfig(); "
         "assert hasattr(a,'echo') and a.echo.enabled is False and abs(a.echo.lambda_-0.05)<1e-9; "
         "print('ECHO_CONFIG_OK', a.echo)"],
        cwd=REMOTE, capture_output=True, text=True,
    )
    results["config_check_rc"] = check.returncode
    results["config_check_out"] = (check.stdout + check.stderr)[-2500:]
    # 2) key training deps import (torch/vllm/flash_attn/ray) — proves the heavy stack installed
    deps = subprocess.run(
        [VENV_PY, "-c",
         "import torch,ray; print('torch',torch.__version__,'ray',ray.__version__); "
         "import vllm; print('vllm',vllm.__version__); "
         "import flash_attn; print('flash_attn',flash_attn.__version__)"],
        cwd=REMOTE, capture_output=True, text=True,
    )
    results["deps_check_rc"] = deps.returncode
    results["deps_check_out"] = (deps.stdout + deps.stderr)[-1500:]
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print(verify.remote())
