"""Shared Modal training image: SkyRL FSDP (ECHO-patched) + terminal_env installed.

The base chain is byte-identical to modal/train_image.py so Modal reuses the cached `uv sync`
layer (no rebuild). We then drop the `terminal_env` package into the synced venv's site-packages
so `import terminal_env.terminal_env` resolves in the driver AND Ray workers.
"""
import modal
from pathlib import Path

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

# install terminal_env into the venv site-packages (importable everywhere, incl. Ray workers)
TRAIN_IMAGE = _base.add_local_dir(
    str(TERMINAL_ENV), f"{VENV_SITE}/terminal_env", copy=True,
    ignore=["**/__pycache__/**", "**/*.pyc", "test_*.py"],
)
