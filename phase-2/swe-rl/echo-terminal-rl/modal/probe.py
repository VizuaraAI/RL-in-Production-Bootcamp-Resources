"""Feasibility probe for the ECHO replication on Modal.

Tests the two make-or-break capabilities:
  1. GPU allocation + torch CUDA on an H100 (the trainer/inference need this).
  2. modal.Sandbox: launching an isolated container to execute agent bash
     commands (this is how RL rollouts run terminal tasks).

Run:  modal run modal/probe.py
"""
import modal

app = modal.App("echo-probe")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch")
)


@app.function(gpu="H100", image=gpu_image, timeout=600)
def gpu_check():
    import subprocess, torch, json
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    x = torch.randn(4096, 4096, device="cuda")
    y = float((x @ x).sum().item())
    # return a JSON string so nothing torch-typed crosses the wire
    return json.dumps({
        "gpu": str(smi),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(torch.cuda.get_device_name(0)),
        "matmul_ok": bool(abs(y) > 0),
    })


@app.function(image=modal.Image.debian_slim(), timeout=600)
def sandbox_check():
    """Spin up a sandbox, run bash in it like a terminal-task rollout would."""
    sb = modal.Sandbox.create(
        image=modal.Image.from_registry("ubuntu:22.04"),
        app=modal.App.lookup("echo-probe", create_if_missing=True),
        timeout=120,
    )
    p = sb.exec("bash", "-c", "echo hello-from-sandbox; uname -a; python3 --version 2>&1 || echo 'no py'; ls /")
    out = p.stdout.read()
    p.wait()
    # second exec to confirm statefulness within the same sandbox
    sb.exec("bash", "-c", "echo persisted > /tmp/state.txt").wait()
    p2 = sb.exec("bash", "-c", "cat /tmp/state.txt")
    persisted = p2.stdout.read().strip()
    p2.wait()
    sb.terminate()
    return {"exec_output": out, "persisted_state": persisted, "returncode": p.returncode}


@app.local_entrypoint()
def main():
    print("=== GPU check (H100) ===")
    try:
        print(gpu_check.remote())
    except Exception as e:
        print("GPU check FAILED:", repr(e))
