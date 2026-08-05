"""API probe — settles the project's riskiest assumption on a real GPU before the arms
are built on top of it.

Verifies, on the exact image and GPU the race will use, that:
  1. vLLM imports at the pinned version
  2. `worker_extension_cls` loads RaceWorkerExtension into every TP worker
  3. `collective_rpc` reaches it
  4. an `update_weights` call actually CHANGES the live model (proved via a weight
     fingerprint, so a silent no-op cannot masquerade as success)
  5. that update can happen WITHOUT stopping generation — the in-flight update the
     entire pipeline arm depends on
  6. sleep/wake really free and restore memory — the conventional arm depends on this

Self-contained by design: the image is defined in this file rather than imported, because
Modal does not auto-mount local Python sources and a cross-module import fails at container
start (learned the hard way: `ModuleNotFoundError: No module named 'image'`).

    modal run --detach runner/probe.py
"""
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "numpy<2.3")
    .env({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_V1": "1",
        # collective_rpc defaults to a SAFE msgpack serializer that cannot carry a CUDA-IPC
        # handle (it contains the torch.Tensor class object) and silently decodes tensors
        # into plain lists. Both failures were observed on GPU:
        #   "Object of type <class 'torch._C._TensorMeta'> is not serializable"
        #   "'list' object has no attribute 'to'"
        # vLLM's own error message names this flag as the remedy. "Insecure" here means the
        # RPC channel accepts pickled payloads — that channel is internal to a single-tenant
        # Modal container we control, so the threat model does not apply.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    })
    # The worker extension must be importable by dotted path inside EVERY vLLM TP worker
    # process, so it lands at /root (on sys.path) as a top-level module.
    .add_local_file(str(ROOT / "race" / "worker_ext.py"), "/root/race_worker_ext.py", copy=True)
)

app = modal.App("race-probe")
hf_cache = modal.Volume.from_name("race-hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@app.function(image=IMAGE, gpu="L4:2", timeout=60 * 40,
              volumes={"/root/.cache/huggingface": hf_cache})
def probe():
    import threading
    import time as _t

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    rep = {"vllm": vllm.__version__, "torch": torch.__version__,
           "gpus": torch.cuda.device_count()}
    print(f"[probe] vllm={vllm.__version__} torch={torch.__version__} "
          f"gpus={torch.cuda.device_count()}", flush=True)

    llm = LLM(model=MODEL, tensor_parallel_size=2, enable_sleep_mode=True,
              gpu_memory_utilization=0.55, max_model_len=1024,
              worker_extension_cls="race_worker_ext.RaceWorkerExtension",
              disable_log_stats=True)
    print("[probe] engine up", flush=True)

    # ---- 3: collective_rpc reaches the extension --------------------------
    try:
        sig0 = llm.collective_rpc("weight_signature")
        rep["collective_rpc"] = f"OK -> {sig0}"
        print(f"[probe] collective_rpc OK: {sig0}", flush=True)
    except Exception as e:
        rep["collective_rpc"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[probe] collective_rpc FAILED: {e}", flush=True)
        sig0 = None

    sp = SamplingParams(temperature=0.0, max_tokens=24)
    rep["gen_before"] = llm.generate(["What is 2+2? Answer:"], sp,
                                     use_tqdm=False)[0].outputs[0].text
    print(f"[probe] gen before: {rep['gen_before']!r}", flush=True)

    # ---- ORDERING MATTERS -------------------------------------------------
    # A raised exception inside collective_rpc POISONS the executor: every later RPC
    # returns the same error. Observed in v3, where sleep/wake "failed" with the CPU-stage
    # error text despite passing in v1 and v2 on an identical engine. So: run the
    # non-destructive checks first, and put anything that may throw at the end.

    # ---- A: sleep / wake (known-good, must not be masked) ------------------
    try:
        free0, _ = torch.cuda.mem_get_info(0)
        llm.sleep(level=1)
        free1, _ = torch.cuda.mem_get_info(0)
        llm.wake_up()
        free2, _ = torch.cuda.mem_get_info(0)
        rep["sleep_wake"] = (f"OK freed={(free1-free0)/2**30:.2f}GiB "
                             f"restored={(free1-free2)/2**30:.2f}GiB")
        print(f"[probe] {rep['sleep_wake']}", flush=True)
    except Exception as e:
        rep["sleep_wake"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[probe] sleep/wake FAILED: {e}", flush=True)

    from torch.multiprocessing.reductions import reduce_tensor
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    gpu_w = []
    for i, (n, p_) in enumerate(hf.named_parameters()):
        t = p_.detach().to(torch.bfloat16).cuda()
        if i < 3:
            t = t * 1.05        # perturb measurably so the fingerprint MUST move
        gpu_w.append((n, t.contiguous()))

    # ---- B: what does the worker actually receive? ------------------------
    try:
        probe_pair = [(gpu_w[0][0], gpu_w[0][1].cpu())]
        rep["payload_seen"] = llm.collective_rpc("inspect_payload", args=(probe_pair,))[0]
        print(f"[probe] worker sees: {rep['payload_seen']}", flush=True)
    except Exception as e:
        rep["payload_seen"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[probe] inspect FAILED: {e}", flush=True)

    # ---- C: CUDA IPC (the transport we actually want) ---------------------
    try:
        handles = [(n, reduce_tensor(t)[1]) for n, t in gpu_w]
        t_ipc = _t.monotonic()
        r = llm.collective_rpc("update_weights_ipc", args=(handles,))
        ipc_ms = (_t.monotonic() - t_ipc) * 1e3
        sig1 = llm.collective_rpc("weight_signature")
        rep["ipc"] = f"OK changed={sig1 != sig0} {r[0]} in {ipc_ms:.0f}ms"
        print(f"[probe] IPC OK changed={sig1 != sig0} {r[0]} {ipc_ms:.0f}ms", flush=True)
        ipc_ok = True
    except Exception as e:
        rep["ipc"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[probe] IPC FAILED: {e}", flush=True)
        ipc_ok = False

    # ---- D: THE decisive test — update mid-decode, KV cache intact --------
    if ipc_ok:
        try:
            result = {}

            def long_gen():
                lsp = SamplingParams(temperature=0.8, max_tokens=384)
                o = llm.generate(["Count slowly from 1 to 200, one number per line."],
                                 lsp, use_tqdm=False)
                result["ntok"] = len(o[0].outputs[0].token_ids)

            kv_before = llm.collective_rpc("kv_cache_fingerprint")
            th = threading.Thread(target=long_gen, daemon=True)
            th.start()
            _t.sleep(2.0)
            t_mid = _t.monotonic()
            handles = [(n, reduce_tensor(t)[1]) for n, t in gpu_w]
            llm.collective_rpc("update_weights_ipc", args=(handles,))
            mid_ms = (_t.monotonic() - t_mid) * 1e3
            th.join(timeout=300)
            kv_after = llm.collective_rpc("kv_cache_fingerprint")
            rep["inflight_update"] = (
                f"OK {result.get('ntok')} tokens, update {mid_ms:.0f}ms, "
                f"kv_blocks {kv_before} -> {kv_after} "
                f"({'RETAINED' if kv_before == kv_after else 'CHANGED'})")
            print(f"[probe] INFLIGHT: {rep['inflight_update']}", flush=True)
        except Exception as e:
            rep["inflight_update"] = f"FAIL {type(e).__name__}: {e}"
            print(f"[probe] inflight FAILED: {e}", flush=True)
    else:
        rep["inflight_update"] = "SKIPPED (ipc failed)"

    # ---- E: CPU staging LAST — it may poison the executor -----------------
    try:
        cpu_w = [(n, t.cpu()) for n, t in gpu_w]
        t_cpu = _t.monotonic()
        r = llm.collective_rpc("update_weights_cpu", args=(cpu_w,))
        rep["cpu_stage"] = f"OK {r[0]} in {(_t.monotonic()-t_cpu)*1e3:.0f}ms"
        print(f"[probe] CPU-stage OK {r[0]}", flush=True)
    except Exception as e:
        rep["cpu_stage"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[probe] CPU-stage FAILED: {e}", flush=True)

    print("\n===== PROBE REPORT =====", flush=True)
    for k, v in rep.items():
        print(f"  {k}: {v}", flush=True)
    print("===== END REPORT =====", flush=True)
    return rep


@app.local_entrypoint()
def main():
    r = probe.remote()
    print("\n--- returned ---")
    for k, v in r.items():
        print(f"{k}: {v}")
