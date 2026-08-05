"""Probe 2 — does vLLM 0.11.0's AsyncLLM support the REAL in-flight weight update?

Modelled directly on ServiceNow/pipelinerl's `WeightUpdateManager.receive_weight_update`,
which does exactly this:

    await engine.pause_generation(mode="keep", clear_cache=False)
    await engine_client.collective_rpc_async("receive_weight_update", ...)
    await engine.resume_generation()

`mode="keep", clear_cache=False` is the whole trick: the engine stops between STEPS, not
between batches, and in-flight requests plus their KV cache survive. Sequences then continue
decoding under the new weights — a genuinely mixed-policy sequence.

This settles three things before any rewrite:
  1. do `pause_generation` / `resume_generation` exist on AsyncLLM in 0.11.0?
  2. does `collective_rpc_async` reach our worker extension?
  3. does a LONG generation started before the update still complete, with the KV block
     count unchanged across it?

    modal run --detach runner/probe_async.py
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
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    })
    .add_local_file(str(ROOT / "race" / "worker_ext.py"), "/root/race_worker_ext.py", copy=True)
)

app = modal.App("race-probe-async")
hf_cache = modal.Volume.from_name("race-hf-cache", create_if_missing=True)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@app.function(image=IMAGE, gpu="L4:1", timeout=60 * 40,
              volumes={"/root/.cache/huggingface": hf_cache})
def probe():
    import asyncio
    import inspect

    import torch
    import vllm

    rep = {"vllm": vllm.__version__}

    # ---- 1: does the API even exist? (free, no GPU work) ------------------
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        for name in ("pause_generation", "resume_generation"):
            fn = getattr(AsyncLLM, name, None)
            if fn is None:
                rep[name] = "MISSING"
            else:
                rep[name] = f"present{inspect.signature(fn)}"
            print(f"[probe] AsyncLLM.{name}: {rep[name]}", flush=True)
        from vllm.v1.engine.core_client import AsyncMPClient  # noqa: F401
        rep["AsyncMPClient"] = "importable"
    except Exception as e:
        rep["api"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[probe] API check FAILED: {e}", flush=True)
        print("\n===== ASYNC PROBE REPORT =====", flush=True)
        for k, v in rep.items():
            print(f"  {k}: {v}", flush=True)
        print("===== END REPORT =====", flush=True)
        return rep

    async def main():
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm import SamplingParams

        args = AsyncEngineArgs(
            model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.60,
            max_model_len=2048, enforce_eager=True,
            worker_extension_cls="race_worker_ext.RaceWorkerExtension",
            disable_log_stats=True,
        )
        engine = AsyncLLM.from_engine_args(args)
        print("[probe] AsyncLLM engine up", flush=True)

        client = engine.engine_core

        # ---- 2: collective_rpc_async reaches the worker extension ----------
        try:
            sig0 = await client.collective_rpc_async("weight_signature")
            rep["collective_rpc_async"] = f"OK -> {sig0}"
            print(f"[probe] collective_rpc_async OK: {sig0}", flush=True)
        except Exception as e:
            rep["collective_rpc_async"] = f"FAIL {type(e).__name__}: {e}"
            print(f"[probe] collective_rpc_async FAILED: {e}", flush=True)
            sig0 = None

        # ---- 3: THE test — long generation survives a paused update --------
        collected = {"ntok": 0, "done": False}

        async def long_gen():
            sp = SamplingParams(temperature=0.8, max_tokens=400)
            async for out in engine.generate(
                    "Count slowly from 1 to 300, one number per line.",
                    sp, request_id="probe-long"):
                collected["ntok"] = len(out.outputs[0].token_ids)
                if out.finished:
                    collected["done"] = True

        task = asyncio.create_task(long_gen())
        await asyncio.sleep(3.0)               # get well into decoding
        mid_tokens = collected["ntok"]
        print(f"[probe] {mid_tokens} tokens decoded before update", flush=True)

        try:
            kv_before = await client.collective_rpc_async("kv_cache_fingerprint")
            t0 = asyncio.get_event_loop().time()
            await engine.pause_generation(mode="keep", clear_cache=False)
            paused_at = asyncio.get_event_loop().time()

            from torch.multiprocessing.reductions import reduce_tensor
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
            handles = []
            for i, (n, p) in enumerate(hf.named_parameters()):
                t = p.detach().to(torch.bfloat16).cuda().contiguous()
                if i < 3:
                    t = (t * 1.02).contiguous()
                handles.append((n, reduce_tensor(t)[1]))
            await client.collective_rpc_async("update_weights_ipc", args=(handles,))
            await engine.resume_generation()
            resumed_at = asyncio.get_event_loop().time()
            kv_after = await client.collective_rpc_async("kv_cache_fingerprint")
            sig1 = await client.collective_rpc_async("weight_signature")

            rep["pause_update_resume"] = (
                f"OK pause={paused_at - t0:.3f}s total={resumed_at - t0:.3f}s "
                f"weights_changed={sig1 != sig0} "
                f"kv_blocks {kv_before} -> {kv_after} "
                f"({'RETAINED' if kv_before == kv_after else 'CHANGED'})")
            print(f"[probe] {rep['pause_update_resume']}", flush=True)
        except Exception as e:
            rep["pause_update_resume"] = f"FAIL {type(e).__name__}: {e}"
            print(f"[probe] pause/update/resume FAILED: {e}", flush=True)

        try:
            await asyncio.wait_for(task, timeout=240)
        except Exception as e:
            rep["long_gen"] = f"FAIL {type(e).__name__}: {e}"
        else:
            rep["long_gen"] = (f"OK finished={collected['done']} "
                               f"tokens {mid_tokens} -> {collected['ntok']} "
                               f"(continued past the update: "
                               f"{collected['ntok'] > mid_tokens})")
        print(f"[probe] long_gen: {rep['long_gen']}", flush=True)

    asyncio.run(main())

    print("\n===== ASYNC PROBE REPORT =====", flush=True)
    for k, v in rep.items():
        print(f"  {k}: {v}", flush=True)
    print("===== END REPORT =====", flush=True)
    return rep


@app.local_entrypoint()
def main():
    for k, v in probe.remote().items():
        print(f"{k}: {v}")
