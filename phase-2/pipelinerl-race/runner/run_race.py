"""Race entrypoints.

    # validate the whole loop cheaply (tiny budget, 2 GPUs, both arms)
    modal run --detach runner/run_race.py::smoke

    # the real thing
    modal run --detach runner/run_race.py::race --gpus 4 --minutes 45

Self-contained image (Modal does not auto-mount local Python sources — a cross-module
import dies at container start). The `race` package is copied to /root, which is on
sys.path inside the container.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    # vLLM 0.18.1, NOT 0.11.0. AsyncLLM.pause_generation / resume_generation — the
    # primitives that make a genuine in-flight weight update possible — do not exist in
    # 0.11 (verified on GPU: "AsyncLLM object has no attribute pause_generation"). This is
    # the version ServiceNow/pipelinerl itself pins. 0.18.1 also still exposes LLM.sleep and
    # LLM.collective_rpc, so the SAME image serves all three arms — which the fairness
    # contract now requires, since RaceConfig.fairness_key() includes the env fingerprint.
    .pip_install(
        "vllm==0.18.1",
        "transformers>=4.56,<5",
        "datasets==4.0.0",
        "numpy<2.3",
        "matplotlib==3.10.0",
    )
    .env({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_V1": "1",
        # REQUIRED. collective_rpc defaults to a safe msgpack serializer that cannot carry
        # a CUDA-IPC handle ("Object of type torch._C._TensorMeta is not serializable").
        # Verified on GPU; without this the pipeline arm cannot push weights at all.
        # The RPC channel is internal to a single-tenant Modal container we control.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    })
    .add_local_dir(str(ROOT / "race"), "/root/race", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(str(ROOT / "race" / "worker_ext.py"), "/root/race_worker_ext.py",
                    copy=True)
)

app = modal.App("pipelinerl-race")
data_vol = modal.Volume.from_name("race-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("race-hf-cache", create_if_missing=True)
VOL = {"/data": data_vol, "/root/.cache/huggingface": hf_cache}


def _run_arm(cfg_kwargs: dict) -> dict:
    """Executed inside the container. One arm, start to finish."""
    import time

    from race.config import RaceConfig
    from race.telemetry import Telemetry

    cfg = RaceConfig(**cfg_kwargs)
    cfg.preflight()   # fails fast on model/TP mismatch, before any GPU work
    out = Path(cfg.out_dir) / cfg.run_id / cfg.arm
    out.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(path=out / "events.jsonl", arm=cfg.arm, n_gpus=cfg.n_gpus)

    print(f"[race] arm={cfg.arm} gpus={cfg.n_gpus} "
          f"(infer={cfg.infer_gpu_ids} train={cfg.train_gpu_ids}) "
          f"budget={cfg.budget} model={cfg.model}", flush=True)

    t0 = time.monotonic()
    try:
        if cfg.arm == "conventional":
            from race.arm_conventional import run as arm_run
        elif cfg.arm == "pipeline_async":
            from race.arm_pipeline_async import run as arm_run
        else:
            from race.arm_pipeline import run as arm_run
        result = arm_run(cfg, tel)
        result["ok"] = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        tel.emit("error", where="arm", err=f"{type(e).__name__}: {e}")
        result = {"arm": cfg.arm, "ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        tel.close()

    result["wallclock_total_s"] = round(time.monotonic() - t0, 2)
    result["config"] = cfg.to_dict()
    result["fairness_key"] = cfg.fairness_key()
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[race] {cfg.arm} done: ok={result.get('ok')} "
          f"steps={result.get('steps')} samples={result.get('samples')} "
          f"in {result['wallclock_total_s']}s", flush=True)
    return result


@app.function(image=IMAGE, gpu="L4:2", volumes=VOL, timeout=60 * 60)
def smoke_arm(cfg_kwargs: dict) -> dict:
    return _run_arm(cfg_kwargs)


@app.function(image=IMAGE, gpu="H100:2", volumes=VOL, timeout=60 * 60 * 5)
def race_arm(cfg_kwargs: dict) -> dict:
    return _run_arm(cfg_kwargs)


def _base(run_id: str, gpus: int, minutes: int, model: str, task: str) -> dict:
    return dict(model=model, task=task, n_gpus=gpus, run_id=run_id,
                budget="wallclock", wallclock_s=minutes * 60)


@app.local_entrypoint()
def smoke(run_id: str = "smoke", model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Cheap end-to-end validation of ALL THREE arms on 2 L4s.

    Mandatory after any dependency change: this repo jumped vLLM 0.11 -> 0.18 and torch
    2.8 -> 2.10 to obtain pause_generation/resume_generation, and a version jump that large
    can break the sync engine, sleep/wake, or the IPC transport in ways only a real GPU
    shows."""
    base = dict(model=model, task="gsm8k", n_gpus=2, run_id=run_id,
                budget="samples", max_samples=256, max_steps=3,
                prompts_per_step=4, group_size=4, max_new_tokens=256)
    cfgs = [dict(base, arm="conventional"),
            dict(base, arm="pipeline", n_infer_gpus=1),
            dict(base, arm="pipeline_async", n_infer_gpus=1)]
    results = list(smoke_arm.map(cfgs))
    _summarise(results)


@app.local_entrypoint()
def race(run_id: str = "race1", gpus: int = 4, minutes: int = 45,
         model: str = "Qwen/Qwen2.5-1.5B-Instruct", task: str = "gsm8k",
         infer_gpus: int = 2, arms: str = "both"):
    """The headline run: identical wall-clock budget, compare reward.

    Default model is 1.5B, not 0.5B: the conventional arm shards across ALL gpus and
    Qwen2.5-0.5B's 14 attention heads do not divide 4. See RaceConfig.preflight().

    `arms` may be both|conventional|pipeline. Re-running a single arm is legitimate because
    the arms execute in separate containers and never interact — and publish_run.py still
    compares the fairness keys, so a mismatched pair cannot be published by accident.
    """
    base = _base(run_id, gpus, minutes, model, task)
    # Three conditions, deliberately. A->B isolates what CONCURRENCY buys; B->C isolates
    # what IN-FLIGHT weight updates buy on top of it. The paper conflates the two.
    ALL = ["conventional", "pipeline", "pipeline_async"]
    want = ALL if arms in ("both", "all") else [a.strip() for a in arms.split(",")]
    cfgs = []
    for a in ALL:
        if a in want:
            cfgs.append(dict(base, arm=a) if a == "conventional"
                        else dict(base, arm=a, n_infer_gpus=infer_gpus))
    if not cfgs:
        raise ValueError(
            f"--arms must be all|conventional|pipeline|pipeline_async (comma-separated), "
            f"got {arms!r}")
    print(f"[race] launching arms: {[c['arm'] for c in cfgs]}")
    results = list(race_arm.map(cfgs))
    _summarise(results)


def _summarise(results: list[dict]) -> None:
    print("\n===== RACE SUMMARY =====")
    keys = {}
    for r in results:
        print(f"  {r.get('arm'):14s} ok={r.get('ok')} steps={r.get('steps')} "
              f"samples={r.get('samples')} t={r.get('wallclock_total_s')}s"
              + (f"  ERROR: {r.get('error')}" if not r.get("ok") else ""))
        keys[r.get("arm")] = r.get("fairness_key")
    # Compare ALL arms pairwise against the first, not just a hard-coded pair — with three
    # conditions a 2-arm check would silently skip the third.
    named = {k: v for k, v in keys.items() if v}
    if len(named) >= 2:
        ref_name, ref = next(iter(named.items()))
        diff = {}
        for name, k in list(named.items())[1:]:
            for f in set(ref) | set(k):
                if ref.get(f) != k.get(f):
                    diff[f] = (f"{ref_name}={ref.get(f)!r}", f"{name}={k.get(f)!r}")
        if diff:
            print("  !! FAIRNESS VIOLATION — arms differ beyond the partition:")
            for k, v in diff.items():
                print(f"       {k}: {v[0]!r} vs {v[1]!r}")
        else:
            print("  fairness key: IDENTICAL across arms ✓")
    print("========================")
