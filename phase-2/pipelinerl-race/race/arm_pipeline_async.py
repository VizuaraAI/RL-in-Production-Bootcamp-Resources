"""Arm C — PipelineRL with GENUINE in-flight weight updates.

Same GPU partition as arm B (`arm_pipeline.py`): inference GPUs generate forever, trainer
GPUs train forever. The one difference is *when* weights land:

    arm B (sync LLM + lock) : updates wait for a whole generate() batch to return.
                              Every sequence is produced by exactly one weight version.
    arm C (AsyncLLM)        : updates pause the engine BETWEEN STEPS with the KV cache
                              retained, so a sequence in progress continues under the new
                              weights. Mixed-policy sequences, nothing recomputed.

Running both is deliberate. A -> B measures what concurrency alone buys; B -> C isolates
what in-flight updates specifically buy, which the paper does not separate.

Everything lives on ONE asyncio event loop — that is also what makes it safe. The
synchronous API crashed precisely because two threads shared the engine's ZMQ socket. Here
the only blocking work is the optimizer step, which is pushed to a worker thread via
asyncio.to_thread so it cannot stall generation.
"""
from __future__ import annotations

import asyncio
import time

from .batching import rollouts_to_batch
from .data import build_chat, load_task, reward_fn
from .engine import EngineWeightError, Rollout, Trainer
from .engine_async import AsyncGenerator
from .telemetry import Telemetry


def run(cfg, tel: Telemetry) -> dict:
    return asyncio.run(_run(cfg, tel))


async def _run(cfg, tel: Telemetry) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    problems = load_task(cfg.task, seed=cfg.seed)
    tel.phase("setup", n_problems=len(problems))

    infer_gpus, train_gpus = cfg.infer_gpu_ids, cfg.train_gpu_ids
    tel.phase("partition", infer=infer_gpus, train=train_gpus)

    gen = AsyncGenerator(cfg, tel, gpu_ids=infer_gpus)
    trainer = Trainer(cfg, tel, gpu_ids=train_gpus)

    # Same physical placement witness as arm B — a CUDA_VISIBLE_DEVICES leak would put both
    # roles on one GPU and still report a clean split.
    import torch as _torch
    free_gib = {g: round(_torch.cuda.mem_get_info(g)[0] / 2**30, 2)
                for g in range(cfg.n_gpus)}
    trainer_alloc = _torch.cuda.memory_allocated(trainer.dev) / 2**30
    tel.emit("placement", trainer_dev=str(trainer.dev), infer=infer_gpus, train=train_gpus,
             free_gib=free_gib, trainer_alloc_gib=round(trainer_alloc, 2))
    print(f"[pipeline-async] placement — trainer={trainer.dev} "
          f"(alloc {trainer_alloc:.2f} GiB) infer={infer_gpus} free={free_gib}", flush=True)
    if trainer.dev.index in infer_gpus:
        raise RuntimeError(f"trainer on cuda:{trainer.dev.index}, an INFERENCE gpu. "
                           f"Refusing to run — the split did not happen.")

    tel.phase("ready")

    q: asyncio.Queue = asyncio.Queue(maxsize=cfg.prompts_per_step * 4)
    version = {"v": 0}
    stop = asyncio.Event()
    produced = {"n": 0}
    t0 = time.monotonic()

    # ---------------- producers: keep the inference GPUs saturated ----------
    async def produce():
        cursor = 0
        rid = 0
        while not stop.is_set():
            probs = [problems[(cursor + i) % len(problems)]
                     for i in range(cfg.prompts_per_step)]
            cursor += cfg.prompts_per_step
            v_start = version["v"]

            # ONE request per rollout, group_size of them per prompt, issued together so
            # they decode concurrently. Grouping is constructed here explicitly rather than
            # inferred from a streamed n>1 response — see AsyncGenerator.generate_one.
            async def one(p, k):
                nonlocal rid
                rid += 1
                out = await gen.generate_one(build_chat(tok, p), f"r{rid}", v_start)
                return p, out

            try:
                jobs = [(p, k) for p in probs for k in range(cfg.group_size)]
                results = await asyncio.gather(*[one(p, k) for p, k in jobs])
            except Exception as e:
                tel.emit("error", where="producer", err=f"{type(e).__name__}: {e}")
                stop.set()
                return

            # lag measured at completion — a sequence that spanned an update is stale by
            # the number of versions that landed while it decoded.
            lag = max(0, version["v"] - v_start)
            rolls = []
            for p, out in results:
                if out is None or not out.outputs:
                    continue
                c = out.outputs[0]
                lp = []
                if c.logprobs:
                    for tokid, d in zip(c.token_ids, c.logprobs):
                        e = d.get(tokid) if isinstance(d, dict) else None
                        lp.append(float(getattr(e, "logprob", 0.0)) if e else 0.0)
                rolls.append(Rollout(pid=p.pid, prompt_ids=list(out.prompt_token_ids),
                                     completion_ids=list(c.token_ids), logp=lp,
                                     reward=reward_fn(c.text, p.answer), lag=lag))
            # `jobs` is prompt-major, and asyncio.gather preserves input order regardless of
            # completion order, so rolls arrive already grouped. rollouts_to_batch asserts
            # this — do not rely on it silently.
            produced["n"] += len(rolls)
            for r in rolls:
                tel.sample(reward=r.reward, tokens=len(r.completion_ids), lag=r.lag,
                           prompt_id=r.pid)
            await q.put(rolls)

    # ---------------- consumer: train, then push weights IN FLIGHT ----------
    async def consume():
        step = 0
        consumed = 0
        history = []
        while not stop.is_set():
            elapsed = time.monotonic() - t0
            if cfg.budget == "wallclock" and elapsed >= cfg.wallclock_s:
                break
            if cfg.budget == "samples" and consumed >= cfg.max_samples:
                break
            if step >= cfg.max_steps:
                break

            for g in train_gpus:
                tel.gpu(g, "train", "waiting")
            try:
                rolls = await asyncio.wait_for(q.get(), timeout=120)
            except asyncio.TimeoutError:
                tel.emit("starved", step=step)
                continue

            keep = len(rolls) - (len(rolls) % cfg.group_size)
            rolls = rolls[:keep]
            if not rolls:
                continue

            batch, info = rollouts_to_batch(rolls, tok, cfg.group_size, trainer.dev)
            # The optimizer step is blocking torch work — run it OFF the event loop so
            # generation keeps streaming while the trainer computes. This is the whole
            # point of the architecture; doing it inline would serialise the two roles
            # and quietly turn arm C back into a conventional loop.
            stats = await asyncio.to_thread(trainer.step, batch)
            consumed += len(rolls)
            step += 1

            try:
                ms = await gen.push_weights_inflight(
                    trainer.inference_weights(), step=step, src_gpus=train_gpus)
            except EngineWeightError as e:
                tel.emit("fatal", where="push_weights_inflight", step=step, err=str(e))
                print(f"[pipeline-async] FATAL at step {step}: {e}", flush=True)
                stop.set()
                break
            version["v"] = step

            history.append({"step": step, "t": round(time.monotonic() - t0, 2),
                            "samples": consumed, "weight_ms": round(ms, 1),
                            **info, **stats})
            if step % 10 == 0:
                tel.checkpoint()
        stop.set()
        return step, consumed, history

    prod = asyncio.create_task(produce())
    step, consumed, history = await consume()
    prod.cancel()
    try:
        await prod
    except asyncio.CancelledError:
        pass
    await gen.shutdown()

    return {"arm": "pipeline_async", "steps": step, "samples": consumed,
            "produced": produced["n"], "inflight": True,
            "wallclock_s": round(time.monotonic() - t0, 2), "history": history}
