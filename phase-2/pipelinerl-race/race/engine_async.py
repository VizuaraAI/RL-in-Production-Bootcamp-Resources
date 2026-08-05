"""AsyncLLM generator — the one that can do a REAL in-flight weight update.

Modelled on ServiceNow/pipelinerl (`pipelinerl/vllm1.py`, `WeightUpdateManager`), which is
the reference implementation of the paper this project reproduces.

Why this file exists separately from engine.py
----------------------------------------------
The synchronous `vllm.LLM` API cannot do in-flight updates, for two compounding reasons:

  1. It is NOT thread-safe. vLLM V1 runs its engine core in a separate process behind a ZMQ
     socket; two threads writing it corrupt the framing and the container aborts with
     `msgspec.ValidationError: Expected array, got int` -> `std::terminate`. Observed.
  2. Serialising it with a lock "fixes" the crash at the wrong granularity: a weight push
     then waits for an entire generate() BATCH to return, so updates land between batches
     and every sequence is produced by exactly one weight version.

`AsyncLLM` solves both. Everything runs on one event loop (no socket race), and it exposes

    await engine.pause_generation(mode="keep", clear_cache=False)
    await engine.engine_core.collective_rpc_async("update_weights_ipc", args=(handles,))
    await engine.resume_generation()

`mode="keep", clear_cache=False` is the crux: the engine stops between STEPS, in-flight
requests and their KV cache survive, and on resume those sequences carry on decoding under
the new weights. That produces the mixed-policy sequence the lecture describes — part of it
written by the old policy, the rest by the new one — with nothing recomputed.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Iterable

import torch

from .telemetry import Telemetry


class AsyncGenerator:
    """vLLM AsyncLLM wrapper owning `gpu_ids`, with true in-flight weight updates."""

    def __init__(self, cfg, tel: Telemetry, gpu_ids: list[int]):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.cfg, self.tel, self.gpu_ids = cfg, tel, gpu_ids
        self.role = "infer"
        for g in gpu_ids:
            tel.gpu(g, self.role, "idle")

        # Same CUDA_VISIBLE_DEVICES discipline as the sync engine: restrict only while the
        # engine is constructed (its workers inherit it at spawn), then restore, or the
        # trainer's cuda:0 silently re-maps onto an inference GPU. See engine.Generator.
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        try:
            args = AsyncEngineArgs(
                model=cfg.model,
                tensor_parallel_size=len(gpu_ids),
                gpu_memory_utilization=cfg.gpu_mem_util_dedicated,
                max_model_len=2048 + cfg.max_new_tokens,
                enforce_eager=False,
                seed=cfg.seed,
                worker_extension_cls="race_worker_ext.RaceWorkerExtension",
                disable_log_stats=True,
            )
            self.engine = AsyncLLM.from_engine_args(args)
        finally:
            if prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev

        self.client = self.engine.engine_core
        self._update_lock = asyncio.Lock()
        self._inflight_seqs = 0
        self.transport = self._choose_transport()

    @property
    def supports_inflight(self) -> bool:
        """True. Unlike the sync engine, this one pauses between engine steps and keeps the
        KV cache, so weights can land mid-sequence."""
        return True

    def _choose_transport(self) -> str:
        """Same up-front decision as the sync engine: an exception inside collective_rpc
        poisons the executor, so the transport can never be chosen by try-and-catch."""
        trainer_gpus = [g for g in range(self.cfg.n_gpus) if g not in self.gpu_ids]
        if not trainer_gpus:
            t = "ipc"
        else:
            ok = all(torch.cuda.can_device_access_peer(src, dst)
                     for dst in self.gpu_ids for src in trainer_gpus)
            t = "ipc" if ok else "shm"
        self.tel.emit("transport", chosen=t, infer=self.gpu_ids, train=trainer_gpus,
                      engine="async")
        print(f"[engine-async] weight transport = {t} (infer={self.gpu_ids} "
              f"train={trainer_gpus})", flush=True)
        return t

    # ---- generation ---------------------------------------------------------
    async def generate_one(self, prompt: str, request_id: str, weight_version: int):
        """Stream ONE sequence to completion. Returns (token_ids, text, logprobs, lag).

        Lag is computed at COMPLETION as (current version - version at start), so a sequence
        that spanned an update is correctly marked stale-by-one even though its later tokens
        came from the newer weights. That is a deliberate lower bound and is documented as
        such — the true staleness varies token by token within the sequence.
        """
        from vllm import SamplingParams

        # n=1 per REQUEST, with the caller issuing group_size separate requests per prompt.
        #
        # Not n=group_size: AsyncLLM.generate() STREAMS, and with n>1 the completions arrive
        # spread across successive yields. Keeping only the final yield therefore captured a
        # partial group — the smoke test found groups of 4 containing 2 distinct prompts,
        # because each request had contributed only 2 completions. Rather than reassemble
        # streamed partials, we make grouping explicit at the call site where it can be
        # asserted.
        sp = SamplingParams(n=1, temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                            max_tokens=self.cfg.max_new_tokens, logprobs=0)
        self._inflight_seqs += 1
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "generating", seqs=self._inflight_seqs)
        final = None
        try:
            async for out in self.engine.generate(prompt, sp, request_id=request_id):
                final = out
        finally:
            self._inflight_seqs -= 1
        return final

    # ---- THE in-flight weight update ---------------------------------------
    async def push_weights_inflight(self, named_tensors: Iterable[tuple[str, torch.Tensor]],
                                    step: int, src_gpus: list[int]) -> float:
        """Pause between engine steps, load new weights, resume. KV cache retained.

        The `finally` around resume_generation is not decoration: if the update raises with
        the engine paused, generation would never restart and the run would hang forever
        looking merely slow.
        """
        from torch.multiprocessing.reductions import reduce_tensor

        from .engine import EngineWeightError

        nt = list(named_tensors)
        nbytes = sum(t.numel() * t.element_size() for _, t in nt)

        async with self._update_lock:
            for g in self.gpu_ids:
                self.tel.gpu(g, self.role, "weight_sync", step=step)
            t0 = time.monotonic()
            await self.engine.pause_generation(mode="keep", clear_cache=False)
            try:
                if self.transport == "ipc":
                    handles = [(n, reduce_tensor(t.contiguous())[1]) for n, t in nt]
                    await self.client.collective_rpc_async(
                        "update_weights_ipc", args=(handles,))
                else:
                    handles = []
                    for n, t in nt:
                        c = t.detach().to("cpu", copy=True).contiguous()
                        c.share_memory_()
                        handles.append((n, reduce_tensor(c)))
                    await self.client.collective_rpc_async(
                        "update_weights_shm", args=(handles,))
            except Exception as e:
                self.tel.emit("error", where="push_weights_inflight", step=step,
                              err=f"{type(e).__name__}: {e}")
                raise EngineWeightError(f"in-flight weight push failed at {step}: {e}") from e
            finally:
                await self.engine.resume_generation()
            ms = (time.monotonic() - t0) * 1e3

        self.tel.weights(step=step, src=src_gpus, dst=self.gpu_ids, nbytes=nbytes,
                         ms=ms, inflight=True, transport=self.transport)
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "generating", seqs=self._inflight_seqs)
        return ms

    async def shutdown(self) -> None:
        try:
            self.engine.shutdown()
        except Exception:
            pass
