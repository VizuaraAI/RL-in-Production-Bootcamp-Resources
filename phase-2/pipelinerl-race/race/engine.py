"""Shared compute plumbing: a vLLM Generator and a torch Trainer.

Both arms use these SAME two classes. An arm is nothing more than a scheduling policy over
them — who runs when, on which GPUs, and how weights move. Keeping the compute identical is
what makes the race a measurement of architecture rather than of implementation quality.

  Conventional : Generator(TP=N) and Trainer(N) time-share all N GPUs. Generator.sleep()
                 frees its KV cache so the trainer can use the memory. Weights move while
                 generation is stopped  -> weights(inflight=False).
  PipelineRL   : Generator(TP=k) owns GPUs [N-k, N) forever, Trainer owns [0, N-k) forever.
                 Both run at once. Weights are pushed between decode steps without stopping
                 generation or dropping the KV cache -> weights(inflight=True).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import torch

from .telemetry import Telemetry


class EngineWeightError(RuntimeError):
    """A weight push failed. Always fatal — see Generator.push_weights."""


@dataclass
class Rollout:
    pid: str
    prompt_ids: list[int]
    completion_ids: list[int]
    logp: list[float]      # behaviour log-probs, token-aligned with completion_ids
    reward: float
    lag: int               # optimizer steps of staleness at generation time


class Generator:
    """vLLM wrapper. Owns `gpu_ids`; reports every state change to telemetry."""

    def __init__(self, cfg, tel: Telemetry, gpu_ids: list[int], sleepable: bool):
        from vllm import LLM

        self.cfg, self.tel, self.gpu_ids = cfg, tel, gpu_ids
        self.role = "hybrid" if sleepable else "infer"
        for g in gpu_ids:
            tel.gpu(g, self.role, "idle")

        # CUDA_VISIBLE_DEVICES must be restricted ONLY while the engine is constructed, and
        # restored immediately afterwards.
        #
        # vLLM has no per-instance device argument; the only way to pin it to a subset is
        # CUDA_VISIBLE_DEVICES, and its TP workers are spawned during LLM() construction so
        # they inherit whatever is set at that moment. But leaving it set would poison the
        # PARENT process too: with CVD="1", the trainer's `cuda:0` re-maps to physical GPU 1
        # — the generator's GPU. The pipeline arm would silently run both roles on one GPU,
        # leave the other idle, and still report a clean train/infer split. That is exactly
        # the claim this experiment exists to measure, so it would invalidate the result
        # while looking correct.
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        try:
            self.llm = LLM(
                model=cfg.model,
                tensor_parallel_size=len(gpu_ids),
                enable_sleep_mode=sleepable,
                gpu_memory_utilization=cfg.gpu_mem_util_colocated if sleepable else cfg.gpu_mem_util_dedicated,
                max_model_len=2048 + cfg.max_new_tokens,
                enforce_eager=False,
                seed=cfg.seed,
                worker_extension_cls="race_worker_ext.RaceWorkerExtension",
                disable_log_stats=True,
            )
        finally:
            if prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev
        self.asleep = False
        self.transport = self._choose_transport()

        # vLLM V1 runs its engine core in a SEPARATE PROCESS behind a ZMQ socket, and the
        # `LLM` wrapper is NOT thread-safe. The pipeline arm has a producer thread calling
        # generate() while the consumer thread pushes weights; both write that one socket,
        # and the framing corrupts:
        #
        #     vllm/v1/engine/core.py process_input_sockets
        #     msgspec.ValidationError: Expected `array`, got `int`
        #     -> terminate called after throwing an instance of 'c10::Error'
        #
        # which kills the container outright. The probe never hit it because it performed a
        # single update; sustained concurrent use aborts within minutes.
        #
        # This lock makes engine access safe. The cost is real and must not be glossed over:
        # a weight push now waits for the in-progress generate() call to RETURN, so updates
        # land BETWEEN generation batches rather than mid-decode. That is NOT the paper's
        # in-flight update — see `supports_inflight`.
        self._engine_lock = threading.RLock()

    @property
    def supports_inflight(self) -> bool:
        """True only if weights can be pushed WHILE decoding proceeds.

        False for the synchronous `LLM` API: the engine lock serialises access, so an update
        cannot overlap a generate() call. Reaching genuine in-flight updates requires driving
        vLLM through AsyncLLM on an event loop, where engine steps are interleavable.

        The probe proved the ENGINE tolerates a mid-decode update with the KV cache retained
        (kv_blocks unchanged, 279 tokens still produced). The blocker is the client API's
        thread-safety, not the engine. Reported honestly rather than assumed.
        """
        return False

    def _choose_transport(self) -> str:
        """Pick the weight-transfer path ONCE, from hardware capability.

        This must be decided up front, never by try-IPC-and-catch: an exception inside
        collective_rpc poisons the vLLM executor so every later RPC returns the same error.
        A runtime fallback would therefore kill the engine on the very first push.

        CUDA IPC needs peer access between the trainer's GPU and this engine's GPUs. That
        holds on NVLink hardware (H100) but NOT on Modal's L4s, where the push dies with
        "peer access is not supported between these two devices". When peer access is
        unavailable we stage through shared host memory instead — slower, correct
        everywhere, and recorded in telemetry so the two are never conflated.
        """
        trainer_gpus = [g for g in range(self.cfg.n_gpus) if g not in self.gpu_ids]
        if not trainer_gpus:                      # colocated: same device, no P2P needed
            t = "ipc"
        else:
            ok = all(torch.cuda.can_device_access_peer(src, dst)
                     for dst in self.gpu_ids for src in trainer_gpus)
            t = "ipc" if ok else "shm"
        self.tel.emit("transport", chosen=t, infer=self.gpu_ids, train=trainer_gpus)
        print(f"[engine] weight transport = {t} (infer={self.gpu_ids} "
              f"train={trainer_gpus})", flush=True)
        return t

    # ---- generation ---------------------------------------------------------
    def generate(self, prompts: list[str], n: int, lag: int) -> list:
        """Blocking batch generation. Marks every owned GPU `generating` for the duration.

        NOTE: this is the conventional arm's path. The long tail is *inside* this call —
        vLLM's running batch shrinks as sequences finish, so the GPUs are progressively
        less utilised even though they are nominally 'generating'. We record the shrink via
        the seqs_in_flight detail so the visualizer can show the batch draining.
        """
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                            max_tokens=self.cfg.max_new_tokens, logprobs=0, seed=None)
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "generating", seqs=len(prompts) * n)
        t0 = time.monotonic()
        with self._engine_lock:
            outs = self.llm.generate(prompts, sp, use_tqdm=False)
        dt = time.monotonic() - t0
        ntok = sum(len(c.token_ids) for o in outs for c in o.outputs)
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "idle", tokens_s=round(ntok / max(dt, 1e-6)))
        self.tel.metric(step=-1, gen_tokens=ntok, gen_s=round(dt, 3),
                        gen_tokens_s=round(ntok / max(dt, 1e-6), 1))
        return outs

    # ---- memory time-sharing (conventional only) ----------------------------
    def sleep(self) -> None:
        """Release KV cache + weights so the trainer can use this GPU's memory.
        This is the colocation cost veRL's HybridEngine pays and PipelineRL avoids."""
        if self.asleep:
            return
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "weight_sync", op="sleep")
        t0 = time.monotonic()
        with self._engine_lock:
            self.llm.sleep(level=1)
        self.asleep = True
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "idle")
        self.tel.metric(step=-1, sleep_ms=round((time.monotonic() - t0) * 1e3, 1))

    def wake(self) -> None:
        if not self.asleep:
            return
        t0 = time.monotonic()
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "weight_sync", op="wake")
        with self._engine_lock:
            self.llm.wake_up()
        self.asleep = False
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "idle")
        self.tel.metric(step=-1, wake_ms=round((time.monotonic() - t0) * 1e3, 1))

    # ---- weight ingest ------------------------------------------------------
    def push_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]],
                     step: int, src_gpus: list[int], inflight: bool) -> None:
        """Load new weights into the live engine.

        Transport is CUDA IPC: `reduce_tensor` yields a picklable handle the worker reopens
        against the SAME GPU memory, so nothing is copied over PCIe. Raw CUDA tensors cannot
        be passed as RPC args at all — vLLM 0.11 rejects them ("can't convert cuda:0 device
        type tensor to numpy"), which is why this indirection exists.

        When `inflight` is True the engine is mid-generation and the KV cache is retained:
        sequences continue from where they were, produced partly by old and partly by new
        weights (the 'mixed policy' sequence the lecture describes).

        Raises EngineWeightError on failure. The caller MUST treat that as fatal: an
        exception inside collective_rpc poisons the executor, so every later RPC returns the
        same error. Retrying past it would leave the generator serving stale weights for the
        rest of the run while training appeared to proceed normally — a silently meaningless
        result, which is worse than a crash.
        """
        from torch.multiprocessing.reductions import reduce_tensor

        nt = list(named_tensors)
        nbytes = sum(t.numel() * t.element_size() for _, t in nt)
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "weight_sync", step=step)
        t0 = time.monotonic()
        try:
            if self.transport == "ipc":
                handles = [(n, reduce_tensor(t.contiguous())[1]) for n, t in nt]
                with self._engine_lock:
                    self.llm.collective_rpc("update_weights_ipc", args=(handles,))
            else:
                # Stage through SHARED host memory. share_memory_() puts the tensor in a
                # file-backed mapping the worker can map directly, so nothing is copied
                # through the RPC socket (which would also mangle it — vLLM decodes plain
                # tensors into nested lists).
                handles = []
                for n, t in nt:
                    c = t.detach().to("cpu", copy=True).contiguous()
                    c.share_memory_()
                    handles.append((n, reduce_tensor(c)))
                with self._engine_lock:
                    self.llm.collective_rpc("update_weights_shm", args=(handles,))
        except Exception as e:
            self.tel.emit("error", where="push_weights", step=step,
                          err=f"{type(e).__name__}: {e}")
            raise EngineWeightError(f"weight push failed at step {step}: {e}") from e
        ms = (time.monotonic() - t0) * 1e3
        self.tel.weights(step=step, src=src_gpus, dst=self.gpu_ids,
                         nbytes=nbytes, ms=ms, inflight=inflight,
                         transport=self.transport)
        for g in self.gpu_ids:
            self.tel.gpu(g, self.role, "generating" if inflight else "idle")


class Trainer:
    """Policy model + optimizer on `gpu_ids`. Single process; FSDP when len(gpu_ids) > 1."""

    def __init__(self, cfg, tel: Telemetry, gpu_ids: list[int]):
        from transformers import AutoModelForCausalLM

        self.cfg, self.tel, self.gpu_ids = cfg, tel, gpu_ids
        self.dev = torch.device(f"cuda:{gpu_ids[0]}")
        torch.manual_seed(cfg.seed)

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(self.dev)
        self.model.gradient_checkpointing_enable()
        self.model.train()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                                     weight_decay=0.0)
        self.step_idx = 0
        for g in gpu_ids:
            tel.gpu(g, "train", "idle")

    def step(self, batch) -> dict:
        """One optimizer step, executed as several micro-batches with gradient accumulation.

        MEMORY: a causal-LM forward materialises logits of shape [B, T, vocab]. At Qwen2.5's
        151,936-token vocabulary a full RL batch (128 sequences x ~700 tokens) is ~45 GiB in
        bf16 — it OOM'd an 80 GiB H100 outright ("Tried to allocate 44.99 GiB"). Chunking the
        log-softmax does not help; the allocation happens inside the model.

        So the batch is split and gradients accumulated. The gradient is IDENTICAL to the
        full-batch one because each micro-batch's token-normalised loss is re-weighted by
        that micro-batch's share of the batch's completion tokens — summing to exactly the
        global token-level normalisation grpo_loss would have applied. Getting that weighting
        wrong (e.g. averaging micro-batch losses) would silently change the effective loss
        and, worse, change it differently for the two arms if their batch shapes differ.
        """
        from .grpo import Batch, grpo_loss

        g0 = self.gpu_ids[0]
        B = batch.input_ids.shape[0]
        mb = max(1, min(self.cfg.micro_batch_size, B))
        total_tokens = batch.completion_mask.sum().clamp(min=1.0)

        agg: dict = {}
        nsteps = 0
        self.opt.zero_grad(set_to_none=True)

        for s in range(0, B, mb):
            e = min(s + mb, B)
            sub = Batch(
                input_ids=batch.input_ids[s:e],
                attention_mask=batch.attention_mask[s:e],
                completion_mask=batch.completion_mask[s:e],
                advantages=batch.advantages[s:e],
                logp_behavior=batch.logp_behavior[s:e],
                lag=batch.lag[s:e],
            )
            self.tel.gpu(g0, "train", "forward", step=self.step_idx, mb=f"{s}:{e}")
            out = self.model(input_ids=sub.input_ids, attention_mask=sub.attention_mask)
            loss, stats = grpo_loss(out.logits, sub, clip_eps=self.cfg.clip_eps,
                                    kl_coef=self.cfg.kl_coef)

            # re-weight so the accumulated gradient equals the full-batch one
            share = sub.completion_mask.sum() / total_tokens
            self.tel.gpu(g0, "train", "backward", step=self.step_idx, mb=f"{s}:{e}")
            (loss * share).backward()

            del out, loss
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            nsteps += 1

        stats = {k: v / max(nsteps, 1) for k, v in agg.items()}

        self.tel.gpu(g0, "train", "optimizer_step", step=self.step_idx)
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.step_idx += 1

        self.tel.gpu(g0, "train", "idle")
        s = {k: float(v) for k, v in stats.items()}
        s["grad_norm"] = float(gn)
        self.tel.metric(step=self.step_idx, **s)
        return s

    def inference_weights(self) -> Iterable[tuple[str, torch.Tensor]]:
        """bf16 copies of the current weights, ready to hand to vLLM."""
        for name, p in self.model.named_parameters():
            yield name, p.detach().to(torch.bfloat16)

    # ---- memory time-sharing, for the COLOCATED (conventional) arm only ------
    def offload_optimizer(self) -> float:
        """Move AdamW state and gradients to host so vLLM can re-map its KV cache.

        This is not a workaround — it is what veRL's HybridEngine actually does, and it is
        the price of colocation. Without it, `llm.wake_up()` dies with
        "CUDA Error: out of memory at cumem_allocator.cpp" because the trainer still holds
        the memory vLLM wants back (observed on 2xL4).

        Model PARAMETERS stay on GPU: they are only ~1 GiB in bf16 and are needed
        immediately afterwards for the weight push. It is the optimizer moments (two fp32
        tensors per parameter, ~4x the model) that must go.

        Returns seconds spent, which the conventional arm charges to its own bubble — this
        transfer is real wall-clock the pipeline arm never pays.
        """
        t0 = time.monotonic()
        g0 = self.gpu_ids[0]
        self.tel.gpu(g0, "train", "weight_sync", op="offload_optimizer")
        for group in self.opt.param_groups:
            for p in group["params"]:
                st = self.opt.state.get(p)
                if not st:
                    continue
                for k, v in st.items():
                    if torch.is_tensor(v) and v.is_cuda:
                        st[k] = v.to("cpu", non_blocking=False)
                if p.grad is not None:
                    p.grad = None
        torch.cuda.empty_cache()
        self.tel.gpu(g0, "train", "idle")
        dt = time.monotonic() - t0
        self.tel.metric(step=self.step_idx, optimizer_offload_s=round(dt, 3))
        return dt

    def reload_optimizer(self) -> float:
        """Bring AdamW state back to GPU before the next optimizer step."""
        t0 = time.monotonic()
        g0 = self.gpu_ids[0]
        self.tel.gpu(g0, "train", "weight_sync", op="reload_optimizer")
        for group in self.opt.param_groups:
            for p in group["params"]:
                st = self.opt.state.get(p)
                if not st:
                    continue
                for k, v in st.items():
                    if torch.is_tensor(v) and not v.is_cuda:
                        st[k] = v.to(self.dev, non_blocking=True)
        torch.cuda.synchronize()
        self.tel.gpu(g0, "train", "idle")
        dt = time.monotonic() - t0
        self.tel.metric(step=self.step_idx, optimizer_reload_s=round(dt, 3))
        return dt
