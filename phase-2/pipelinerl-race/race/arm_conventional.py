"""Arm A — Conventional / veRL-style RL.

All N GPUs do the same thing at the same time, alternating between two phases:

    [ generate on all N ] -> [ swap memory ] -> [ train on all N ] -> [ push weights ] -> repeat

This is the colocated HybridEngine pattern veRL uses, and it is the baseline the PipelineRL
paper compares against ("we believe veRL's throughput would be similar to our Conventional
RL baseline").

Where the bubble comes from — two distinct places, and the telemetry separates them:
  1. THE LONG TAIL, inside generation. vLLM's running batch drains as sequences finish;
     the last straggler holds the whole cluster. This is the big one and it grows with
     sequence-length variance.
  2. THE PHASE SWAP. sleep/wake plus the weight push, during which nothing useful happens.

Data here is always perfectly on-policy: lag == 0 for every sample, by construction.
"""
from __future__ import annotations

import time

from .batching import rollouts_to_batch, vllm_output_to_rollouts
from .data import build_chat, load_task, reward_fn
from .engine import EngineWeightError, Generator, Trainer
from .telemetry import Telemetry


def run(cfg, tel: Telemetry) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    problems = load_task(cfg.task, seed=cfg.seed)
    tel.phase("setup", n_problems=len(problems))

    # Colocated: the generator owns every GPU and must be able to give the memory back.
    gen = Generator(cfg, tel, gpu_ids=list(range(cfg.n_gpus)), sleepable=True)
    trainer = Trainer(cfg, tel, gpu_ids=list(range(cfg.n_gpus)))

    # Engines are up. Everything before this point is one-off setup (model load,
    # torch.compile, CUDA-graph capture) during which the GPUs are legitimately not
    # doing RL work. Counting it as "idle" made gpu_busy_frac read 8% on a smoke run
    # and would have buried the real bubble under startup cost.
    tel.phase("ready")

    t0 = time.monotonic()
    cursor = 0
    step = 0
    total_samples = 0
    history = []

    while True:
        elapsed = time.monotonic() - t0
        if cfg.budget == "wallclock" and elapsed >= cfg.wallclock_s:
            break
        if cfg.budget == "samples" and total_samples >= cfg.max_samples:
            break
        if step >= cfg.max_steps:
            break

        # ---------------- PHASE 1: generate (all N GPUs) ----------------
        tel.phase("generate", step=step)
        batch_probs = [problems[(cursor + i) % len(problems)]
                       for i in range(cfg.prompts_per_step)]
        cursor += cfg.prompts_per_step
        prompts = [build_chat(tok, p) for p in batch_probs]

        gen.wake()  # no-op on the first iteration; memory was freed in phase 4
        outs = gen.generate(prompts, n=cfg.group_size, lag=0)

        rolls = vllm_output_to_rollouts(outs, batch_probs, reward_fn, lag=0, tokenizer=tok)
        for r in rolls:
            tel.sample(reward=r.reward, tokens=len(r.completion_ids), lag=0, prompt_id=r.pid)
        total_samples += len(rolls)

        # ---------------- PHASE 2: hand the GPUs to the trainer ----------
        # Conventional RL's defining cost: generation must fully STOP and release its KV
        # cache before training can use the memory.
        gen.sleep()

        # ---------------- PHASE 3: train (all N GPUs) -------------------
        tel.phase("train", step=step)
        trainer.reload_optimizer()
        batch, info = rollouts_to_batch(rolls, tok, cfg.group_size, trainer.dev)
        stats = trainer.step(batch)

        # ---------------- PHASE 4: publish weights ----------------------
        # Free the trainer's memory BEFORE waking vLLM. This is the peak-pressure moment of
        # the whole loop: gradients and AdamW moments are both resident, and vLLM is about
        # to demand its KV cache back. PyTorch's caching allocator also RESERVES freed
        # blocks rather than returning them to CUDA, so vLLM's separate cumem allocator
        # cannot see them — hence the empty_cache() inside offload_optimizer().
        #
        # Order matters and cost me a smoke run: an earlier version offloaded at the TOP of
        # the loop, one full iteration too late, and wake_up() died with
        # "CUDA Error: out of memory at cumem_allocator.cpp:62" every time.
        #
        # Stop-the-world: generation is not running, so inflight=False. Contrast with the
        # pipeline arm, where this same call happens with generation still in flight.
        trainer.offload_optimizer()
        gen.wake()
        try:
            gen.push_weights(trainer.inference_weights(), step=step,
                             src_gpus=list(range(cfg.n_gpus)), inflight=False)
        except EngineWeightError as e:
            # Same reasoning as the pipeline arm: a poisoned executor means every
            # subsequent generation uses stale weights. Fatal, not retryable.
            tel.emit("fatal", where="push_weights", step=step, err=str(e))
            print(f"[conventional] FATAL at step {step}: {e}", flush=True)
            break

        history.append({"step": step, "t": round(time.monotonic() - t0, 2),
                        "samples": total_samples, **info, **stats})
        step += 1
        if step % 10 == 0:
            tel.checkpoint()   # survive a hard abort with a usable partial stream

    return {"arm": "conventional", "steps": step, "samples": total_samples,
            "wallclock_s": round(time.monotonic() - t0, 2), "history": history}
