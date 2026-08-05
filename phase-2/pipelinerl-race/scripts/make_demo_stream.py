#!/usr/bin/env python3
"""Generate a SCHEMA-VALID SYNTHETIC event stream, purely to develop/verify the visualizer
before a real run exists.

!! THIS IS NOT DATA. !!  Nothing produced here may be used in the deck, the paper, the
dashboard, or any figure. It exists so the Three.js front-end can be exercised against the
exact schema race/telemetry.py emits. Real runs overwrite viz/runs/<arm>/events.jsonl.
Every file written here is stamped with "synthetic": true in its run event so it is
impossible to confuse with a real capture.

    python3 scripts/make_demo_stream.py
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).parent.parent / "viz" / "runs"
N_GPUS = 4
DUR = 180.0


def w(f, **ev):
    f.write(json.dumps(ev, separators=(",", ":")) + "\n")


def conventional(path: Path, rng: random.Random):
    """All 4 GPUs alternate: generate together, then train together. The bubble is the
    generation long tail — the running batch drains and GPUs go idle at different times."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w(f, t=0.0, kind="run", detail={"arm": "conventional", "n_gpus": N_GPUS,
                                        "synthetic": True})
        t, step = 0.0, 0
        while t < DUR:
            # ---- generation phase: all GPUs generating, then draining one by one ----
            for g in range(N_GPUS):
                w(f, t=round(t, 3), kind="gpu", gpu=g, role="hybrid", state="generating",
                  detail={"seqs": 64})
            gen_len = rng.uniform(9.0, 12.0)
            # stragglers: GPUs finish at spread-out times -> the tail
            finish = sorted(rng.uniform(0.45, 1.0) * gen_len for _ in range(N_GPUS))
            for g, fin in enumerate(finish):
                w(f, t=round(t + fin, 3), kind="gpu", gpu=g, role="hybrid", state="idle")
            for _ in range(96):
                w(f, t=round(t + rng.uniform(0, gen_len), 3), kind="sample",
                  reward=1.0 if rng.random() < min(0.55, 0.14 + step * 0.012) else 0.0,
                  tokens=rng.randint(120, 480), lag=0)
            t += gen_len
            # ---- stop-the-world weight/memory swap ----
            for g in range(N_GPUS):
                w(f, t=round(t, 3), kind="gpu", gpu=g, role="hybrid", state="weight_sync")
            t += 1.4
            # ---- training phase ----
            for st in ("forward", "backward", "optimizer_step"):
                for g in range(N_GPUS):
                    w(f, t=round(t, 3), kind="gpu", gpu=g, role="hybrid", state=st,
                      detail={"step": step})
                t += {"forward": 1.5, "backward": 2.4, "optimizer_step": 0.5}[st]
            w(f, t=round(t, 3), kind="weights", step=step, src=[0, 1], dst=[2, 3],
              bytes=988_000_000, ms=980, inflight=False)
            t += 1.0
            step += 1
        w(f, t=round(t, 3), kind="end")


def pipeline(path: Path, rng: random.Random):
    """GPUs 0-1 train forever, GPUs 2-3 generate forever. Weights move in flight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w(f, t=0.0, kind="run", detail={"arm": "pipeline", "n_gpus": N_GPUS,
                                        "synthetic": True})
        for g in (2, 3):
            w(f, t=0.0, kind="gpu", gpu=g, role="infer", state="generating",
              detail={"seqs": 64})
        t, step = 0.0, 0
        while t < DUR:
            for st, dur in (("forward", 1.5), ("backward", 2.4), ("optimizer_step", 0.5)):
                for g in (0, 1):
                    w(f, t=round(t, 3), kind="gpu", gpu=g, role="train", state=st,
                      detail={"step": step})
                t += dur
            # in-flight update: inference GPUs blip to weight_sync and immediately resume
            w(f, t=round(t, 3), kind="weights", step=step, src=[0, 1], dst=[2, 3],
              bytes=988_000_000, ms=310, inflight=True)
            for g in (2, 3):
                w(f, t=round(t, 3), kind="gpu", gpu=g, role="infer", state="weight_sync")
                w(f, t=round(t + 0.31, 3), kind="gpu", gpu=g, role="infer",
                  state="generating", detail={"seqs": 64})
            # samples stream continuously, with a small staleness spread
            for _ in range(int(rng.uniform(28, 36))):
                w(f, t=round(t + rng.uniform(0, 4.4), 3), kind="sample",
                  reward=1.0 if rng.random() < min(0.55, 0.14 + step * 0.012) else 0.0,
                  tokens=rng.randint(120, 480), lag=rng.randint(0, 3))
            t += 0.4
            step += 1
        w(f, t=round(t, 3), kind="end")


if __name__ == "__main__":
    rng = random.Random(0)
    conventional(OUT / "conventional" / "events.jsonl", rng)
    pipeline(OUT / "pipeline" / "events.jsonl", rng)
    for arm in ("conventional", "pipeline"):
        p = OUT / arm / "events.jsonl"
        print(f"{arm:14s} {sum(1 for _ in p.open()):6d} events -> {p}")
    print("\n!! SYNTHETIC — for visualizer development only, never for figures !!")
