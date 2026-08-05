#!/usr/bin/env python3
"""Turn a completed race into publishable artifacts.

    python3 scripts/publish_run.py runs/race1

Does four things, and refuses rather than guesses if anything looks wrong:
  1. validates BOTH arms' event streams (the GPU-second invariant in race.metrics)
  2. checks the two arms' fairness keys actually match
  3. writes reports/<run>/ figures + summary.json
  4. copies the streams into viz/runs/ and the summary into dashboard/reference/

Refusing is the point. A run that fails validation must not reach a slide.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from race.metrics import ARMS, compare, load, summarise  # noqa: E402



def main(run_dir: Path) -> int:
    run_id = run_dir.name
    problems: list[str] = []

    # ---- 1. both arms present and parseable ------------------------------
    streams = {}
    for arm in ARMS:
        p = run_dir / arm / "events.jsonl"
        if not p.exists():
            problems.append(f"missing {p}")
            continue
        try:
            streams[arm] = summarise(load(p))
        except ValueError as e:      # the accounting invariant
            problems.append(f"{arm}: {e}")

    # ---- 2. the arms must be comparable ----------------------------------
    keys = {}
    for arm in ARMS:
        r = run_dir / arm / "result.json"
        if r.exists():
            d = json.loads(r.read_text())
            keys[arm] = d.get("fairness_key")
            if not d.get("ok"):
                problems.append(f"{arm}: run did not complete ok "
                                f"({d.get('error', 'unknown')})")
    if len(keys) == 2 and all(keys.values()):
        a, b = keys[ARMS[0]], keys[ARMS[1]]
        diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        if diff:
            problems.append(f"FAIRNESS VIOLATION — arms differ beyond the partition: {diff}")

    # ---- 3. no synthetic data may be published ---------------------------
    for arm, s in streams.items():
        if s.get("synthetic"):
            problems.append(f"{arm}: stream is marked synthetic — fixtures are not results")

    # ---- 3b. every arm must actually USE the GPUs it was given ------------
    # The arms are compared at a fixed GPU budget, so an arm that leaves a GPU idle is
    # racing with a handicap and any conclusion drawn from it is about the handicap, not the
    # architecture. This is not hypothetical: race4's concurrent arms were given
    # train=[0,1] while the Trainer only ever used gpu_ids[0], so GPU 1 sat at
    # waiting=100% for 45 minutes and the aggregate busy metric still read 97.8%.
    for arm, s in streams.items():
        if s.get("unused_gpus"):
            problems.append(
                f"{arm}: GPU(s) {s['unused_gpus']} did essentially no compute "
                f"(per-GPU busy: {s.get('per_gpu_busy_frac')}). The arm ran on fewer GPUs "
                f"than its budget, so it is not comparable to the others.")

    if problems:
        print("REFUSING TO PUBLISH:")
        for p in problems:
            print(f"  - {p}")
        return 1

    # ---- 4. publish -------------------------------------------------------
    reports = ROOT / "reports" / run_id
    # Use compare()'s output, not the local summarise() results: only compare() knows every
    # arm's sample count and can therefore fill the matched-sample control. Printing from
    # the local dict silently reported it as None.
    streams = compare(run_dir, reports)

    viz = ROOT / "viz" / "runs"
    for arm in ARMS:
        dst = viz / arm
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / arm / "events.jsonl", dst / "events.jsonl")

    ref = ROOT / "dashboard" / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reports / "summary.json", ref / "summary.json")

    print(f"\npublished {run_id}")
    print(f"  figures   -> {reports}")
    print(f"  visualizer-> {viz}")
    print(f"  dashboard -> {ref}")

    # ---- headline, printed plainly, ALL arms ------------------------------
    present = [a for a in ARMS if a in streams]
    w = 16
    print("\n===== HEADLINE =====")
    print("  " + " " * 20 + "".join(f"{a:>{w}}" for a in present))
    for k in ("duration_s", "setup_s", "samples", "samples_per_s",
              "mean_reward", "final_reward", "reward_at_matched_samples",
              "matched_sample_budget",
              "gpu_busy_frac", "idle_gpu_s", "weight_updates", "inflight_updates",
              "mean_weight_ms", "mean_lag", "max_lag"):
        print(f"  {k:20s}" + "".join(f"{streams[a].get(k)!s:>{w}}" for a in present))

    # Ratios against the conventional baseline. Reported separately, never multiplied into
    # a single "learning speed" number — throughput and effectiveness trade off, and
    # collapsing them hides which one moved.
    if "conventional" in streams:
        base = streams["conventional"]
        print("\n  vs conventional:")
        for a in present:
            if a == "conventional":
                continue
            s = streams[a]
            tp = s["samples_per_s"] / max(base["samples_per_s"], 1e-9)
            fr = ((s["final_reward"] / max(base["final_reward"], 1e-9))
                  if s.get("final_reward") and base.get("final_reward") else float("nan"))
            ms = ((s.get("reward_at_matched_samples", 0)
                   / max(base.get("reward_at_matched_samples", 1e-9), 1e-9))
                  if s.get("reward_at_matched_samples") else float("nan"))
            print(f"    {a:16s} throughput {tp:5.3f}x   "
                  f"FINAL reward {fr:5.3f}x   @matched-samples {ms:5.3f}x")

    # B -> C: what in-flight updates buy on top of concurrency.
    if "pipeline" in streams and "pipeline_async" in streams:
        b, c2 = streams["pipeline"], streams["pipeline_async"]
        print("\n  in-flight effect (pipeline_async vs pipeline):")
        print(f"    throughput     {c2['samples_per_s'] / max(b['samples_per_s'],1e-9):5.3f}x")
        if b.get("mean_reward") and c2.get("mean_reward"):
            print(f"    reward/sample  {c2['mean_reward'] / max(b['mean_reward'],1e-9):5.3f}x")
        print(f"    mean lag       {b.get('mean_lag')} -> {c2.get('mean_lag')}")
        print(f"    inflight upd   {b.get('inflight_updates')} -> "
              f"{c2.get('inflight_updates')} / {c2.get('weight_updates')}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
