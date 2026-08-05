"""Figures, computed from events.jsonl only.

Deliberately the SAME input the visualizer reads, so a plot and the animation can never
tell different stories. Nothing here recomputes or re-simulates anything — if a number is
not in the event stream it does not appear in a figure.

    python3 -m race.metrics viz/runs --out reports/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# States that represent a GPU actually DOING something. `waiting` is deliberately excluded:
# a GPU blocked on an empty queue is not working, and counting it as busy reported 97.8%
# utilisation for an arm in which one GPU sat at waiting=100% and did no compute at all.
# `occupied` (the weaker notion — has work assigned) is reported separately so the two are
# never confused.
COMPUTE = ("generating", "prefill", "forward", "backward", "optimizer_step", "weight_sync")
BUSY = lambda s: s in COMPUTE            # noqa: E731
OCCUPIED = lambda s: s not in ("idle", None)  # noqa: E731


def load(path: Path) -> list[dict]:
    """Load and SORT by t.

    telemetry.py stamps t under its lock so a live stream is monotonic, but sorting here is
    cheap insurance: any out-of-order line would be integrated as a forward dt twice and
    silently inflate GPU-seconds rather than failing loudly.
    """
    ev = [json.loads(l) for l in path.open() if l.strip()]
    ev.sort(key=lambda e: e["t"])
    return ev


def summarise(events: list[dict]) -> dict:
    """Reduce one arm's stream to the numbers the comparison rests on."""
    run = next((e for e in events if e["kind"] == "run"), {})
    n_gpus = run.get("detail", {}).get("n_gpus", 0)
    synthetic = bool(run.get("detail", {}).get("synthetic"))

    # Busy/idle is measured from the "ready" marker, not from t=0. Everything before it is
    # one-off engine startup (weight load, torch.compile, CUDA-graph capture) — real cost,
    # but not the scheduling bubble this experiment is about, and identical in both arms.
    # Counting it made gpu_busy_frac read 8% on a 3-minute smoke, i.e. the startup swamped
    # the signal. Reported separately as setup_s so it is disclosed rather than discarded.
    ready_t = next((e["t"] for e in events
                    if e["kind"] == "phase" and e.get("name") == "ready"), 0.0)

    state, last_t = {}, ready_t
    busy_s = idle_s = 0.0
    per_gpu_busy = {g: 0.0 for g in range(n_gpus)}

    samples, lags, rewards_t = [], [], []
    updates, inflight_updates, weight_ms = 0, 0, []

    for e in events:
        t = e["t"]
        if t < ready_t:          # replay setup events for state, but do not accrue time
            if e["kind"] == "gpu":
                state[e["gpu"]] = e["state"]
            continue
        dt = max(0.0, t - last_t)
        if dt:
            for g in range(n_gpus):
                st = state.get(g)
                if BUSY(st):
                    busy_s += dt
                    per_gpu_busy[g] += dt
                else:
                    idle_s += dt
        last_t = t

        k = e["kind"]
        if k == "gpu":
            state[e["gpu"]] = e["state"]
        elif k == "sample":
            samples.append(e["reward"])
            lags.append(e.get("lag", 0))
            rewards_t.append((t - ready_t, e["reward"]))
        elif k == "weights":
            updates += 1
            inflight_updates += 1 if e.get("inflight") else 0
            weight_ms.append(e.get("ms", 0.0))

    total_gpu_s = busy_s + idle_s
    dur = (last_t - ready_t) or 1.0      # steady-state duration

    # INVARIANT: every GPU is in exactly one state at every instant, so the GPU-seconds we
    # accounted for must equal the cluster's capacity over the run. If this trips, the
    # event stream is out of order or a GPU changed state without emitting — either way the
    # busy/idle split is fiction and must not be plotted.
    capacity = n_gpus * dur
    if capacity and abs(total_gpu_s - capacity) > 0.01 * capacity:
        raise ValueError(
            f"GPU-second accounting is off: accounted {total_gpu_s:.1f} vs capacity "
            f"{capacity:.1f} ({n_gpus} GPUs x {dur:.1f}s). The event stream is likely "
            f"out of order — refusing to report a busy/idle split from it.")

    return {
        "synthetic": synthetic,
        "n_gpus": n_gpus,
        "duration_s": round(dur, 2),
        "setup_s": round(ready_t, 2),
        "wallclock_total_s": round(last_t, 2),
        "samples": len(samples),
        # mean over the WHOLE run — a summary statistic, NOT the answer to "who learned
        # more". It conflates early and late training, and an arm that generated twice as
        # many samples spends proportionally longer at its early, weaker policy.
        "mean_reward": round(sum(samples) / len(samples), 4) if samples else None,
        # THE headline: reward over the final 10% of samples, i.e. how good the policy
        # actually was when the clock ran out. This is what "learned more per second" means.
        "final_reward": (round(sum(samples[-max(1, len(samples) // 10):])
                               / max(1, len(samples) // 10), 4) if samples else None),
        # Control: reward over the first N samples where N = the SMALLEST arm's sample
        # count, so arms can also be compared at matched sample budget rather than matched
        # wall-clock. Filled in by compare(), which alone knows every arm's size.
        "_all_rewards": samples,
        "samples_per_s": round(len(samples) / dur, 3),
        "gpu_busy_frac": round(busy_s / total_gpu_s, 4) if total_gpu_s else 0.0,
        "idle_gpu_s": round(idle_s, 1),
        "busy_gpu_s": round(busy_s, 1),
        # Per-GPU compute share. An aggregate can look healthy while one GPU does nothing —
        # exactly what happened in race4, where a train GPU sat at waiting=100% for the
        # whole run and the aggregate still read 97.8%.
        "per_gpu_busy_frac": {g: round(per_gpu_busy[g] / dur, 4) for g in range(n_gpus)},
        "unused_gpus": [g for g in range(n_gpus) if per_gpu_busy[g] / dur < 0.02],
        "weight_updates": updates,
        "inflight_updates": inflight_updates,
        "mean_weight_ms": round(sum(weight_ms) / len(weight_ms), 1) if weight_ms else None,
        "mean_lag": round(sum(lags) / len(lags), 3) if lags else None,
        "max_lag": max(lags) if lags else None,
        "_reward_series": rewards_t,
    }


def reward_vs_time(series, window: int = 200):
    """Running mean reward against wall-clock — THE headline plot."""
    xs, ys, acc = [], [], []
    for t, r in series:
        acc.append(r)
        if len(acc) > window:
            acc.pop(0)
        xs.append(t)
        ys.append(sum(acc) / len(acc))
    return xs, ys


# Three conditions. A->B isolates what concurrency buys; B->C isolates what in-flight
# weight updates buy on top of it — a separation the paper does not make.
ARMS = ("conventional", "pipeline", "pipeline_async")
LABEL = {"conventional": "conventional (alternating)",
         "pipeline": "concurrent, batch-boundary updates",
         "pipeline_async": "concurrent, in-flight updates"}
COLOR = {"conventional": "#8E6B2F", "pipeline": "#4A5D7E", "pipeline_async": "#8B3A3A"}


def compare(run_dir: Path, out_dir: Path) -> dict:
    arms = {}
    for arm in ARMS:
        p = run_dir / arm / "events.jsonl"
        if p.exists():
            arms[arm] = summarise(load(p))

    if any(a["synthetic"] for a in arms.values()):
        print("!! SYNTHETIC DATA — figures suppressed. These are fixtures, not results.")
        return arms

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PAPER, INK = "#F7F2E8", "#1F1B16"

        # --- headline: reward vs wall-clock ---
        fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=PAPER)
        ax.set_facecolor(PAPER)
        for arm, s in arms.items():
            xs, ys = reward_vs_time(s["_reward_series"])
            ax.plot(xs, ys, lw=2.1, color=COLOR[arm], label=LABEL.get(arm, arm))
        ax.set_xlabel("wall-clock seconds"); ax.set_ylabel("mean reward (running)")
        ax.set_title("Reward vs wall-clock — same GPUs, same algorithm", color=INK)
        ax.legend(frameon=False); ax.grid(alpha=.18)
        for sp in ax.spines.values():
            sp.set_color("#D8CFBE")
        fig.tight_layout(); fig.savefig(out_dir / "reward_vs_wallclock.png", dpi=200)

        # --- the bubble: GPU busy fraction ---
        fig, ax = plt.subplots(figsize=(5.4, 3.6), facecolor=PAPER)
        ax.set_facecolor(PAPER)
        names = list(arms)
        ax.bar([LABEL.get(a, a).replace(", ", ",\n") for a in names],
               [arms[a]["gpu_busy_frac"] * 100 for a in names],
               color=[COLOR[a] for a in names], width=.55)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("GPU busy %"); ax.set_ylim(0, 100)
        ax.set_title("How much of the cluster actually worked", color=INK)
        for sp in ax.spines.values():
            sp.set_color("#D8CFBE")
        fig.tight_layout(); fig.savefig(out_dir / "gpu_busy.png", dpi=200)
        print(f"figures -> {out_dir}")
    except ImportError:
        print("matplotlib unavailable — numbers only")

    # Matched-sample control: compare every arm over the first N samples, N = the smallest
    # arm's count. Matched wall-clock answers "who learned more per second"; matched samples
    # answers "who learned more per sample". They can disagree, and both belong in the
    # write-up — reporting only the flattering one is how throughput results get oversold.
    if arms:
        n_min = min(len(a.get("_all_rewards") or []) for a in arms.values())
        for a in arms.values():
            r = a.get("_all_rewards") or []
            if n_min:
                tail = r[:n_min][-max(1, n_min // 10):]
                a["reward_at_matched_samples"] = round(sum(tail) / len(tail), 4)
                a["matched_sample_budget"] = n_min

    for a in arms.values():
        a.pop("_reward_series", None)
        a.pop("_all_rewards", None)
    (out_dir / "summary.json").write_text(json.dumps(arms, indent=2))
    return arms


if __name__ == "__main__":
    rd = Path(sys.argv[1] if len(sys.argv) > 1 else "viz/runs")
    od = Path(sys.argv[2] if len(sys.argv) > 2 else "reports")
    res = compare(rd, od)
    print(json.dumps(res, indent=2))
