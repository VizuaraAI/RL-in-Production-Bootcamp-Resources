#!/usr/bin/env python3
"""The four graphs the lecture says decide the argument (deck §12.3).

    .venv-plot/bin/python scripts/four_graphs.py runs/race5 reports/race5

  1. Reward vs wall-clock   — the headline: does the pipeline reach a given reward sooner?
  2. Reward vs samples      — the control: if these overlap, the speedup is PURE THROUGHPUT.
  3. GPU utilisation over time — the mechanism: the conventional bubble should be visible.
  4. Staleness              — lag distribution and how off-policy the data actually was.

Everything is derived from events.jsonl, the same stream the visualizer reads, so a slide
and the animation can never disagree. Figures are styled to match the deck.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from race.metrics import BUSY, COLOR, LABEL, load  # noqa: E402

PAPER, PANEL, INK, INK3, RULE = "#F7F2E8", "#FDFAF3", "#1F1B16", "#7C7165", "#D8CFBE"
ARMS = ("conventional", "pipeline", "pipeline_async")
SHORT = {"conventional": "conventional",
         "pipeline": "concurrent (batch-boundary)",
         "pipeline_async": "concurrent (in-flight)"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(PAPER)
    ax.set_title(title, color=INK, fontsize=12, pad=10, loc="left")
    ax.set_xlabel(xlabel, color=INK3, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK3, fontsize=9.5)
    ax.grid(alpha=.16, color=RULE)
    ax.tick_params(colors=INK3, labelsize=8.5)
    for s in ax.spines.values():
        s.set_color(RULE)


def running(vals, w):
    """Running mean with window w — smooths binary 0/1 rewards into a visible curve."""
    out, acc = [], []
    for v in vals:
        acc.append(v)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def read(run_dir: Path):
    data = {}
    for arm in ARMS:
        p = run_dir / arm / "events.jsonl"
        if not p.exists():
            continue
        ev = load(p)
        ready = next((e["t"] for e in ev
                      if e["kind"] == "phase" and e.get("name") == "ready"), 0.0)
        ngpu = next(e for e in ev if e["kind"] == "run")["detail"]["n_gpus"]
        data[arm] = {"ev": ev, "ready": ready, "ngpu": ngpu}
    return data


# ---------------------------------------------------------------- graphs 1 & 2
def g1_g2(data, out: Path):
    for kind, xlab, fname, title in (
        ("time", "wall-clock seconds (after engine start-up)", "g1_reward_vs_wallclock.png",
         "1 · Reward vs wall-clock — the headline"),
        ("samples", "training samples generated", "g2_reward_vs_samples.png",
         "2 · Reward vs samples — the control"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=PAPER)
        lo, hi = 1.0, 0.0
        NB = 40
        for arm, d in data.items():
            xs, ys = [], []
            n = 0
            for e in d["ev"]:
                if e["kind"] != "sample" or e["t"] < d["ready"]:
                    continue
                n += 1
                xs.append(e["t"] - d["ready"] if kind == "time" else n)
                ys.append(e["reward"])
            if not xs:
                continue
            # Fixed-width BINS, not a running mean. Reward is binary, so a running mean is
            # dominated by noise and its warm-up produces a spurious spike at t=0 (the first
            # sample is literally 0 or 1). Binning gives each point an equal, stated sample
            # size, and the +/-1 s.e. band shows when a gap is real rather than noise.
            edges = [xs[0] + i * (xs[-1] - xs[0]) / NB for i in range(NB + 1)]
            bx, by, be = [], [], []
            j = 0
            for b in range(NB):
                acc = []
                while j < len(xs) and xs[j] <= edges[b + 1]:
                    acc.append(ys[j])
                    j += 1
                if len(acc) < 30:
                    continue
                m = sum(acc) / len(acc)
                bx.append((edges[b] + edges[b + 1]) / 2)
                by.append(m)
                be.append((m * (1 - m) / len(acc)) ** 0.5)   # binomial s.e.
            if not bx:
                continue
            ax.plot(bx, by, lw=2.0, color=COLOR[arm], label=SHORT[arm])
            ax.fill_between(bx, [m - s for m, s in zip(by, be)],
                            [m + s for m, s in zip(by, be)], color=COLOR[arm], alpha=.14,
                            linewidth=0)
            lo = min(lo, min(m - s for m, s in zip(by, be)))
            hi = max(hi, max(m + s for m, s in zip(by, be)))
        pad = (hi - lo) * 0.18 or 0.02
        ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
        style(ax, title, xlab, "mean reward (binned, ±1 s.e.)")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200, facecolor=PAPER)
        plt.close(fig)
        print(f"  {fname}")


# ------------------------------------------------------------------- graph 3
def g3(data, out: Path):
    """GPU utilisation over time — the bubble made visible.

    Fraction of GPUs actually COMPUTING, sampled on a fixed grid. `waiting` is not
    computing (a GPU blocked on an empty queue is idle from the cluster's point of view).
    """
    # One panel per arm, stacked and sharing an x-axis. Overlaying three sawtooths made the
    # figure unreadable and buried the point: the conventional trace should visibly SAG on
    # every generate/train alternation, and the in-flight trace should not. Filled area
    # makes the wasted region literal — it is the white space above each curve.
    arms = list(data)
    fig, axes = plt.subplots(len(arms), 1, figsize=(7.6, 5.6), facecolor=PAPER,
                             sharex=True, sharey=True)
    if len(arms) == 1:
        axes = [axes]
    for ax, arm in zip(axes, arms):
        d = data[arm]
        ev, ready, ngpu = d["ev"], d["ready"], d["ngpu"]
        end = ev[-1]["t"]
        grid = [ready + i * (end - ready) / 900 for i in range(901)]
        state, gi, busy_series = {}, 0, []
        for e in ev:
            while gi < len(grid) and e["t"] > grid[gi]:
                busy_series.append(sum(1 for g in range(ngpu) if BUSY(state.get(g))) / ngpu)
                gi += 1
            if e["kind"] == "gpu":
                state[e["gpu"]] = e["state"]
        while gi < len(grid):
            busy_series.append(sum(1 for g in range(ngpu) if BUSY(state.get(g))) / ngpu)
            gi += 1
        xs = [g - ready for g in grid][:len(busy_series)]
        ys = [b * 100 for b in running(busy_series, 5)]
        ax.fill_between(xs, ys, color=COLOR[arm], alpha=.30, linewidth=0)
        ax.plot(xs, ys, lw=1.0, color=COLOR[arm])
        mean = sum(ys) / max(len(ys), 1)
        ax.axhline(mean, color=INK3, lw=.9, ls=":")
        ax.text(0.995, 0.08, f"{SHORT[arm]} — mean {mean:.0f}% computing",
                transform=ax.transAxes, ha="right", fontsize=9, color=INK)
        ax.set_facecolor(PAPER)
        ax.set_ylim(0, 105)
        ax.grid(alpha=.16, color=RULE)
        ax.tick_params(colors=INK3, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(RULE)
    axes[0].set_title("3 · GPU utilisation over time — the mechanism",
                      color=INK, fontsize=12, pad=10, loc="left")
    axes[-1].set_xlabel("wall-clock seconds", color=INK3, fontsize=9.5)
    axes[len(axes) // 2].set_ylabel("% of GPUs computing", color=INK3, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "g3_gpu_utilisation.png", dpi=200, facecolor=PAPER)
    plt.close(fig)
    print("  g3_gpu_utilisation.png")


# ------------------------------------------------------------------- graph 4
def g4(data, out: Path):
    """Staleness: how far off-policy each arm's training data actually was.

    LEFT  — distribution of per-sample lag (optimizer steps between generation and use).
    RIGHT — clip fraction over training, the *effect* of that staleness on the loss: the
            share of tokens whose importance ratio left the trust region. This is the
            observable proxy for effective-sample-size loss. True ESS needs per-token
            importance weights, which are not in this run's telemetry — grpo.py now logs
            `ess_frac` so subsequent runs report it directly, and that is stated rather
            than silently substituted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), facecolor=PAPER)

    ax = axes[0]
    width = 0.26
    for i, (arm, d) in enumerate(data.items()):
        lags = [e.get("lag", 0) for e in d["ev"]
                if e["kind"] == "sample" and e["t"] >= d["ready"]]
        if not lags:
            continue
        n = len(lags)
        buckets = [0, 1, 2]
        fracs = [sum(1 for l in lags if l == b) / n for b in buckets]
        fracs.append(sum(1 for l in lags if l > 2) / n)
        pos = [b + (i - 1) * width for b in range(4)]
        ax.bar(pos, [f * 100 for f in fracs], width=width, color=COLOR[arm],
               label=SHORT[arm])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["lag 0\n(on-policy)", "lag 1", "lag 2", "lag >2"])
    style(ax, "4a · Staleness distribution", "", "% of samples")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    for arm, d in data.items():
        xs = [e["step"] for e in d["ev"] if e["kind"] == "metric" and "clip_frac" in e]
        ys = [e["clip_frac"] for e in d["ev"] if e["kind"] == "metric" and "clip_frac" in e]
        if not xs:
            continue
        ax.plot(xs, [v * 100 for v in running(ys, 15)], lw=1.8, color=COLOR[arm],
                label=SHORT[arm])
    style(ax, "4b · Effect on the loss — clipped tokens",
          "optimizer step", "% tokens outside trust region")
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out / "g4_staleness.png", dpi=200, facecolor=PAPER)
    plt.close(fig)
    print("  g4_staleness.png")


def main(run_dir: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    data = read(run_dir)
    if not data:
        print(f"no arms found under {run_dir}")
        return 1
    print(f"figures from {run_dir} ({', '.join(data)}) ->")
    g1_g2(data, out)
    g3(data, out)
    g4(data, out)
    return 0


if __name__ == "__main__":
    rd = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/race5")
    od = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/race5")
    raise SystemExit(main(rd, od))
