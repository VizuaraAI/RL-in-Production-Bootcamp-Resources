"""Result figures + animations, ALL computed from the real training runs.

Palette matches the veRL/PipelineRL lecture deck (warm cream #F7F2E8) so these
drop straight into the bonus Slidev deck next to the PaperBanana concept art.

    python scripts/make_result_figures.py A     # navigation figures (needs results/Q.npy)
    python scripts/make_result_figures.py B     # gait figures (needs results/gait_Q.npy)
    python scripts/make_result_figures.py anim  # the GIF animations (slower)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from swimmer.env import ACTION_NAMES, VortexWorld           # noqa: E402
from swimmer.qlearn import evaluate                          # noqa: E402
from swimmer import three_sphere as ts                       # noqa: E402

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
RES = ROOT / "results"

PAPER, PANEL, INK, INK3, RULE = "#F7F2E8", "#FDFAF3", "#1F1B16", "#7C7165", "#D8CFBE"
BLUE, RED, GOLD, GREEN, PLUM, GREY = "#4A5D7E", "#8B3A3A", "#8E6B2F", "#3D5A4A", "#6B4E7E", "#C9BFAC"


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PAPER)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK3, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK3, fontsize=9.5)
    ax.grid(alpha=.16, color=RULE)
    ax.tick_params(colors=INK3, labelsize=8.5)
    for s in ax.spines.values():
        s.set_color(RULE)


# ============================ PROBLEM A: navigation ==========================
# Zermelo point-to-point navigation: naive "aim at target" vs Q-learned detours.

def _flow_background(ax, alpha=.35):
    gx, gy = np.meshgrid(np.linspace(0, 2 * np.pi, 200), np.linspace(0, 2 * np.pi, 200))
    speed = np.hypot(np.cos(gx) * np.sin(gy), np.sin(gx) * np.cos(gy))
    ax.imshow(speed, extent=[0, 2 * np.pi, 0, 2 * np.pi], origin="lower",
              cmap="Blues", alpha=alpha, vmin=0, vmax=1.15)
    sx, sy = np.meshgrid(np.linspace(.2, 2 * np.pi - .2, 16),
                         np.linspace(.2, 2 * np.pi - .2, 16))
    ax.quiver(sx, sy, np.cos(sx) * np.sin(sy), -np.sin(sx) * np.cos(sy),
              color=BLUE, alpha=.45, scale=26, width=.0035)


def _mark_endpoints(ax, world):
    ax.scatter(*world.start, marker="o", s=120, c=GOLD, edgecolors=INK, zorder=6)
    ax.annotate("start", world.start, xytext=(8, -14), textcoords="offset points",
                color=INK, fontsize=9, weight="bold")
    circ = plt.Circle(world.target, world.target_r, fill=False, color=GREEN, lw=2.2, zorder=6)
    ax.add_patch(circ)
    ax.annotate("target", world.target, xytext=(10, 8), textcoords="offset points",
                color=GREEN, fontsize=9, weight="bold")


def figs_navigation():
    from swimmer.navigate import NavWorld, N_ACT, HEAD_VEC, evaluate_nav
    Q = np.load(RES / "nav_Q.npy")
    hist = np.load(RES / "nav_history.npy")

    # ---- A1: trajectories, naive vs smart (THE picture) ---------------------
    for name, pol, col in [("naive", "naive", PLUM), ("smart", Q, RED)]:
        r = evaluate_nav(pol, n=14, seed=17, record_traj=14)
        traj = r["traj"]                          # (T, k, 3): x, y, done
        fig, ax = plt.subplots(figsize=(5.9, 5.9), facecolor=PAPER)
        _flow_background(ax)
        world = NavWorld(1)
        for k in range(traj.shape[1]):
            alive = ~traj[:, k, 2].astype(bool)
            x, y = traj[:, k, 0], traj[:, k, 1]
            keep = np.concatenate([[True], (np.abs(np.diff(x)) < np.pi) &
                                            (np.abs(np.diff(y)) < np.pi)])
            xm = np.ma.array(x, mask=~(keep & np.concatenate([[True], alive[1:]])))
            ax.plot(xm, y, lw=1.25, color=col, alpha=.8, zorder=4)
        _mark_endpoints(ax, world)
        ttl = ("naive — aim straight at the target" if name == "naive"
               else "smart — the learned detours")
        style(ax, ttl, "", "")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
        fig.tight_layout(); fig.savefig(OUT / f"nav_traj_{name}.png", dpi=200,
                                        facecolor=PAPER)
        plt.close(fig)
        print(f"  nav_traj_{name}.png")

    # ---- A2: the learned policy as a vector field over the map --------------
    world = NavWorld(1)
    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor=PAPER)
    _flow_background(ax, alpha=.25)
    g = world.grid
    cx = (np.arange(g) + .5) * 2 * np.pi / g
    CX, CY = np.meshgrid(cx, cx)
    best = Q.argmax(axis=1).reshape(g, g)         # (gy, gx)
    Uq = HEAD_VEC[best][:, :, 0]; Vq = HEAD_VEC[best][:, :, 1]
    visited = (np.abs(Q).sum(axis=1) > 1e-9).reshape(g, g)
    ax.quiver(CX[visited], CY[visited], Uq[visited], Vq[visited],
              color=RED, scale=20, width=.006, zorder=5)
    _mark_endpoints(ax, world)
    style(ax, "the entire trained brain — one arrow per map cell", "", "")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
    fig.tight_layout(); fig.savefig(OUT / "nav_policy_field.png", dpi=200, facecolor=PAPER)
    plt.close(fig)
    print("  nav_policy_field.png")

    # ---- A3: the headline numbers -------------------------------------------
    import json
    summ = json.loads((RES / "nav_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), facecolor=PAPER)
    names = ["naive", "smart"]; cols = [PLUM, RED]
    axes[0].bar(["aim at target", "learned"],
                [100 * summ[n]["arrival_rate"] for n in names], color=cols, width=.55)
    axes[0].set_ylim(0, 105)
    style(axes[0], "robots that ever arrive", "", "%")
    axes[1].bar(["aim at target", "learned"],
                [summ[n]["median_time"] for n in names], color=cols, width=.55)
    style(axes[1], "median journey time (arrivers only)", "", "decisions")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout(); fig.savefig(OUT / "nav_headline.png", dpi=200, facecolor=PAPER)
    plt.close(fig)
    print("  nav_headline.png")

    # ---- A4: learning curve --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.8), facecolor=PAPER)
    gen, arr = hist[:, 0], 100 * hist[:, 1]
    ax.plot(gen, arr, lw=2.0, color=RED)
    ax.axhline(100 * summ["naive"]["arrival_rate"], color=PLUM, lw=1.6, ls="--")
    ax.text(gen[-1], 100 * summ["naive"]["arrival_rate"] - 5, "naive baseline",
            ha="right", color=PLUM, fontsize=9)
    ax.set_ylim(0, 103)
    style(ax, "arrival rate during training", "generation", "% of robots arriving")
    fig.tight_layout(); fig.savefig(OUT / "nav_learning_curve.png", dpi=200,
                                    facecolor=PAPER)
    plt.close(fig)
    print("  nav_learning_curve.png")


def anim_navigation(fps=16):
    """Side-by-side GIF: naive vs smart robot swarms racing to the target."""
    from swimmer.navigate import NavWorld
    Q = np.load(RES / "nav_Q.npy")
    n = 34
    worlds = {"naive — aim at target": (NavWorld(n, rng=np.random.default_rng(5)), PLUM),
              "smart — learned policy": (NavWorld(n, rng=np.random.default_rng(5)), RED)}
    for w, _ in worlds.values():
        w.reset()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.1), facecolor=PAPER)
    arts = {}
    for ax, (name, (world, col)) in zip(axes, worlds.items()):
        _flow_background(ax, alpha=.30)
        _mark_endpoints(ax, world)
        sc = ax.scatter(world.x, world.y, s=30, c=col, edgecolors=INK,
                        linewidths=.4, zorder=5)
        ttl = ax.set_title("", color=INK, fontsize=10.5, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
        for s_ in ax.spines.values():
            s_.set_color(RULE)
        arts[name] = (sc, ttl, world, col)
    fig.suptitle("same current, same start, same speed — 34 robots each",
                 color=INK3, fontsize=10)

    def frame(i):
        for name, (sc, ttl, world, col) in arts.items():
            if not world.done.all():
                a = (world.bearing_action() if "naive" in name
                     else Q[world.observe()].argmax(axis=1))
                world.act(a)
            live = ~world.done | world.arrived
            sc.set_offsets(np.c_[world.x, world.y])
            sizes = np.where(world.arrived, 46, np.where(world.done, 6, 30))
            sc.set_sizes(sizes)
            ttl.set_text(f"{name}   arrived: {world.arrived.sum()}/{world.n}")
        return []

    an = FuncAnimation(fig, frame, frames=170, blit=False)
    an.save(OUT / "nav_race.gif", writer=PillowWriter(fps=fps), dpi=82,
            savefig_kwargs={"facecolor": PAPER})
    plt.close(fig)
    print("  nav_race.gif")


# ============================ PROBLEM B: the gait ============================

def figs_gait():
    Q = np.load(RES / "gait_Q.npy")
    hist = np.load(RES / "gait_history.npy")

    # ---- B1: displacement, learned vs reciprocal (scallop theorem live) ----
    learned = ts.rollout(Q, steps=24)
    recip = ts.rollout("reciprocal", steps=24)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor=PAPER)
    ax.plot(range(len(learned["cm"])), learned["cm"], lw=2.4, color=RED, marker="o",
            ms=3.5, label="learned gait — the Najafi-Golestanian cycle")
    ax.plot(range(len(recip["cm"])), recip["cm"], lw=2.4, color=GREY, marker="o",
            ms=3.5, label="reciprocal gait — scallop theorem says: zero")
    ax.axhline(0, color=INK3, lw=.8, ls=":")
    style(ax, "Centre-of-mass displacement over 24 strokes", "stroke number",
          "distance swum")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "gait_displacement.png", dpi=200, facecolor=PAPER)
    plt.close(fig)
    print("  gait_displacement.png")

    # ---- B2: learning curve -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.8), facecolor=PAPER)
    ep, disp, _ = hist.T
    ax.plot(ep, disp, lw=.8, color=RULE)
    w = 15
    ax.plot(ep[w - 1:], np.convolve(disp, np.ones(w) / w, "valid"), lw=2.2, color=RED)
    ax.axhline(0, color=INK3, lw=.8, ls=":")
    style(ax, "Discovering the gait (training)", "episode",
          "displacement per episode (24 strokes)")
    fig.tight_layout(); fig.savefig(OUT / "gait_learning_curve.png", dpi=200, facecolor=PAPER)
    plt.close(fig)
    print("  gait_learning_curve.png")


def anim_gait(fps=24):
    """GIF: learned gait crawling vs reciprocal flapping in place."""
    Q = np.load(RES / "gait_Q.npy")
    rolls = {"learned gait": (ts.rollout(Q, steps=16), RED),
             "reciprocal (scallop)": (ts.rollout("reciprocal", steps=16), GREY)}

    # stitch stroke trajectories: (frames, 3 sphere positions), subsample
    stitched = {}
    for name, (roll, col) in rolls.items():
        frames = np.concatenate(roll["trajs"], axis=0)[::6]
        stitched[name] = (frames, col, roll["cm"])

    n_frames = min(len(v[0]) for v in stitched.values())
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 3.9), facecolor=PAPER)
    arts = {}
    for ax, (name, (frames, col, cm)) in zip(axes, stitched.items()):
        ax.set_facecolor(PAPER)
        ax.set_xlim(-1.8, 2.2); ax.set_ylim(-.55, .55)
        ax.set_yticks([]); ax.tick_params(colors=INK3, labelsize=8)
        for s in ax.spines.values():
            s.set_color(RULE)
        ax.axvline(0, color=RULE, lw=.8, ls=":")
        x0 = frames[0]
        line, = ax.plot([x0[0], x0[2]], [0, 0], lw=2.4, color=INK3, zorder=2)
        dots = ax.scatter(x0, np.zeros(3), s=[260, 260, 260], c=col, edgecolors=INK,
                          linewidths=.8, zorder=3)
        lab = ax.text(.995, .82, name, transform=ax.transAxes, ha="right",
                      color=INK, fontsize=10, weight="bold")
        cmtxt = ax.text(.995, .58, "", transform=ax.transAxes, ha="right",
                        color=INK3, fontsize=9)
        arts[name] = (line, dots, cmtxt, frames)
    fig.suptitle("16 strokes each — only one of them goes anywhere",
                 color=INK3, fontsize=10)
    fig.tight_layout()

    def frame(i):
        for name, (line, dots, cmtxt, frames) in arts.items():
            x = frames[min(i, len(frames) - 1)]
            line.set_data([x[0], x[2]], [0, 0])
            dots.set_offsets(np.c_[x, np.zeros(3)])
            cmtxt.set_text(f"centre of mass: {x.mean():+.3f}")
        return []

    an = FuncAnimation(fig, frame, frames=n_frames, blit=False)
    an.save(OUT / "gait_race.gif", writer=PillowWriter(fps=fps), dpi=85,
            savefig_kwargs={"facecolor": PAPER})
    plt.close(fig)
    print("  gait_race.gif")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("A", "all"):
        figs_navigation()
    if which in ("B", "all"):
        figs_gait()
    if which in ("anim", "all"):
        anim_gait()
        anim_navigation()
