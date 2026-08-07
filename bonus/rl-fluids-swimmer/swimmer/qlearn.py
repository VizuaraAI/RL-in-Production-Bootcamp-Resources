"""Tabular Q-learning for the smart swimmer — the ENTIRE brain is a 12x4 table.

This is deliberately the exact algorithm from the bootcamp's Lecture 02 (DQN's
tabular ancestor), applied to a physics problem instead of a game:

    Q[s, a]  <-  Q[s, a] + lr * ( r + gamma * max_a' Q[s', a']  -  Q[s, a] )

48 numbers. That is the whole policy. The point of this bonus lecture is that
the LOOP is identical to what trains a 1.5B-parameter language model with GRPO —
observe, act, get a verifiable reward, update — only the policy shrank from a
transformer to a lookup table, and the environment grew from "a maths dataset"
to "the Navier-Stokes solution being carried under your feet".

Training is vectorised: N swimmers explore in parallel, all reading and writing
one shared Q-table (updates within a batch are averaged per (s, a) cell).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import N_ACTIONS, VortexWorld


def train(n_swimmers=256, rounds=800, decisions_per_round=200,
          lr=0.10, lr_rho=0.01, eps_start=0.5, eps_end=0.05,
          seed=0, verbose=True, eval_every=100, **env_kw):
    """DIFFERENTIAL (average-reward) Q-learning — the continuing-task form.

    Why not ordinary discounted Q-learning here? The swimmer's task never ends, so
    with gamma near 1 every Q[s, a] inflates toward the same enormous baseline
    (~ mean_reward / (1 - gamma)); the tiny BETWEEN-ACTION differences that argmax
    actually needs get buried under reward noise. The average-reward form
    (Sutton & Barto, ch. 10) subtracts a learned mean reward rho each step:

        Q[s,a] += lr * ( r - rho + max_a' Q[s',a'] - Q[s,a] )
        rho    += lr_rho * ( r - rho )          (on greedy steps)

    so the table stores pure ADVANTAGES — perfectly conditioned for a continuing
    task, and with no gamma to tune against the physics timescales at all.

    Returns (Q, history): rows (round, behaviour rise/decision, eps, greedy_eval).
    greedy_eval is a quick no-exploration probe every `eval_every` rounds (else nan).
    """
    rng = np.random.default_rng(seed)
    world = VortexWorld(n_swimmers, rng=rng, **env_kw)
    Q = np.zeros((world.N_STATES, N_ACTIONS))
    rho = 0.0
    history = []

    for rnd in range(rounds):
        eps = eps_start + (eps_end - eps_start) * rnd / max(rounds - 1, 1)
        s = world.reset()
        total_rise = 0.0
        for _ in range(decisions_per_round):
            greedy = Q[s].argmax(axis=1)
            explore = rng.random(n_swimmers) < eps
            a = np.where(explore, rng.integers(0, N_ACTIONS, n_swimmers), greedy)

            r, s2 = world.act(a)
            total_rise += r.mean()

            td = r - rho + Q[s2].max(axis=1) - Q[s, a]
            np.add.at(Q, (s, a), lr * td / np.maximum(
                np.bincount(s * N_ACTIONS + a, minlength=Q.size)
                .reshape(Q.shape)[s, a], 1))
            # update rho only from greedy transitions (standard differential Q)
            if (~explore).any():
                rho += lr_rho * (r[~explore].mean() - rho)
            s = s2

        g_eval = np.nan
        if eval_every and (rnd % eval_every == 0 or rnd == rounds - 1):
            g = evaluate(Q, n=500, decisions=200, seed=999, **env_kw)
            g_eval = g["mean_rise"] / (200 * world.decision_every * world.dt)
        history.append((rnd, total_rise / decisions_per_round, eps, g_eval))
        if verbose and (rnd % 50 == 0 or rnd == rounds - 1):
            msg = f"  round {rnd:4d}  eps={eps:.2f}  behaviour={history[-1][1]:+.4f}"
            if not np.isnan(g_eval):
                msg += f"  GREEDY rise/time = {g_eval:+.3f}"
            print(msg, flush=True)
    return Q, np.array(history)


def evaluate(policy, n=4000, decisions=600, seed=123, record_traj=0, **env_kw):
    """Run swimmers with a FIXED policy, no exploration, fresh seed.

    policy: either "naive" (always steer up) or a Q-table (greedy).
    Returns dict with rise statistics and (optionally) recorded trajectories.
    """
    rng = np.random.default_rng(seed)
    world = VortexWorld(n, rng=rng, **env_kw)
    s = world.reset()
    rises = np.zeros(n)
    t_axis, mean_curve = [], []
    traj = []      # (decisions, k, 3): x, y_unwrapped, th for first k swimmers

    for d in range(decisions):
        if isinstance(policy, str) and policy == "naive":
            a = np.zeros(n, dtype=int)                    # action 0 = steer up, always
        else:
            a = policy[s].argmax(axis=1)
        r, s = world.act(a)
        rises += r
        t_axis.append((d + 1) * world.decision_every * world.dt)
        mean_curve.append(rises.mean())
        if record_traj:
            traj.append(np.stack([world.x[:record_traj],
                                  world.y_unwrapped[:record_traj],
                                  world.th[:record_traj]], axis=1))

    out = {"mean_rise": float(rises.mean()), "std_rise": float(rises.std()),
           "median_rise": float(np.median(rises)),
           "frac_trapped": float((rises < 1.0).mean()),   # barely rose at all
           "t": np.array(t_axis), "curve": np.array(mean_curve)}
    if record_traj:
        out["traj"] = np.array(traj)                      # (T, k, 3)
    return out


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("training smart swimmers (tabular Q-learning, 12 states x 4 actions)...")
    Q, hist = train()
    np.save(out_dir / "Q.npy", Q)
    np.save(out_dir / "history.npy", hist)

    print("\nevaluating (fresh seed, no exploration)...")
    naive = evaluate("naive")
    smart = evaluate(Q)
    summary = {
        "naive": {k: naive[k] for k in ("mean_rise", "std_rise", "median_rise", "frac_trapped")},
        "smart": {k: smart[k] for k in ("mean_rise", "std_rise", "median_rise", "frac_trapped")},
        "ratio": smart["mean_rise"] / max(naive["mean_rise"], 1e-9),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
