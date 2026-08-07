"""Problem A (final form): Zermelo navigation — cross the current to a target.

THE TASK (after Biferale, Bonaccorso, Buzzicotti, Clark Di Leoni & Gustavsson,
"Zermelo's problem: optimal point-to-point navigation in 2D turbulent flows
using reinforcement learning", Chaos 29, 103138, 2019):

A micro-robot swims at fixed speed v_s = 0.3 through the Taylor-Green vortex
lattice whose currents reach U = 1.0 — more than THREE TIMES its own speed.
It must reach a target disc on the far side of the lattice.

  * NAIVE policy: always point straight at the target. In still water this is
    optimal. In a strong flow it is a disaster: wherever the opposing current
    exceeds v_s, the robot physically cannot make headway; it gets deflected,
    swept around vortices, and often never arrives at all.
  * SMART policy: tabular Q-learning over a coarse position grid. It discovers
    DETOURS — riding a vortex's favourable side like a roundabout, crossing
    between cells where the jets help instead of hurt.

Students will recognise this instantly: it is the WINDY GRIDWORLD from Sutton &
Barto / bootcamp Phase 1 — except the wind is a real incompressible flow field.

  state   : which cell of a GRID x GRID partition the robot is in   (position only)
  actions : 8 compass headings
  reward  : -1 per decision (time penalty), +200 on reaching the target disc
  episode : ends at target or after `max_steps` decisions

Everything vectorised over N parallel robots sharing one Q-table.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
N_ACT = 8
HEADINGS = np.arange(N_ACT) * (TWO_PI / N_ACT)          # angle from +y axis, CW
HEAD_VEC = np.stack([np.sin(HEADINGS), np.cos(HEADINGS)], axis=1)   # (8, 2) unit vectors


class NavWorld:
    """Vectorised navigation episodes in the Taylor-Green lattice."""

    def __init__(self, n: int, U: float = 1.0, v_s: float = 0.3,
                 start=(1.2, 1.2), target=(5.2, 5.2), target_r: float = 0.45,
                 grid: int = 12, dt: float = 0.01, decision_every: int = 20,
                 max_steps: int = 400, jitter: float = 0.25, rng=None):
        self.n = n
        self.U, self.v_s = U, v_s
        self.start = np.array(start, dtype=float)
        self.target = np.array(target, dtype=float)
        self.target_r = target_r
        self.grid = grid
        self.dt, self.decision_every, self.max_steps = dt, decision_every, max_steps
        self.jitter = jitter                              # start-position spread
        self.rng = rng or np.random.default_rng(0)
        self.N_STATES = grid * grid
        self.reset()

    def flow(self, x, y):
        return (self.U * np.cos(x) * np.sin(y),
                -self.U * np.sin(x) * np.cos(y))

    # ---- state: coarse position (a robot with a position fix) --------------
    def observe(self):
        gx = np.clip((self.x / TWO_PI * self.grid).astype(int), 0, self.grid - 1)
        gy = np.clip((self.y / TWO_PI * self.grid).astype(int), 0, self.grid - 1)
        return gy * self.grid + gx

    def reset(self):
        self.x = np.mod(self.start[0] + self.rng.uniform(-self.jitter, self.jitter, self.n),
                        TWO_PI)
        self.y = np.mod(self.start[1] + self.rng.uniform(-self.jitter, self.jitter, self.n),
                        TWO_PI)
        self.t_steps = np.zeros(self.n, dtype=int)
        self.done = np.zeros(self.n, dtype=bool)
        self.arrived = np.zeros(self.n, dtype=bool)
        return self.observe()

    def _dist_to_target(self):
        # periodic-aware distance to the target
        dx = np.abs(self.x - self.target[0]); dx = np.minimum(dx, TWO_PI - dx)
        dy = np.abs(self.y - self.target[1]); dy = np.minimum(dy, TWO_PI - dy)
        return np.hypot(dx, dy)

    def bearing_action(self):
        """The NAIVE policy: the compass heading pointing most directly at the
        target (periodic-aware). A genuine member of the same action space."""
        dx = self.target[0] - self.x
        dx = np.where(np.abs(dx) > np.pi, dx - np.sign(dx) * TWO_PI, dx)
        dy = self.target[1] - self.y
        dy = np.where(np.abs(dy) > np.pi, dy - np.sign(dy) * TWO_PI, dy)
        ang = np.mod(np.arctan2(dx, dy), TWO_PI)          # angle from +y, CW
        return (np.round(ang / (TWO_PI / N_ACT)).astype(int)) % N_ACT

    def act(self, actions):
        """Returns (reward, next_state, done). Finished robots stay frozen."""
        v = HEAD_VEC[actions]
        active = ~self.done
        for _ in range(self.decision_every):
            ux, uy = self.flow(self.x, self.y)
            self.x = np.where(active,
                              np.mod(self.x + (ux + self.v_s * v[:, 0]) * self.dt, TWO_PI),
                              self.x)
            self.y = np.where(active,
                              np.mod(self.y + (uy + self.v_s * v[:, 1]) * self.dt, TWO_PI),
                              self.y)
        self.t_steps += active
        hit = active & (self._dist_to_target() < self.target_r)
        timeout = active & (self.t_steps >= self.max_steps)
        reward = np.where(hit, 200.0, -1.0) * active
        self.arrived |= hit
        self.done |= hit | timeout
        return reward, self.observe(), self.done


def train_nav(n=512, generations=300, lr=0.2, gamma=0.995,
              eps_start=0.4, eps_end=0.01, seed=0, verbose=True, **env_kw):
    """Episodic tabular Q-learning, N parallel robots per generation."""
    rng = np.random.default_rng(seed)
    world = NavWorld(n, rng=rng, **env_kw)
    Q = np.zeros((world.N_STATES, N_ACT))
    history = []                                          # (gen, arrival%, mean_time)

    for g in range(generations):
        eps = eps_start + (eps_end - eps_start) * g / max(generations - 1, 1)
        s = world.reset()
        while not world.done.all():
            greedy = Q[s].argmax(axis=1)
            explore = rng.random(n) < eps
            a = np.where(explore, rng.integers(0, N_ACT, n), greedy)
            r, s2, done = world.act(a)
            active = r != 0                                # frozen robots emit r == 0
            target_v = r + gamma * Q[s2].max(axis=1) * (~done)
            td = (target_v - Q[s, a]) * active
            np.add.at(Q, (s, a), lr * td / np.maximum(
                np.bincount(s * N_ACT + a, weights=active.astype(float),
                            minlength=Q.size).reshape(Q.shape)[s, a], 1))
            s = s2
        arr = world.arrived.mean()
        mt = world.t_steps[world.arrived].mean() if world.arrived.any() else np.nan
        history.append((g, arr, mt, eps))
        if verbose and (g % 25 == 0 or g == generations - 1):
            print(f"  gen {g:4d}  eps={eps:.2f}  arrival={100*arr:5.1f}%  "
                  f"mean time={mt:6.1f}", flush=True)
    return Q, np.array(history)


def evaluate_nav(policy, n=4000, seed=321, record_traj=0, **env_kw):
    """policy: 'naive' (aim at target) or a Q-table (greedy)."""
    rng = np.random.default_rng(seed)
    world = NavWorld(n, rng=rng, **env_kw)
    s = world.reset()
    traj = []
    while not world.done.all():
        if isinstance(policy, str) and policy == "naive":
            a = world.bearing_action()
        else:
            a = policy[s].argmax(axis=1)
        _, s, _ = world.act(a)
        if record_traj:
            traj.append(np.stack([world.x[:record_traj], world.y[:record_traj],
                                  world.done[:record_traj]], axis=1))
    out = {"arrival_rate": float(world.arrived.mean()),
           "mean_time": float(world.t_steps[world.arrived].mean())
                        if world.arrived.any() else float("nan"),
           "median_time": float(np.median(world.t_steps[world.arrived]))
                          if world.arrived.any() else float("nan")}
    if record_traj:
        out["traj"] = np.array(traj)
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)

    print("Zermelo navigation: Q-learning on the windy-gridworld-with-real-wind")
    Q, hist = train_nav()
    np.save(out / "nav_Q.npy", Q)
    np.save(out / "nav_history.npy", hist)

    print("\nevaluating naive vs smart (fresh seed)...")
    res = {}
    for name, pol in [("naive", "naive"), ("smart", Q)]:
        r = evaluate_nav(pol)
        res[name] = {k: r[k] for k in ("arrival_rate", "mean_time", "median_time")}
        print(f"  {name:6s}: arrive {100*r['arrival_rate']:5.1f}%  "
              f"mean time {r['mean_time']:6.1f}")
    (out / "nav_summary.json").write_text(json.dumps(res, indent=2))
