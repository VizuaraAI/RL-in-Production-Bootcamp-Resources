"""The three-sphere swimmer — learning a GAIT from scratch at low Reynolds number.

THE PHYSICS (Najafi & Golestanian, PRE 69, 062901, 2004)
--------------------------------------------------------
At micro-scale, fluid has no inertia (Stokes flow). Purcell's SCALLOP THEOREM:
any time-reversible ("reciprocal") stroke produces ZERO net motion — a scallop
that just opens and closes goes nowhere, no matter how fast it flaps.

The simplest swimmer that CAN move: three spheres in a line, joined by two arms
that can each be Extended or Contracted. The magic sequence

    (E,E) -> (C,E) -> (C,C) -> (E,C) -> (E,E)   ...and repeat

is a LOOP in shape space (never retraces itself backwards), so it is
non-reciprocal — and it crawls a little each cycle. Reverse the loop and you
swim the other way.

We do the honest hydrodynamics, not a displacement lookup table:
  * each sphere i feels a force f_i along the line; sphere velocities follow
    from Stokes mobility with Oseen interactions:

        v_i = f_i / (6 pi mu a)  +  sum_{j != i}  f_j / (4 pi mu r_ij)

  * the arms impose relative velocities (an arm changes length at speed w),
    and the swimmer as a whole is force-free: f_1 + f_2 + f_3 = 0.
  * that is 3 linear equations for 3 unknown forces at every substep -> solve,
    move spheres, integrate. Net displacement per stroke EMERGES from physics.

THE RL PROBLEM (replicating Tsang, Tong, Nallan & Pak, PRFluids 5, 074101, 2020)
--------------------------------------------------------------------------------
  state   = which arms are extended: (arm1, arm2) in {C,E}^2      -> 4 states
  action  = toggle arm 1, or toggle arm 2                          -> 2 actions
  reward  = centre-of-mass displacement of the swimmer during the stroke
            (positive = to the right)

A Q-table of EIGHT numbers, and the agent must discover, purely from "did I
move", the one cyclic policy that beats the scallop theorem. Reciprocal
policies (toggle the same arm forever) are the failure mode — and the theorem
guarantees their reward is exactly zero, which makes this the cleanest
"the environment itself refutes bad policies" example in all of RL.
"""
from __future__ import annotations

import numpy as np

# geometry & fluid (nondimensional: mu = 1)
A = 0.10          # sphere radius
L_EXT = 1.0       # extended arm length
EPS = 0.40        # contraction distance  (contracted arm = L_EXT - EPS)
W_ARM = 0.20      # arm extension/contraction speed
DT = 0.01         # hydrodynamic substep
MU = 1.0

N_STATES = 4      # (arm1, arm2) in {0=Contracted, 1=Extended}^2 -> s = 2*arm1 + arm2
N_ACTIONS = 2     # 0: toggle arm 1,  1: toggle arm 2


def state_id(arm1: int, arm2: int) -> int:
    return 2 * arm1 + arm2


def _mobility_matrix(x):
    """Oseen mobility M (3x3) for collinear spheres at positions x: v = M f."""
    M = np.eye(3) / (6 * np.pi * MU * A)
    for i in range(3):
        for j in range(3):
            if i != j:
                M[i, j] = 1.0 / (4 * np.pi * MU * abs(x[i] - x[j]))
    return M


def stroke(x, arm: int, direction: int):
    """Change one arm's length by EPS at speed W_ARM, integrating the full
    force-free Oseen dynamics. Returns (new_x, cm_displacement, trajectory).

    arm: 0 (between spheres 0-1) or 1 (between spheres 1-2)
    direction: +1 extend, -1 contract
    """
    x = x.copy()
    n_sub = int(round(EPS / W_ARM / DT))
    cm0 = x.mean()
    traj = [x.copy()]
    # prescribed relative velocities of the two arms during this stroke
    dL = np.zeros(2)
    dL[arm] = direction * W_ARM
    for _ in range(n_sub):
        M = _mobility_matrix(x)
        # unknowns: f = (f0, f1, f2). Equations:
        #   (v1 - v0) = dL[0]   with v = M f     -> (M[1]-M[0]) f = dL[0]
        #   (v2 - v1) = dL[1]                    -> (M[2]-M[1]) f = dL[1]
        #   force-free:  f0 + f1 + f2 = 0
        Asys = np.vstack([M[1] - M[0], M[2] - M[1], np.ones(3)])
        b = np.array([dL[0], dL[1], 0.0])
        f = np.linalg.solve(Asys, b)
        v = M @ f
        x += v * DT
        traj.append(x.copy())
    return x, x.mean() - cm0, np.array(traj)


class ThreeSphereSwimmer:
    """Environment: hold the swimmer's shape state, apply toggle-actions."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.arms = [1, 1]                       # start (E, E)
        # sphere positions consistent with (E, E), centred at 0
        self.x = np.array([-L_EXT, 0.0, L_EXT])
        self.cm_history = [0.0]
        return state_id(*self.arms)

    def step(self, action: int):
        """Toggle one arm; integrate the hydrodynamics; reward = CM displacement."""
        direction = -1 if self.arms[action] == 1 else +1     # E->contract, C->extend
        self.x, d_cm, traj = stroke(self.x, action, direction)
        self.arms[action] ^= 1
        self.cm_history.append(self.cm_history[-1] + d_cm)
        return state_id(*self.arms), d_cm, traj


def train_gait(episodes=300, steps_per_ep=24, lr=0.15, gamma=0.9,
               eps_start=0.6, eps_end=0.02, seed=0, verbose=True):
    """Tabular Q-learning over 4 states x 2 actions = 8 numbers."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    env = ThreeSphereSwimmer()
    history = []                                  # (episode, displacement per episode)

    for ep in range(episodes):
        epsilon = eps_start + (eps_end - eps_start) * ep / max(episodes - 1, 1)
        s = env.reset()
        total = 0.0
        for _ in range(steps_per_ep):
            if rng.random() < epsilon:
                a = rng.integers(0, N_ACTIONS)
            else:
                a = int(Q[s].argmax())
            s2, r, _ = env.step(a)
            Q[s, a] += lr * (r + gamma * Q[s2].max() - Q[s, a])
            s = s2
            total += r
        history.append((ep, total, epsilon))
        if verbose and ep % 50 == 0:
            print(f"  ep {ep:4d}  eps={epsilon:.2f}  displacement/episode = {total:+.5f}",
                  flush=True)
    return Q, np.array(history)


def rollout(policy, steps=24, record=True):
    """Run a fixed policy. policy: Q-table (greedy), or 'reciprocal' (toggle arm 0
    forever — the scallop-theorem loser), or an explicit action list to cycle."""
    env = ThreeSphereSwimmer()
    s = env.reset()
    trajs, states_seq = [], [s]
    for i in range(steps):
        if isinstance(policy, str) and policy == "reciprocal":
            a = 0
        elif isinstance(policy, (list, tuple)):
            a = policy[i % len(policy)]
        else:
            a = int(policy[s].argmax())
        s, _, traj = env.step(a)
        trajs.append(traj)
        states_seq.append(s)
    return {"cm": np.array(env.cm_history), "trajs": trajs, "states": states_seq,
            "x_final": env.x}


if __name__ == "__main__":
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)

    print("training the gait (tabular Q-learning, 4 states x 2 actions)...")
    Q, hist = train_gait()
    np.save(out / "gait_Q.npy", Q)
    np.save(out / "gait_history.npy", hist)

    print("\nevaluating gaits over 24 strokes...")
    learned = rollout(Q)
    recip = rollout("reciprocal")
    summary = {
        "learned_displacement": float(learned["cm"][-1]),
        "reciprocal_displacement": float(recip["cm"][-1]),
        "learned_states_cycle": learned["states"][:9],
        "per_stroke": float(learned["cm"][-1]) / 24,
    }
    (out / "gait_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
