"""Smart microswimmers in a vortex lattice — the environment.

The physical setup (Colabrese, Gustavsson, Celani & Mehlig, PRL 119, 244501, 2017):

  * FLOW — a 2D Taylor-Green vortex lattice, the "egg-crate" of counter-rotating
    vortices. It is analytic, so there is no CFD anywhere in this project:

        u_x =  U cos(x) sin(y)
        u_y = -U sin(x) cos(y)
        vorticity  w(x,y) = -2 U cos(x) cos(y)

    Periodic in both directions with period 2*pi.

  * SWIMMER — a gyrotactic micro-organism (think: bottom-heavy algae cell).
    It always swims at fixed speed v_s along its own orientation p = (sin th, cos th),
    and it is carried by the flow. Its orientation feels TWO torques:

      1. a restoring torque toward a preferred direction k_hat (bottom-heaviness),
         with reorientation timescale B;
      2. rotation by the local flow vorticity, at rate w/2.

        dx/dt  = u_x + v_s sin(th)
        dy/dt  = u_y + v_s cos(th)
        dth/dt = -sin(th - th_pref)/(2B) + w/2

    th is measured from the +y axis, so th = 0 means "pointing straight up".

  * THE CONFLICT — the swimmer *wants* to go up (biologically: toward light).
    A naive swimmer sets th_pref = up, always. But wherever |w/2| beats the
    restoring rate 1/(2B), the vortex spins the swimmer faster than it can
    right itself: it tumbles, gets swept around the vortex, and stops rising.

  * THE RL TWIST — give the swimmer a brain. Every DECISION_EVERY time units it
    observes a tiny local state and picks th_pref from {up, left, down, right}.
    Reward = vertical distance risen since the last decision. That's it.
    The optimal behaviour it discovers: steer *sideways* at the right moments to
    hop out of a trapping vortex and ride the upwelling sheets between vortices.

Everything is vectorised over N independent swimmers (numpy, no loops over agents).
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

# ---- action set: the four preferred directions the brain can pick -----------
# angle of k_hat measured from +y ("up"):  up, left, down, right
ACTION_ANGLES = np.array([0.0, -np.pi / 2, np.pi, np.pi / 2])
ACTION_NAMES = ["up", "left", "down", "right"]
N_ACTIONS = 4


class VortexWorld:
    """Taylor-Green lattice + a batch of N gyrotactic swimmers."""

    OBS_MODES = {"w3o4": 12, "w3uy2o4": 24, "uy2ux2o4": 16, "u33": 9}

    def __init__(self, n: int, U: float = 1.0, v_s: float = 0.3, B: float = 2.0,
                 dt: float = 0.01, decision_every: int = 10, obs_mode: str = "w3o4",
                 actuation: str = "torque", rng=None):
        # actuation="torque": gyrotactic — actions bias a WEAK righting torque (hard mode;
        #   at large B the vortex spin overwhelms it, and as our ablation shows, even the
        #   MDP-optimal policy gains <25% — a lesson about control authority, kept in the
        #   report). actuation="direct": a motile micro-robot that sets its heading
        #   instantly (Zermelo navigation, Biferale et al. 2019) but is SLOW: v_s < U, so
        #   the current still dominates its motion. All the difficulty moves into WHERE
        #   to point — which is exactly what RL is for.
        self.actuation = actuation
        self.n = n
        self.U, self.v_s, self.B = U, v_s, B
        self.dt = dt
        self.decision_every = decision_every       # integration substeps per decision
        self.obs_mode = obs_mode
        self.N_STATES = self.OBS_MODES[obs_mode]
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    # ---- the flow field (analytic — this is the whole "CFD") ---------------
    def flow(self, x, y):
        ux = self.U * np.cos(x) * np.sin(y)
        uy = -self.U * np.sin(x) * np.cos(y)
        return ux, uy

    def vorticity(self, x, y):
        return -2.0 * self.U * np.cos(x) * np.cos(y)

    # ---- state: what the swimmer is allowed to sense -----------------------
    # 3 vorticity bins  x  4 orientation quadrants  = 12 discrete states.
    # This tiny observation is deliberate: a micro-organism cannot see the whole
    # flow; it can plausibly sense "am I being spun, and which way am I facing".
    def observe(self):
        """Observation ablations — WHAT the swimmer can sense is a design choice,
        and (as the lecture shows) it matters more than the learning algorithm:

          w3o4    — local vorticity (3 bins) x own heading (4)          = 12 states
          w3uy2o4 — the above x "am I in an up-draft?" (sign of u_y)    = 24 states
          uy2ux2o4— local flow direction signs (u_y, u_x) x heading     = 16 states
        """
        th = np.mod(self.th + np.pi / 4, TWO_PI)           # quadrants centred on U/R/D/L
        o_bin = (th / (np.pi / 2)).astype(int) % 4         # 0 up, 1 right, 2 down, 3 left
        if self.obs_mode == "w3o4":
            w = self.vorticity(self.x, self.y)
            w_bin = np.digitize(w, [-self.U, self.U])      # strong CW / weak / strong CCW
            return w_bin * 4 + o_bin
        ux, uy = self.flow(self.x, self.y)
        if self.obs_mode == "w3uy2o4":
            w = self.vorticity(self.x, self.y)
            w_bin = np.digitize(w, [-self.U, self.U])
            return (w_bin * 2 + (uy > 0)) * 4 + o_bin
        if self.obs_mode == "uy2ux2o4":
            return ((uy > 0) * 2 + (ux > 0)) * 4 + o_bin
        if self.obs_mode == "u33":
            # local flow velocity, each component in 3 bins — 9 states, no heading
            uxb = np.digitize(ux, [-self.U / 2, self.U / 2])
            uyb = np.digitize(uy, [-self.U / 2, self.U / 2])
            return uyb * 3 + uxb
        raise ValueError(self.obs_mode)

    # ---- dynamics ----------------------------------------------------------
    def step_physics(self, th_pref):
        """Advance all swimmers by `decision_every` Euler substeps under a fixed
        preferred direction. Returns the vertical rise achieved (the reward)."""
        y0 = self.y_unwrapped.copy()
        if self.actuation == "direct":
            self.th = th_pref if np.ndim(th_pref) else np.full(self.n, th_pref)
        for _ in range(self.decision_every):
            ux, uy = self.flow(self.x, self.y)
            w = self.vorticity(self.x, self.y)
            self.x = np.mod(self.x + (ux + self.v_s * np.sin(self.th)) * self.dt, TWO_PI)
            dy = (uy + self.v_s * np.cos(self.th)) * self.dt
            self.y = np.mod(self.y + dy, TWO_PI)
            self.y_unwrapped += dy                          # true rise, not wrapped
            if self.actuation == "direct":
                continue                                    # heading fixed for the decision
            # NOTE the minus sign on the vorticity term: theta is measured from +y,
            # so INCREASING theta turns the swimmer clockwise — while positive vorticity
            # rotates fluid counter-clockwise. First version had +w/2: a self-consistent
            # mirror world (learning still works), but unfaithful physics. Fixed.
            dth = (-np.sin(self.th - th_pref) / (2 * self.B) - 0.5 * w) * self.dt
            self.th = np.mod(self.th + dth, TWO_PI)
        return self.y_unwrapped - y0

    def act(self, actions):
        """actions: int array (N,) in [0,4). Returns (reward, next_state)."""
        rise = self.step_physics(ACTION_ANGLES[actions])
        return rise, self.observe()

    def reset(self):
        self.x = self.rng.uniform(0, TWO_PI, self.n)
        self.y = self.rng.uniform(0, TWO_PI, self.n)
        self.th = self.rng.uniform(0, TWO_PI, self.n)
        self.y_unwrapped = self.y.copy()
        return self.observe()
