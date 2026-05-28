"""Temporal Difference (TD(0)) value learning for tic-tac-toe.

The point of this file is to contrast with both DP and MC.

  - Like MC: no model needed. We just play.
  - Unlike MC: we don't wait until the episode ends to update. After every
    move, we update V(s) using V(s'), the value of the state we just moved to.

The TD(0) update rule, in one line:

      V(s) ← V(s) + α · [ r + γ · V(s') − V(s) ]

That bracket is the "TD error" δ. We're nudging our estimate of V(s) toward
the bootstrapped target  r + γ·V(s').

Crucially: r + γ·V(s') is an ESTIMATE. We're updating an estimate using
another estimate. This is *bootstrapping*. It's lower-variance than MC
(one transition vs whole episode of noise) but it's biased while V(s')
is still learning.
"""

from __future__ import annotations
import random
import time
from typing import Dict

import env as E


GAMMA = 1.0
ALPHA = 0.10        # step size
EPSILON = 0.10      # ε-greedy exploration
NUM_EPISODES = 10_000
SEED = 42


def value_of(s: E.State, V: Dict[E.State, float]) -> float:
    return V.get(s, 0.0)


def greedy_action(s: E.State, V: Dict[E.State, float]) -> E.Action:
    best_a, best_v = None, -float("inf")
    for a in E.legal_actions(s):
        s_after_x = E.step(s, a, E.X)
        v = E.reward(s_after_x, E.X) if E.is_terminal(s_after_x) else value_of(s_after_x, V)
        if v > best_v:
            best_v, best_a = v, a
    return best_a


def epsilon_greedy(s: E.State, V: Dict[E.State, float], rng: random.Random) -> E.Action:
    if rng.random() < EPSILON:
        return rng.choice(E.legal_actions(s))
    return greedy_action(s, V)


def td_zero() -> Dict[E.State, float]:
    """One TD(0) update per agent move. No replay, no targets, no tricks."""
    V: Dict[E.State, float] = {}
    rng = random.Random(SEED)
    t0 = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        s = E.initial_state()
        while True:
            # X moves
            a = epsilon_greedy(s, V, rng)
            s_after_x = E.step(s, a, E.X)

            if E.is_terminal(s_after_x):
                # Terminal transition: target is the terminal reward, no bootstrap
                r = E.reward(s_after_x, E.X)
                v_s = V.get(s, 0.0)
                V[s] = v_s + ALPHA * (r - v_s)
                break

            # O replies uniformly at random
            o = rng.choice(E.legal_actions(s_after_x))
            s_next = E.step(s_after_x, o, E.O)

            if E.is_terminal(s_next):
                r = E.reward(s_next, E.X)
                v_s = V.get(s, 0.0)
                V[s] = v_s + ALPHA * (r - v_s)
                break

            # Non-terminal: bootstrap from V(s_next)
            r = 0.0   # rewards only at terminals
            v_s = V.get(s, 0.0)
            v_next = V.get(s_next, 0.0)
            td_error = r + GAMMA * v_next - v_s
            V[s] = v_s + ALPHA * td_error
            s = s_next

        if episode % 1_000 == 0:
            print(f"[TD] episode {episode:>5}  ·  states_seen={len(V):>4}  ·  elapsed={time.time()-t0:.1f}s")

    print(f"[TD] done · {NUM_EPISODES} episodes · {time.time()-t0:.1f}s · final |V| = {len(V)}")
    return V


if __name__ == "__main__":
    V = td_zero()
    s0 = E.initial_state()
    print(f"\n[TD] V(empty board) = {V.get(s0, 0.0):+.4f}")
    print(f"[TD] V(center-X)     = {V.get((0,0,0,0,1,0,0,0,0), 0.0):+.4f}")
