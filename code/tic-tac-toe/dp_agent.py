"""Dynamic Programming (value iteration) for tic-tac-toe.

The point of this file is to show what DP actually does, on a problem small
enough to enumerate exhaustively.

We:
  1. Enumerate every reachable state.
  2. Initialize V(s) = 0 everywhere.
  3. Sweep all states, applying the Bellman backup:
         V(s) ← max_a  Σ_{s'}  P(s' | s, a) · [ r(s, a, s') + γ · V(s') ]
     Here, P(s' | s, a) is determined by the opponent's policy.
  4. Repeat until V stops changing by more than ε.

Requirements:
  - We need to KNOW the opponent's policy in advance (so we can compute the
    expectation over s'). Here we use a random opponent.

This is the cleanest example of the "needs a model" property of DP:
  the agent never plays a single game while learning. It just computes.
"""

from __future__ import annotations
import random
import time
from typing import Dict

import env as E   # the shared environment


GAMMA = 1.0   # episode terminates in ≤ 9 moves; no need to discount
EPS = 1e-6    # convergence threshold


def opponent_action_distribution(s: E.State, rng_seed: int = 0) -> Dict[E.Action, float]:
    """Probability distribution over the opponent's next move from state s.

    We model the opponent as uniform-random. Returns {action: probability}.
    """
    legals = E.legal_actions(s)
    p = 1.0 / len(legals)
    return {a: p for a in legals}


def expected_value_after_action(s: E.State, a: E.Action, V: Dict[E.State, float]) -> float:
    """Compute Q(s, a) = E[ r + γ V(s') ] where the randomness is the opponent's reply.

    Steps:
      1. Agent (X) plays action a.
      2. If that move ends the game, we collect the terminal reward; no opponent move.
      3. Otherwise the opponent (O) replies according to their (assumed) policy.
      4. We recurse: V(s'') from the resulting state.
    """
    s_after_x = E.step(s, a, E.X)
    if E.is_terminal(s_after_x):
        return E.reward(s_after_x, player=E.X)

    expected = 0.0
    for o_action, prob in opponent_action_distribution(s_after_x).items():
        s_after_o = E.step(s_after_x, o_action, E.O)
        if E.is_terminal(s_after_o):
            v_next = E.reward(s_after_o, player=E.X)
        else:
            v_next = V[s_after_o]
        expected += prob * (0.0 + GAMMA * v_next)
        # the immediate reward inside the bracket is 0 — rewards only fire at terminals
    return expected


def value_iteration() -> Dict[E.State, float]:
    """Run value iteration on every reachable non-terminal state where it is X's turn."""
    # Enumerate all reachable states.
    all_states = E.reachable_states()
    non_terminal = {s for s in all_states if not E.is_terminal(s)}

    # We only want X-to-move states (since the agent only chooses on its turn).
    def x_to_move(s: E.State) -> bool:
        return sum(1 for v in s if v == E.X) == sum(1 for v in s if v == E.O)

    x_states = {s for s in non_terminal if x_to_move(s)}

    V: Dict[E.State, float] = {s: 0.0 for s in x_states}
    sweeps = 0
    t0 = time.time()

    while True:
        sweeps += 1
        max_change = 0.0
        for s in x_states:
            old_v = V[s]
            new_v = max(expected_value_after_action(s, a, V) for a in E.legal_actions(s))
            V[s] = new_v
            max_change = max(max_change, abs(new_v - old_v))
        if max_change < EPS:
            break

    print(f"[DP] converged in {sweeps} sweeps · {time.time()-t0:.2f}s · |X-to-move states| = {len(V)}")
    return V


def greedy_policy(V: Dict[E.State, float]):
    """Return a function π(s) → best action under V."""
    def pi(s: E.State) -> E.Action:
        return max(E.legal_actions(s),
                   key=lambda a: expected_value_after_action(s, a, V))
    return pi


if __name__ == "__main__":
    V = value_iteration()
    pi = greedy_policy(V)
    # Show the agent's first move and its value
    s0 = E.initial_state()
    a0 = pi(s0)
    print(f"\n[DP] V(empty board) = {V[s0]:+.4f}")
    print(f"[DP] best first move = cell {a0} (center is 4)")
