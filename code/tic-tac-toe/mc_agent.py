"""Monte Carlo (first-visit) value learning for tic-tac-toe.

The point of this file is to contrast with DP. We do NOT know the opponent's
policy. We do NOT enumerate states. We just PLAY games — many of them — and
average the returns we observe.

The loop is:
  for episode in 1..N:
      play one full game (agent uses ε-greedy w.r.t. current V)
      record every (state, return-from-that-state) pair
      for each FIRST visit to a state s in this episode:
          N(s) += 1
          V(s) += (G - V(s)) / N(s)        # incremental average

Notes:
  - V values are stored per-state in a dict. Unseen states default to 0.
  - The agent uses an opponent that picks uniformly random legal moves
    (same as the DP file), so the two value functions are directly comparable.
  - "First-visit" MC means: if a state appears multiple times in an episode,
    only the FIRST occurrence contributes to its update. This avoids bias
    in the return estimate.
"""

from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple

import env as E


GAMMA = 1.0
ALPHA = None        # using running average instead — see below
EPSILON = 0.10      # ε-greedy exploration
NUM_EPISODES = 100_000
SEED = 42


def value_of(s: E.State, V: Dict[E.State, float]) -> float:
    return V.get(s, 0.0)


def greedy_action(s: E.State, V: Dict[E.State, float]) -> E.Action:
    """Pick the action whose RESULTING X-to-move state has the highest V.

    Since rewards only fire at terminals, V(s'') alone determines preference.
    For MC, V(terminal) is just the reward we'd collect.
    """
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


def play_one_episode(V: Dict[E.State, float], rng: random.Random) -> List[Tuple[E.State, float]]:
    """Play one game. Return list of (state, return-from-that-state) for X-to-move states."""
    states_visited: List[E.State] = []
    s = E.initial_state()

    while True:
        # X (the agent) moves
        states_visited.append(s)
        a = epsilon_greedy(s, V, rng)
        s = E.step(s, a, E.X)
        if E.is_terminal(s):
            break
        # O (the opponent) moves uniformly at random
        o = rng.choice(E.legal_actions(s))
        s = E.step(s, o, E.O)
        if E.is_terminal(s):
            break

    G_final = E.reward(s, player=E.X)   # +1, -1, or 0
    # Since reward only fires at terminal and γ=1, return from EVERY visited
    # X-to-move state in this episode equals G_final.
    return [(state, G_final) for state in states_visited]


def monte_carlo() -> Dict[E.State, float]:
    V: Dict[E.State, float] = {}
    counts: Dict[E.State, int] = {}
    rng = random.Random(SEED)
    t0 = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        trajectory = play_one_episode(V, rng)

        # First-visit MC: track which states we've already updated in this episode.
        already_seen: set[E.State] = set()
        for state, G in trajectory:
            if state in already_seen:
                continue
            already_seen.add(state)
            counts[state] = counts.get(state, 0) + 1
            n = counts[state]
            v_old = V.get(state, 0.0)
            V[state] = v_old + (G - v_old) / n   # running mean

        if episode % 10_000 == 0:
            print(f"[MC] episode {episode:>6}  ·  states_seen={len(V):>4}  ·  elapsed={time.time()-t0:.1f}s")

    print(f"[MC] done · {NUM_EPISODES} episodes · {time.time()-t0:.1f}s · final |V| = {len(V)}")
    return V


if __name__ == "__main__":
    V = monte_carlo()
    s0 = E.initial_state()
    print(f"\n[MC] V(empty board) = {V.get(s0, 0.0):+.4f}")
    print(f"[MC] V(center-X)     = {V.get((0,0,0,0,1,0,0,0,0), 0.0):+.4f}")
