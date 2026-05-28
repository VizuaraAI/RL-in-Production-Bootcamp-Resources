"""Tic-Tac-Toe environment used by the DP / MC / TD agents.

State: tuple of 9 cells, each in {0, 1, -1}
  0  → empty
  1  → X (the learning agent)
 -1  → O (the opponent)

The board is indexed row-major:
    0 | 1 | 2
   ---+---+---
    3 | 4 | 5
   ---+---+---
    6 | 7 | 8

This file is deliberately tiny and pure — no NumPy, no Gym wrapper —
so it can be imported into any of the three agent scripts without ceremony.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import random

State = Tuple[int, ...]   # length-9 tuple
Action = int              # 0..8

EMPTY = 0
X = 1
O = -1

# All winning triples — rows, columns, diagonals.
LINES: List[Tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def initial_state() -> State:
    return (EMPTY,) * 9


def legal_actions(s: State) -> List[Action]:
    return [i for i, v in enumerate(s) if v == EMPTY]


def step(s: State, a: Action, player: int) -> State:
    """Place `player` (1 or -1) at cell `a`. Returns new state."""
    assert s[a] == EMPTY, f"illegal move at cell {a}"
    new = list(s)
    new[a] = player
    return tuple(new)


def winner(s: State) -> Optional[int]:
    """Return +1, -1, or None if no one has won yet."""
    for i, j, k in LINES:
        if s[i] != EMPTY and s[i] == s[j] == s[k]:
            return s[i]
    return None


def is_terminal(s: State) -> bool:
    return winner(s) is not None or all(c != EMPTY for c in s)


def reward(s: State, player: int = X) -> float:
    """Reward FROM THE PERSPECTIVE OF `player`. Only nonzero at terminal states."""
    w = winner(s)
    if w == player:
        return 1.0
    if w == -player:
        return -1.0
    return 0.0


def render(s: State) -> str:
    """Pretty-print the board for debugging."""
    chars = {EMPTY: ".", X: "X", O: "O"}
    rows = [" ".join(chars[s[3 * r + c]] for c in range(3)) for r in range(3)]
    return "\n".join(rows)


# ── Opponent policies ──────────────────────────────────────────────────

def random_opponent(s: State, rng: random.Random) -> Action:
    """The opponent picks a uniformly random legal move."""
    return rng.choice(legal_actions(s))


def can_win_now(s: State, player: int) -> Optional[Action]:
    """If `player` has a one-move win, return that move. Else None."""
    for a in legal_actions(s):
        if winner(step(s, a, player)) == player:
            return a
    return None


def greedy_opponent(s: State, rng: random.Random) -> Action:
    """A modestly strong opponent:
       1) take a winning move if one exists,
       2) block the agent's winning move if one exists,
       3) otherwise random.
    """
    me = O
    them = X
    if (a := can_win_now(s, me)) is not None:
        return a
    if (a := can_win_now(s, them)) is not None:
        return a
    return rng.choice(legal_actions(s))


# ── State space enumeration (used by DP) ───────────────────────────────

def reachable_states(opponent_first: bool = False) -> set[State]:
    """Enumerate every reachable (non-terminal) state from the empty board.

    We assume X (the agent) and O alternate moves. Terminal states are excluded
    because we never need V(s) for terminal s — V(terminal) is just the reward.

    Returns: set of states where it is the AGENT's (X's) turn to move.
    """
    seen: set[State] = set()
    frontier: List[Tuple[State, int]] = [(initial_state(), X if not opponent_first else O)]
    all_states: set[State] = set()
    while frontier:
        s, turn = frontier.pop()
        if s in seen:
            continue
        seen.add(s)
        all_states.add(s)
        if is_terminal(s):
            continue
        for a in legal_actions(s):
            frontier.append((step(s, a, turn), -turn))
    return all_states


if __name__ == "__main__":
    print(f"Initial state:\n{render(initial_state())}\n")
    print(f"Total reachable states (X to move OR O to move): {len(reachable_states())}")
