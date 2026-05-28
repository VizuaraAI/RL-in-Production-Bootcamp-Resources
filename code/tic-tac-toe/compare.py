"""Side-by-side comparison: DP vs MC vs TD on tic-tac-toe.

Runs all three. Reports:
  - convergence wall-clock
  - V(empty board) — should be ≈ 0.91 against a random opponent
  - sample policy on a few diagnostic positions

Run with:  python compare.py
"""
from __future__ import annotations
import time

import env as E
import dp_agent
import mc_agent
import td_agent


def evaluate_v_at_diagnostics(V, label: str):
    """Print V at a few hand-picked, illustrative positions."""
    print(f"\n[{label}] diagnostic V-values:")
    s0 = E.initial_state()
    s_center = (0, 0, 0, 0, 1, 0, 0, 0, 0)  # X in center
    s_corner = (1, 0, 0, 0, 0, 0, 0, 0, 0)  # X in corner
    s_about_to_win = (1, 1, 0, 0, -1, 0, 0, 0, -1)  # X to play, can win at cell 2
    for name, s in [
        ("empty board (start)", s0),
        ("X center (after first move)", s_center),
        ("X corner (after first move)", s_corner),
        ("X about to win at cell 2", s_about_to_win),
    ]:
        v = V.get(s, None)
        print(f"  {name:<35s}  V = {v:+.4f}" if v is not None else f"  {name:<35s}  V = (unseen)")


def main():
    print("=" * 70)
    print("DP vs MC vs TD on tic-tac-toe — all three learn V against a random opponent")
    print("=" * 70)

    # ── DP ──
    print("\n──── 1. Dynamic Programming (value iteration) ────")
    t0 = time.time()
    V_dp = dp_agent.value_iteration()
    dp_time = time.time() - t0
    evaluate_v_at_diagnostics(V_dp, "DP")

    # ── MC ──
    print("\n──── 2. Monte Carlo (first-visit) ────")
    t0 = time.time()
    V_mc = mc_agent.monte_carlo()
    mc_time = time.time() - t0
    evaluate_v_at_diagnostics(V_mc, "MC")

    # ── TD ──
    print("\n──── 3. Temporal Difference (TD(0)) ────")
    t0 = time.time()
    V_td = td_agent.td_zero()
    td_time = time.time() - t0
    evaluate_v_at_diagnostics(V_td, "TD")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<8} {'Wall clock':>12} {'Episodes played':>18} {'|V|':>6}")
    print("-" * 70)
    print(f"{'DP':<8} {dp_time:>10.2f}s {'0 (computes only)':>18} {len(V_dp):>6}")
    print(f"{'MC':<8} {mc_time:>10.2f}s {mc_agent.NUM_EPISODES:>18,} {len(V_mc):>6}")
    print(f"{'TD':<8} {td_time:>10.2f}s {td_agent.NUM_EPISODES:>18,} {len(V_td):>6}")


if __name__ == "__main__":
    main()
