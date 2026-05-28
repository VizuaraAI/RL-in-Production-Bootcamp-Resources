# Tic-Tac-Toe — DP vs MC vs TD

Three implementations of the same problem — *learning the value function for tic-tac-toe against a random opponent* — using three different families of RL algorithm. Use this folder as a clean reference for **how the three families actually differ in code**.

## Files

| File | What it does |
|---|---|
| [`env.py`](./env.py) | Pure-Python tic-tac-toe environment. State, actions, rewards, terminal check, opponent policies, state-space enumeration. **Imported by all three agents.** |
| [`dp_agent.py`](./dp_agent.py) | Value iteration. Enumerates every reachable state, sweeps the Bellman backup until convergence. Uses the *known* opponent model. **No games played.** |
| [`mc_agent.py`](./mc_agent.py) | First-visit Monte Carlo. Plays self-play episodes against a random opponent. Updates each visited state toward the realized return at episode end. |
| [`td_agent.py`](./td_agent.py) | TD(0). Plays self-play episodes. Updates V(s) after every single move using `V(s) ← V(s) + α[r + γ·V(s') − V(s)]`. |
| [`compare.py`](./compare.py) | Runs all three back-to-back. Reports wall-clock, diagnostic V values, and a summary table. |

## How to run

No dependencies beyond the Python standard library.

```bash
python compare.py
```

Expected output (approximate):

```
DP    wall:  ~3s   episodes played:    0   |V| =  ~2400
MC    wall: ~20s   episodes played: 100,000   |V| =  ~2400
TD    wall:  ~2s   episodes played:  10,000   |V| =  ~2400
```

Against a uniformly-random opponent, the optimal X-strategy wins about **91%** of games. All three algorithms should converge to V(empty board) ≈ +0.91.

## What you should take away

| Property | DP | MC | TD |
|---|---|---|---|
| Needs a model `P(s'|s,a)` | **yes** | no | no |
| Plays games while learning | no | yes | yes |
| Updates use a **sampled** outcome | no — exact expectation | yes — full-episode return | yes — one transition |
| Bootstraps from `V(s')` | yes | **no** | yes |
| When does the update fire? | every sweep, every state | end of episode only | after every move |
| Variance | none (it's deterministic) | high (whole-episode return) | moderate (one-step) |
| Bias | none at convergence | none at convergence | nonzero while V(s') is still learning |

The lecture's "throughline" claim is exactly visible here: **all three algorithms compute the same V** at convergence. They differ only in *what data they use* and *when they update*. Every modern algorithm — Q-learning, DQN, PPO, GRPO — is some variation on the TD update rule scaled up to a problem too big to enumerate.

## Going further

- Switch `random_opponent` in `env.py` to `greedy_opponent` and re-run. Watch V(empty) drop. The agent's optimal achievable value is opponent-dependent.
- In `td_agent.py`, vary `ALPHA` (step size) and `EPSILON` (exploration). See how convergence speed and quality trade off.
- Replace the dictionary V with a small neural network. That's the leap from "tabular RL" (this file) to "function-approximation RL" (DQN, the next lecture).
