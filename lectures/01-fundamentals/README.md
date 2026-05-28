# Lecture 01 — Fundamentals

**Duration:** 2 hours
**Date:** Cohort 2026 — opening lecture

## What this lecture covers

| Part | Topic | Time |
|---|---|---|
| 1 | A brief history of RL — Bellman → Sutton → DQN → AlphaGo → PPO → RLHF → GRPO → VLA | 22 min |
| 2 | Markov Decision Processes — the (S, A, P, R, γ) tuple, the Markov property, discounting | 15 min |
| 3 | Value functions — V(s) and Q(s, a), the return, optimal vs. policy-specific | 18 min |
| 4 | The Bellman backup — derivation, value iteration, AlphaGo as the worked example, bootstrapping | 25 min |
| 5 | DP, Monte Carlo, and TD on tic-tac-toe — three recipes for the same Bellman recursion | 15 min |

## The single sentence

> **The value of where I am is the reward I just got, plus a discounted value of where I'll be one step from now.**

Every algorithm in the rest of the bootcamp — Q-learning, DQN, PPO, RLHF, GRPO, VLA — is a variation on that one sentence.

## Hands-on code

Run the comparison from the codebase:

```bash
cd code/tic-tac-toe
python compare.py
```

You should see all three algorithms (DP, MC, TD) converge to **V(empty board) ≈ +0.99** against a random opponent — three different paths to the same answer.

Read [`code/tic-tac-toe/README.md`](../../code/tic-tac-toe/README.md) for the full walkthrough.

## Suggested reading

Pick the format that suits you best.

### Textbook

- **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed)**
  - Ch. 3 — Finite Markov Decision Processes
  - Ch. 4 — Dynamic Programming
  - Ch. 5 — Monte Carlo Methods
  - Ch. 6 — Temporal-Difference Learning

### Papers (in chronological order, matching the §2 timeline)

- Bellman, *Dynamic Programming*, 1957 — read the preface and §1.
- Sutton, *Learning to predict by the methods of temporal differences*, 1988.
- Tesauro, *Temporal Difference Learning and TD-Gammon*, 1995.
- Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013 (DQN).
- Silver et al., *Mastering the game of Go with deep neural networks and tree search*, 2016 (AlphaGo).
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 (PPO).
- Christiano et al., *Deep Reinforcement Learning from Human Preferences*, 2017 (the RLHF foundation).
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2024 (GRPO).

### Talks

- David Silver, *DeepMind RL Lectures* — Lectures 1–4 line up almost exactly with today's material.
- Andrej Karpathy, *Deep Reinforcement Learning: Pong from Pixels* (blog post).

## Before Lecture 02

The next lecture is on Q-learning and DQN. To prepare:

1. Clone this repo and run `python code/tic-tac-toe/compare.py`. Confirm the three algorithms agree.
2. Modify `env.py` to use `greedy_opponent` instead of `random_opponent`. Re-run the comparison. Notice how V(empty board) drops — the agent can't dominate a strong opponent the way it dominates a random one.
3. (Optional) Try swapping the dictionary V in `td_agent.py` for a small PyTorch MLP. That's the leap we make in Lecture 02.

## Errata, questions, discussion

Open an issue on this repo, or email `hello@vizuara.com`.
