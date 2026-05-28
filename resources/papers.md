# Reading list — papers, by lecture

A curated path through the literature. Each entry: link → one-sentence summary → which lecture(s) it supports.

## Lecture 01 — Fundamentals

| Year | Paper | Why read it |
|---|---|---|
| 1957 | Bellman, *Dynamic Programming* | The original statement of the principle of optimality. Read the preface and §1 — the rest is operations research. |
| 1988 | Sutton, *Learning to predict by the methods of temporal differences* | The paper that introduced TD learning. Short, beautifully written. |
| 1995 | Tesauro, *Temporal Difference Learning and TD-Gammon* ([PDF](https://www.bkgm.com/articles/tesauro/tdl.html)) | The first time RL beat top humans at a hard game. Foreshadows AlphaGo by 20 years. |

## Lecture 02 — Q-learning and DQN

| Year | Paper | Why read it |
|---|---|---|
| 1989 | Watkins, *Learning from Delayed Rewards* (PhD thesis) | The origin of Q-learning. |
| 2013 | Mnih et al., *Playing Atari with Deep Reinforcement Learning* | DQN. The replay buffer + target network trick that made deep RL stable. |
| 2015 | Mnih et al., *Human-level control through deep reinforcement learning* (Nature DQN) | Polished publication of the above; reads more cleanly. |
| 2015 | van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* | Fixes DQN's overestimation bias. The first of many incremental improvements. |
| 2015 | Schaul et al., *Prioritized Experience Replay* | Better sampling from the replay buffer. |

## Lecture 03 — Policy gradients (REINFORCE → TRPO → PPO)

| Year | Paper | Why read it |
|---|---|---|
| 1992 | Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning* | REINFORCE. The clean derivation of the policy gradient. |
| 2015 | Schulman et al., *Trust Region Policy Optimization* (TRPO) | The first stable policy-gradient method. |
| 2017 | Schulman et al., *Proximal Policy Optimization Algorithms* (PPO) | TRPO's practical successor. Most-used policy-gradient algorithm in the world. |
| 2016 | Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE) | The advantage estimator that powers PPO. |

## Lecture 04 — Actor-critic and continuous control

| Year | Paper | Why read it |
|---|---|---|
| 2014 | Silver et al., *Deterministic Policy Gradient Algorithms* (DPG) | The DPG theorem. |
| 2015 | Lillicrap et al., *Continuous control with deep reinforcement learning* (DDPG) | Deep DPG. First widely-cited deep RL for continuous control. |
| 2018 | Haarnoja et al., *Soft Actor-Critic* (SAC) | The state-of-the-art on continuous control benchmarks for years. Entropy-regularized. |

## Lecture 05 — RLHF: language meets RL

| Year | Paper | Why read it |
|---|---|---|
| 2017 | Christiano et al., *Deep Reinforcement Learning from Human Preferences* | The original preference-based RL paper. |
| 2020 | Stiennon et al., *Learning to summarize with human feedback* (OpenAI) | RLHF for summarization. First "modern" RLHF pipeline. |
| 2022 | Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT) | The blueprint for ChatGPT. |

## Lecture 06 — DPO and direct preference optimization

| Year | Paper | Why read it |
|---|---|---|
| 2023 | Rafailov et al., *Direct Preference Optimization* (DPO) | RLHF without RL. A clean reformulation. |
| 2024 | Hong et al., *ORPO: Monolithic Preference Optimization* | A further simplification. |

## Lecture 07 — GRPO and reasoning RL

| Year | Paper | Why read it |
|---|---|---|
| 2024 | DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL* | GRPO. The current state of the art for open-source reasoning. |
| 2024 | Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* | The original GRPO paper. |

## Lecture 08 — VLA / embodied RL

| Year | Paper | Why read it |
|---|---|---|
| 2023 | Octo team, *Octo: An Open-Source Generalist Robot Policy* | A general-purpose VLA. |
| 2024 | Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model* | The open reference for VLA architecture. |
| 2024 | Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control* | Physical Intelligence's reference model. |

---

PDFs of each paper, where licensing allows, live in `/resources/papers/`. The rest are linked above.
