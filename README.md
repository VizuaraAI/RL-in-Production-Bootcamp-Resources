# RL in Production — Bootcamp Resources

The companion repository for the **RL in Production** intensive workshop by Vizuara AI Labs.

Course site: <https://rl-production.vizuara.ai>

---

## What's in here

| Path | Contents |
|---|---|
| [`lectures/`](./lectures/) | One folder per lecture. Each contains a syllabus outline, reading list, and links to the slide deck. |
| [`code/`](./code/) | Hands-on, runnable implementations of every algorithm we cover. Pedagogical first — short, commented, single-file where possible. |
| [`resources/`](./resources/) | Curated reading lists, paper PDFs, and links to external talks. |

## How to use this repo as a student

1. Read the lecture outline in `lectures/<n>-<title>/README.md` *before* attending.
2. Run the corresponding code in `code/<topic>/` *during or after* the lecture.
3. Work through the suggested papers from `resources/`.

## Lecture index

| # | Topic | Status |
|---|---|---|
| 01 | [Fundamentals — MDPs, value, Bellman, DP/MC/TD](./lectures/01-fundamentals/) | ✅ Live |
| 02 | Q-learning and DQN | 🟡 Upcoming |
| 03 | Policy gradients (REINFORCE → TRPO → PPO) | 🟡 Upcoming |
| 04 | Actor-critic and GAE | 🟡 Upcoming |
| 05 | RLHF — language meets RL | 🟡 Upcoming |
| 06 | DPO and direct preference optimization | 🟡 Upcoming |
| 07 | GRPO and reasoning RL | 🟡 Upcoming |
| 08 | Embodied RL and VLA models | 🟡 Upcoming |
| … | … | … |

## Code index

| Path | What it teaches |
|---|---|
| [`code/tic-tac-toe/`](./code/tic-tac-toe/) | DP vs MC vs TD value learning on a small, exhaustively-solvable MDP. The cleanest setting in which to *see* the three families of algorithms differ. |

---

## License

MIT. Use freely for teaching, learning, and research.

## Citing

If you build on this material, please cite the bootcamp:

```bibtex
@misc{vizuara_rl_in_production_2026,
  title  = {RL in Production — Cohort 2026},
  author = {Vizuara AI Labs},
  year   = {2026},
  howpublished = {\url{https://rl-production.vizuara.ai}},
}
```
