# Teaching Machines to Code — SWE-RL

> **Phase 2 · Project 05 · SWE-RL.** Reinforcement learning, pointed at *software engineering* —
> teaching a language model to fix real bugs and pass real tests, from the basics, through three real projects.

**Slides:** https://rl-bootcamp-decks.vercel.app/lecture-p2-teaching-machines-to-code/ ·
**Book:** https://teaching-machines-to-code.vercel.app ·
**Project-2 site:** https://swe-rl-ipr.vercel.app ·
**Before/after demo:** https://rl-teaches-code.vercel.app

This lecture builds one idea from the ground up — **the environment is the teacher**: nobody hand-labels the
right answer; a real environment (the *tests*, or a *terminal*) grades the model's own attempts, and RL
(**GRPO**) reinforces what worked. We then watch that single idea work at three very different scales.

## The three projects

**Project 1 — Mini-SWE-RL (RL on your laptop).** A from-scratch bug-fixing agent. `Qwen2.5-Coder-1.5B` on an
Apple M4 Pro, ~30 minutes, GRPO on a tiny `CodeFixEnv` where the tests hand back a 1 or a 0.
Result: **66.7% → 73.3%** solve rate, **7 new bugs** learned. Same algorithm as DeepSWE (which uses a 32B
model on 64 H100s for 6 days) — just miniaturised. Code: https://github.com/RajatDandekar/Mini-SWE-RL

**Project 2 — Agentic RL on real code.** Real programming tasks (MBPP+, 378 problems) with *hidden* tests,
trained on cloud GPUs (Modal, H100), across models from 0.5B to 7B. Real before/after code shows the model
going from failing a task to solving it (the featured 0.5B model: **44 → 51** held-out solved, **14 tasks
fixed**). Full research paper + code:
[`Mini-SWE-RL/swe-rl-ipr`](https://github.com/RajatDandekar/Mini-SWE-RL/tree/main/swe-rl-ipr)

**Project 3 — ECHO: a world model, for free.** A terminal agent trained with GRPO plus one extra job —
predict what the terminal will say back (`L_ECHO = L_GRPO + λ·L_env`, λ = 0.05). Learning to predict the
computer's replies gives the agent a *world model* for almost no extra compute, roughly **doubling** GRPO in a
controlled A/B on TerminalBench-2.0 (89 tasks). *Full 8B/14B runs are training on the cluster; the extra loss
is verified active and the small smoke test passed.* Code: [`echo-terminal-rl/`](echo-terminal-rl/) — the
~62-line SkyRL patch, the terminal rollout environment (modal.Sandbox), matched GRPO-vs-ECHO configs,
correctness tests, and the replication journal.

**Homework — the un-cheatable reward.** Because the reward is "did the visible tests pass?", a clever model
can learn to *game* the tests. One research direction (an *isomorphic perturbation reward*) re-grades each
attempt on altered-but-equivalent versions of the tests, so an answer shaped to the exact visible test no
longer scores. The paper + code are in `swe-rl-ipr/` — a perfect first research project.

## Layout

```
slides/    the Slidev lecture deck (slides.md + public/figures + config) — build with `npm i && npm run build`
book/      the companion book "Teaching Machines to Code" — 16 chapters (articles/*.md), the pinned FACTS.md,
           STYLE.md, build.py (static-site generator) and gen_figures.py (hand-drawn figure pipeline)
paper/     the Project-2 research paper (PDF)
echo-terminal-rl/  Project-3 code: the ECHO×SkyRL patch, terminal env, Modal trainers, tests, notes
```

The **65 hand-drawn figures** live in `slides/public/figures/`; the deck and the book share them. To rebuild
the book's figures from scratch, run `book/gen_figures.py` with a `GEMINI_API_KEY` set (never hardcode it).

## Reproduce

- **Project 1:** clone https://github.com/RajatDandekar/Mini-SWE-RL and follow its README (runs on a laptop).
- **Project 2:** see [`swe-rl-ipr/`](https://github.com/RajatDandekar/Mini-SWE-RL/tree/main/swe-rl-ipr) —
  `ipr/` (the reward + env), `scripts/` (GRPO trainers for Modal), `paper/`, and the two site sources.
- **Project 3:** see [`echo-terminal-rl/`](echo-terminal-rl/) — start with its README: verify the loss math
  on CPU, prove the rollout cycle on one task, then run the matched GRPO-vs-ECHO A/B on Modal.
- **The deck:** `cd slides && npm install && npm run dev` (or `npm run build`).

See `RESOURCES.md` for every link in one place.

*Vizuara AI Labs · RL in Production.*
