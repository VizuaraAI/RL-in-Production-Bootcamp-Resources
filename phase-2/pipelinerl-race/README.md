# Two Kinds of Asynchrony — separating concurrency from in-flight weight updates

**RL in Production · Phase 2 · Project 06**
Project page: **https://pipelinerl-race.vercel.app** ·
Lecture: [veRL & PipelineRL](https://rl-bootcamp-decks.vercel.app/lecture-p2-verl-pipelinerl/)

Asynchronous RL breaks synchronous lockstep in **two independent ways** — *spatially*, by
letting different GPUs generate and train at the same moment, and *temporally*, by letting the
policy change while a sequence is still being written. Every reported speedup bundles both.
This project runs them apart.

You are going to run the same RL algorithm **three times**, on the same model, the same data
and the same number of GPUs — changing **only how the work is scheduled** — and measure what
each change actually buys.

Then you are going to *watch* it happen: a live 3D view of which GPU is doing what, and
when the weights move.

---

## The question

Reinforcement learning on an LLM has two phases that fight each other:

- **Generation** — the policy writes answers. Memory-hungry, embarrassingly parallel, and
  it finishes at wildly different times per sequence.
- **Training** — you backprop through those answers. Compute-hungry, synchronous.

**Conventional RL** (what veRL does, efficiently) runs them in turn: every GPU generates,
everyone stops, every GPU trains, repeat. Simple, and perfectly on-policy — but while the
cluster waits for the single slowest sequence, most of it is doing nothing.

**PipelineRL** splits the cluster instead: some GPUs generate *forever*, others train
*forever*, and new weights are pushed into the generators **mid-sequence**, without
stopping them or throwing away the KV cache.

The trade is explicit:

```
learning speed  =  effectiveness  ×  throughput
                   (reward/sample)   (samples/sec)
```

Conventional maximises effectiveness (data is always fresh) and pays in throughput.
PipelineRL buys throughput and pays a little effectiveness (data is slightly stale).
**Which wins is an empirical question.** That is what you are measuring.

---

## Run it

### 0. Prerequisites

```bash
pip install modal
modal token new        # one-time, free tier is enough for the smoke test
```

### 0b. Run the math tests (seconds, free, no GPU)

```bash
python3 tests/test_math.py        # expects: ALL PASS
```

Two of this repo's memory optimisations — micro-batched gradient accumulation and chunked
log-probs — exist because a full `[batch, seq, 151936]` logits tensor OOMs an 80 GiB H100.
Both change *how* a quantity is computed, and neither is allowed to change *what* it is.
These tests prove that. Run them before anything you pay for.

### 1. Smoke test first (~10 min, 2×L4, a few dollars)

Never spend H100 hours on an unvalidated loop.

```bash
modal run --detach runner/run_race.py::smoke
```

This runs **all three** arms with a tiny budget and prints a summary. You want `ok=True` for
each, matching sample counts, and `fairness key: IDENTICAL across arms ✓`.

The three arms are:

| arm | GPUs | weight updates |
|---|---|---|
| `conventional` | all alternate generate/train | stop-the-world |
| `pipeline` | split, concurrent | between generation batches |
| `pipeline_async` | split, concurrent | **mid-decode, KV cache retained** |

`conventional → pipeline` isolates what *concurrency* buys; `pipeline → pipeline_async`
isolates what *in-flight updates* buy on top of it.

### 1b. Choose a model your GPU count can actually shard

vLLM splits attention heads across tensor-parallel ranks, so **TP must divide the model's
head count** — and the conventional arm shards across *all* your GPUs. Get this wrong and
the run dies ~20 seconds in, after the GPUs are already allocated. `RaceConfig.preflight()`
now catches it before anything is launched, but pick knowingly:

| model | heads | usable GPU counts (conventional arm) |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 14 | **1 or 2 only** |
| Qwen2.5-1.5B-Instruct | 12 | 1, 2, 3, 4 |
| Qwen2.5-3B-Instruct | 16 | 1, 2, 4, 8 |

So 0.5B is fine for the 2-GPU smoke but **cannot** run the 4-GPU race. Use 1.5B there.

### 2. The real race (~45 min per arm, 2×H100)

```bash
modal run --detach runner/run_race.py::race --gpus 2 --minutes 45 \
  --model Qwen/Qwen2.5-1.5B-Instruct --arms all
```

Every arm gets **the same wall-clock budget**. `--detach` matters: the run survives your
laptop closing.

**Why 2 GPUs and not 4.** The trainer here is single-GPU, so a fair split needs
`n_infer = n_gpus - 1`. At 4 GPUs that means TP=3 for the generator — and vLLM cannot shard a
model with 2 key-value heads across 3 ranks. Two GPUs is the largest configuration in which
*no GPU is left stranded*, and `preflight()` will reject the alternatives rather than let you
run a handicapped arm. Going to 4+ needs a sharded trainer (FSDP/DDP) or data-parallel
inference engines — a genuinely good extension.

### 3. Validate, then make the four graphs

```bash
modal volume get race-data /runs/race1 ./runs/
python3 scripts/publish_run.py runs/race1          # REFUSES if any invariant fails
.venv-plot/bin/python scripts/four_graphs.py runs/race1 reports/race1
```

`publish_run.py` will not emit a figure if GPU-second accounting is inconsistent, if an arm
left a GPU idle, if the fairness keys differ, or if a stream is marked synthetic. That is the
point of it.

### 4. Watch it

```bash
cp -r runs/race1/* viz/runs/
cd viz && python3 -m http.server 8000     # open http://localhost:8000
```

Or drag any `events.jsonl` onto the page. Nothing is uploaded — it is read in your browser.

---

## What to look for

| Signal | Where | What it means |
|---|---|---|
| **idle GPU-seconds** | visualizer, bottom-right of each panel | Compute you paid for and did not use. The single clearest difference between the arms. |
| The batch draining | conventional panel, end of each generation phase | GPUs going pale one by one — the long tail. This is the bubble. |
| Slate pulses | pipeline panel | Weights moving *while* generation continues. |
| `mean_lag` | `reports/summary.json` | How stale PipelineRL's data was. If this is large, the generator is outrunning the trainer — re-split the GPUs. |
| `reward_vs_wallclock.png` | `reports/` | **The headline.** Who learned more per second. |
| `gpu_busy.png` | `reports/` | Who actually used the cluster. |

---

## The fairness contract

A comparison is only worth something if the two arms differ in exactly one way. This repo
enforces that structurally, not by good intentions:

- **One** GRPO implementation ([`race/grpo.py`](race/grpo.py)), imported by both arms.
- **One** batching path ([`race/batching.py`](race/batching.py)) — same padding, same masks.
- `RaceConfig.fairness_key()` strips the partition fields and is compared after the run.
  If anything else differs, the summary prints `!! FAIRNESS VIOLATION` and names the field.
- The only knob an arm may interpret differently is `n_infer_gpus` — how the same total N
  is split.

If you change a hyperparameter, change it for both arms or the result means nothing.

---

## Honest caveats

Read these before you quote a number.

- **Small clusters compress the effect.** The bubble grows with GPU count and with
  sequence-length variance. At N=4 the gap will be smaller than the paper's; do not expect
  to reproduce their headline ratio at this scale.
- **Staleness is measured at sequence completion**, which is a lower bound — a sequence can
  be part-generated by old weights and part by new. Reported `lag` is therefore optimistic.
- **A 0.5B policy may be too weak on GSM8K** to separate the arms on reward within 45
  minutes. If your reward curves are flat and overlapping, that is a signal about the
  *model*, not about the architecture — switch to `--task countdown` for a denser reward,
  or extend the budget, and say which you did.
- **If `inflight_updates` is false in your `result.json`**, your run quiesced generation
  around each weight update. That is a weaker version of PipelineRL and your numbers should
  be labelled as such.

Report what you measured, including a failure to reproduce. A negative result that is
honestly obtained is worth more to this cohort than a positive one that is not.

---

## Upload your run

Drop your `reports/summary.json` on the dashboard to see your run against everyone else's,
and against the reference run. Include your GPU type and count — they are the first thing
anyone will ask.

## Layout

```
race/
  config.py      the fairness contract, in code
  data.py        GSM8K / countdown + verifiable reward
  grpo.py        the ONE loss both arms use
  engine.py      vLLM generator + torch trainer
  batching.py    rollouts -> padded tensors
  telemetry.py   the event stream (drives figures AND the visualizer)
  arm_conventional.py   generate <-> train alternation
  arm_pipeline.py       concurrent, updates between batches
  arm_pipeline_async.py concurrent, TRUE in-flight updates (AsyncLLM)
  engine_async.py       AsyncLLM wrapper: pause(keep) -> update -> resume
  metrics.py     events.jsonl -> figures
runner/
  probe.py       validates the sync vLLM API on a real GPU before anything else
  probe_async.py validates AsyncLLM pause/resume — the in-flight primitive
  run_race.py    smoke / race entrypoints
tests/
  test_math.py   gradient/logprob invariants — run before any paid job
scripts/
  publish_run.py validate a finished run, then emit figures + site data
  assemble_site.sh
viz/             the Three.js visualizer
dashboard/       cohort run comparison
site/            the deployable landing page (index + viz + dashboard)
```

---

## The reference run

`reference_run/` contains our published 3-arm race (2×H100, Qwen2.5-1.5B, GSM8K, 45 min/arm),
gzipped. To replay it in the visualizer:

```bash
mkdir -p viz/runs/conventional viz/runs/pipeline viz/runs/pipeline_async
for a in conventional pipeline pipeline_async; do
  gunzip -c reference_run/$a.events.jsonl.gz > viz/runs/$a/events.jsonl
done
cd viz && python3 -m http.server 8000     # http://localhost:8000
```

Headline numbers, and the caveat that goes with them, are on the project page:
**https://pipelinerl-race.vercel.app**

| | conventional | concurrent | concurrent + in-flight |
|---|---|---|---|
| samples/sec | 14.34 | **29.37** (2.05×) | 27.39 (1.91×) |
| GPUs computing | 65.6% | 70.0% | **95.2%** |
| idle GPU-seconds | 1861 | 1638 | **258** |
| in-flight updates | 0 | 0 | **514 / 514** |
| mean lag (steps) | 0.0 | 0.051 | 0.035 |

Concurrency accounts for essentially the whole throughput gain. In-flight updates work exactly
as specified and buy nothing *at this scale*, because 95% of samples are already on-policy —
they should matter when the generator/trainer ratio, sequence length, or length variance is
much larger. The learning-quality comparison is **open**: the arms start at different rewards
because arm A generates at TP=2 and arms B/C at TP=1, so the generator differs and not only
the scheduler. See the paper's Limitations section.
