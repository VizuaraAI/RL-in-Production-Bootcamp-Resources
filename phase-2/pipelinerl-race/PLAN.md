# The Race: veRL-style Conventional RL vs PipelineRL

**Lecture:** RL in Production, Phase 2, Project 06 (veRL & PipelineRL)
**Deck:** https://rl-bootcamp-decks.vercel.app/lecture-p2-verl-pipelinerl/
**Owner:** Rajat Dandekar, Vizuara AI Labs

---

## 0. Locked decisions (recovered from session 2026-08-05 — do not re-litigate)

| Question | Decision |
|---|---|
| Deck relationship | New deck, **full veRL re-teach** — the cohort has never been taught veRL |
| What is the baseline? | **PipelineRL vs conventional / veRL-style RL.** The PipelineRL paper itself states: *"veRL implements Conventional RL efficiently… we believe veRL's throughput would be similar to our Conventional RL baseline."* So we implement the conventional loop faithfully rather than shelling out to the veRL library. |
| Compute | **Modal, 2–8 GPUs, small model.** Qwen2.5-0.5B for cohort iteration; 1.5B for the headline run. Math task with a verifiable reward. |
| Visualizer | **Three.js interactive page.** Rajat: *"beautifully show how the GPUs are filled up and how the weights are transferred… which GPUs are filled is a black box, so we need to show exactly which GPU is used when."* |
| Audience goal | The **cohort runs this themselves**, produces the figures, and uploads results to a dashboard. |

**Expected result (Rajat's prior):** PipelineRL is faster. The paper reports ~2×. We report what we measure — if it does not reproduce, that is the finding.

---

## 1. The scientific claim we are testing

> For a fixed GPU budget and identical algorithm/model/data/hyperparameters,
> **PipelineRL reaches a given reward faster in wall-clock time than conventional
> batched RL**, because it removes the generate/train alternation bubble and keeps
> the data fresher via in-flight weight updates.

Two things are being measured, and they must not be conflated:

- **Throughput** — samples/sec, tokens/sec, GPU busy-fraction. Systems metric.
- **Effectiveness** — reward per sample. Algorithmic metric, degraded by stale data.

`learning speed = effectiveness × throughput`. Conventional RL maximizes effectiveness
(perfectly on-policy) at the cost of throughput (idle bubbles). PipelineRL claims you can
have most of both. **Reward-vs-wall-clock is the headline plot**; reward-vs-samples is the
control that shows what you paid in off-policyness.

## 2. Fairness contract (this is what makes it a real experiment)

Both arms MUST share, bit-for-bit where possible:

- Same base model, same seed, same init
- Same task, same prompt set, same ordering seed
- **Same GRPO implementation** — one `grpo.py`, imported by both arms. No arm gets its own loss.
- Same hyperparameters: lr, group size G, batch size, max new tokens, temperature, KL coef
- **Same total GPU count N.** The arms differ only in how N is *partitioned and scheduled*:
  - Conventional: all N GPUs alternate generate → train → generate → train
  - PipelineRL: k GPUs generate continuously, N−k GPUs train continuously
- Same wall-clock budget for the headline run (race to a fixed time, compare reward),
  AND same sample budget for the control run.

Anything that differs beyond the above is a confound and must be recorded in `reports/confounds.md`.

## 3. Architecture

```
race/
  config.py        RaceConfig — model, task, N gpus, arm, hparams, seed
  data.py          GSM8K load + strict answer verification (the reward)
  grpo.py          SHARED GRPO: group-normalized advantage + policy loss
  telemetry.py     the event stream (see §4) — the visualizer's data source
  arm_conventional.py   generate-then-train, all GPUs alternate phases
  arm_pipeline.py       concurrent gen+train, in-flight weight updates
  metrics.py       reward/throughput/utilization aggregation -> figures
modal/
  image.py         the shared training image (vLLM + torch + trl-free custom loop)
  run_race.py      entrypoints: smoke, race, replay
viz/               Three.js real-time GPU + weight-transfer visualizer
dashboard/         cohort uploads their run JSON -> comparison figures
```

## 4. Telemetry — the contract between the run and the visualizer

Every state change appends one JSON line to `events.jsonl` (and streams to the live endpoint).
This single stream drives **both** the metrics figures and the Three.js visualizer, so the
visualizer can never drift from what actually happened.

```jsonc
{"t": 12.481, "kind": "gpu",    "gpu": 3, "role": "infer",  "state": "generating",
 "detail": {"seqs_in_flight": 48, "tokens_s": 1820}}
{"t": 12.«», "kind": "gpu",    "gpu": 0, "role": "train",  "state": "optimizer_step", "step": 41}
{"t": 12.«», "kind": "weights", "step": 41, "src": [0,1], "dst": [2,3],
 "bytes": 988_000_000, "ms": 310, "inflight": true}
{"t": 12.«», "kind": "sample",  "reward": 1.0, "lag": 2, "tokens": 271}
```

`state` ∈ `idle | generating | forward | backward | optimizer_step | weight_sync | waiting`.
**`idle` is the whole point of the lecture** — it is the bubble the visualizer must make visceral.

## 5. Deliverables

1. **The runnable project** — cohort clones, sets Modal creds, runs `modal run modal/run_race.py::race --arm both`.
2. **Real figures from a real run** we execute ourselves (not simulated).
3. **The Three.js visualizer** — live during a run, and replayable from a recorded `events.jsonl`.
4. **The dashboard** — cohort uploads their `results.json`, sees their run against ours.
5. A short write-up of what we measured, including any failure to reproduce.

## 6. Honest risks

- ~~**In-flight weight update is the hard part.**~~ **RESOLVED 2026-08-05** — verified on
  2×L4, vLLM 0.11.0 / torch 2.8.0 (raw output in `reports/probe_result.txt`):

  ```
  ipc:              OK changed=True {'loaded': 170, 'transport': 'ipc'} in 567ms
  inflight_update:  OK 279 tokens, update 307ms,
                    kv_blocks [111432, 111432] -> [111432, 111432] (RETAINED)
  sleep_wake:       OK freed=12.26GiB restored=10.76GiB
  ```

  Generation continued through the push and the KV block count was unchanged. No fallback is
  needed and no figure carries an asterisk. Four things had to be found on a real GPU to get
  here, all now encoded in the code and its comments:

  1. Raw CUDA tensors cannot cross `collective_rpc` (numpy conversion error).
  2. The default msgpack serializer rejects CUDA-IPC handles, so
     `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is **required** in the image.
  3. `reduce_tensor(t)[1]` already carries `torch.Tensor` at index 0 — do not prepend it.
  4. **An exception inside `collective_rpc` poisons the executor**: every later RPC returns
     the same error. This produced a false "sleep/wake failed" report on a component that
     had passed twice. Both arms now treat a failed weight push as fatal
     (`EngineWeightError`) instead of retrying past it — retrying would leave the generator
     serving stale weights for the whole run while the reward curve kept moving.

  CPU staging was implemented, tested, and **deleted** — vLLM mangles plain CPU tensors into
  nested lists. CUDA IPC needs trainer and inference on one node, which Modal guarantees.

  **CORRECTION (same day).** The above proved the *engine* tolerates a mid-decode update.
  It did NOT give us in-flight updates in the run, and I wrongly concluded they were
  impossible after the synchronous `LLM` API crashed under concurrent use:

  ```
  vllm/v1/engine/core.py process_input_sockets
  msgspec.ValidationError: Expected `array`, got `int`  ->  std::terminate
  ```

  That crash is real — `LLM` is not thread-safe, because vLLM V1 runs its engine core in a
  separate process behind a ZMQ socket. But the correct response was not "in-flight is
  impossible", it was "use the right API". Checking ServiceNow/pipelinerl (the reference
  implementation) shows exactly how:

  ```python
  await engine.pause_generation(mode="keep", clear_cache=False)
  await engine_client.collective_rpc_async("receive_weight_update", ...)
  await engine.resume_generation()
  ```

  `mode="keep", clear_cache=False` pauses between engine STEPS, retains in-flight requests
  and their KV cache, and resumes — so a sequence continues decoding under new weights.
  Everything on one asyncio loop, which is also why they never hit the socket race.

  Verified on GPU: those methods are **MISSING in vLLM 0.11.0** and **present in 0.18.1**
  (the version pipelinerl pins). So the blocker was a version, not a limitation.

  Consequences, all now enforced:
  - the image moved to `vllm==0.18.1` / torch 2.10; 0.18.1 still exposes `LLM.sleep` and
    `LLM.collective_rpc`, so ONE image serves all three arms;
  - `fairness_key()` now includes an **environment fingerprint** (vllm/torch/transformers
    versions + GPU name). Running one arm on a different engine version would confound
    "scheduling" with "engine upgraded", and the previous key compared only config fields —
    it would have passed silently;
  - every arm must be re-run on the new image. A completed 45-minute conventional run on
    0.11.0 was discarded for exactly this reason.

## 6b. Three conditions, not two

Because the batch-boundary variant exists and works, we run it as its own arm rather than
throwing it away. This is a finer experiment than the paper's:

| arm | GPUs | weight updates |
|---|---|---|
| A `conventional` | all alternate generate/train | stop-the-world |
| B `pipeline` | split, concurrent | between generation batches (sync `LLM` + lock) |
| C `pipeline_async` | split, concurrent | **mid-decode, KV cache retained** (`AsyncLLM`) |

**A → B** isolates what *concurrency* alone buys.
**B → C** isolates what *in-flight updates* specifically buy on top of it.
The paper reports only the combined effect.
- **A 0.5B model on GSM8K may have too weak a reward signal** to show a reward-vs-time
  separation in a short run. Mitigation: use countdown/arithmetic with dense verifiable reward,
  or accept that the headline becomes a throughput result with reward as a secondary plot.
- **Small N (2–8 GPUs) compresses the effect.** The bubble grows with cluster size and
  sequence-length variance; at N=4 the gap will be smaller than the paper's. Report honestly.
