"""One config object, shared by both arms.

The fairness contract from PLAN.md §2 is enforced structurally: everything that must be
identical between the arms lives here, and the ONLY field an arm is allowed to interpret
differently is `n_infer_gpus` (how the same total N is partitioned).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal


@dataclass
class RaceConfig:
    # ---- identical across arms (the fairness contract) ----------------------
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    task: Literal["gsm8k", "countdown"] = "gsm8k"
    seed: int = 0

    group_size: int = 8          # G rollouts per prompt (GRPO group)
    prompts_per_step: int = 16   # optimizer step sees group_size * prompts_per_step samples
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0

    # Micro-batch for the trainer's forward/backward. A full RL batch's [B, T, vocab]
    # logits are ~45 GiB at Qwen's 151,936 vocab and OOM'd an 80 GiB H100. Gradients are
    # accumulated so the optimizer step is mathematically unchanged. Part of the fairness
    # key: both arms must micro-batch identically or their gradients differ.
    micro_batch_size: int = 8

    lr: float = 1e-6
    kl_coef: float = 0.0         # PipelineRL paper runs KL-free; keep 0 unless ablating
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0

    n_gpus: int = 4              # TOTAL budget — identical for both arms

    # ---- the budget the race is run against --------------------------------
    # Headline run: fixed wall-clock, compare reward (who learns more per second).
    # Control run:  fixed sample count, compare reward (who learns more per sample).
    budget: Literal["wallclock", "samples"] = "wallclock"
    wallclock_s: int = 45 * 60
    max_samples: int = 40_000
    max_steps: int = 10_000      # hard backstop

    # ---- the ONLY arm-dependent knob ---------------------------------------
    # Conventional ignores this (all N GPUs alternate roles).
    # PipelineRL splits: n_infer_gpus generate, n_gpus - n_infer_gpus train.
    n_infer_gpus: int = 2

    # ---- memory ------------------------------------------------------------
    # Colocated (conventional) must leave room for the trainer's peak: params + grads +
    # AdamW moments + activations, all resident at the moment vLLM wakes to take weights.
    # 0.55 was not enough on a 22 GiB L4 and OOM'd in vLLM's cumem allocator.
    gpu_mem_util_colocated: float = 0.42
    gpu_mem_util_dedicated: float = 0.82

    # ---- PipelineRL-specific -----------------------------------------------
    max_lag: int = 8             # drop samples staler than this many optimizer steps
    inflight_updates: bool = True

    # ---- bookkeeping --------------------------------------------------------
    arm: Literal["conventional", "pipeline", "pipeline_async"] = "conventional"
    run_id: str = ""
    out_dir: str = "/data/runs"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.arm in ("pipeline", "pipeline_async"):
            if not (0 < self.n_infer_gpus < self.n_gpus):
                raise ValueError(
                    f"pipeline arm needs 0 < n_infer_gpus < n_gpus, "
                    f"got {self.n_infer_gpus}/{self.n_gpus}")
        if self.group_size < 2:
            raise ValueError("GRPO needs group_size >= 2 to form an advantage baseline")

    @property
    def n_train_gpus(self) -> int:
        return self.n_gpus if self.arm == "conventional" else self.n_gpus - self.n_infer_gpus

    @property
    def infer_gpu_ids(self) -> list[int]:
        if self.arm == "conventional":
            return list(range(self.n_gpus))          # colocated: every GPU generates
        return list(range(self.n_gpus - self.n_infer_gpus, self.n_gpus))  # tail GPUs

    @property
    def train_gpu_ids(self) -> list[int]:
        if self.arm == "conventional":
            return list(range(self.n_gpus))          # colocated: every GPU trains
        return list(range(self.n_gpus - self.n_infer_gpus))

    @property
    def samples_per_step(self) -> int:
        return self.group_size * self.prompts_per_step

    def preflight(self) -> None:
        """Validate the model/topology combination BEFORE any GPU is touched.

        vLLM shards attention heads across tensor-parallel ranks, so TP must divide the
        model's head count. This bit us on a 4xH100 launch: Qwen2.5-0.5B has 14 heads, the
        conventional arm uses TP = n_gpus = 4, and 14 % 4 != 0 — the run died 22 seconds in
        with a pydantic ValidationError, after the GPUs were already allocated.

        The conventional arm is the strict one: it shards across ALL n_gpus. The pipeline
        arm only shards across n_infer_gpus, which is why it survived that launch and made
        the failure look arm-specific when it is really a config error.
        """
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(self.model)
        heads = getattr(hf, "num_attention_heads", None)
        kv = getattr(hf, "num_key_value_heads", heads)
        if not heads:
            return                      # unknown architecture; let vLLM decide

        tps = {"conventional": self.n_gpus, "pipeline": self.n_infer_gpus,
               "pipeline_async": self.n_infer_gpus}
        tp = tps[self.arm]
        if heads % tp:
            valid = [d for d in range(1, heads + 1) if heads % d == 0 and d <= self.n_gpus]
            raise ValueError(
                f"{self.model} has {heads} attention heads, which is not divisible by "
                f"tensor-parallel size {tp} (arm={self.arm}, n_gpus={self.n_gpus}). "
                f"vLLM will refuse to build the engine. Valid TP values at this GPU count: "
                f"{valid}. Either pick a model whose head count divides {tp}, or change the "
                f"GPU split.")
        if kv and tp % kv and kv % tp:
            raise ValueError(
                f"{self.model} has {kv} key-value heads, incompatible with TP={tp}: vLLM "
                f"needs TP divisible by kv-heads or vice versa.")

        # The Trainer is SINGLE-GPU (engine.Trainer pins self.dev = gpu_ids[0] and never
        # shards). So allocating more than one training GPU strands the rest: they are
        # counted in the budget, marked `waiting`, and do no compute.
        #
        # This actually happened. race4 ran the concurrent arms with train=[0,1]; GPU 1 sat
        # at waiting=100% for 45 minutes, so those arms raced on 3 effective GPUs against
        # conventional's 4 — and "conventional wins on final reward" was an artefact of the
        # handicap. Caught only by a per-GPU breakdown after the fact.
        #
        # Until the Trainer supports FSDP/DDP, the only fair pipeline split is
        # n_infer_gpus = n_gpus - 1.
        if self.arm in ("pipeline", "pipeline_async") and self.n_train_gpus != 1:
            raise ValueError(
                f"pipeline arms allocate {self.n_train_gpus} training GPUs but the Trainer "
                f"is single-GPU, so {self.n_train_gpus - 1} would sit idle and the arm "
                f"would race on fewer GPUs than its budget. Set n_infer_gpus="
                f"{self.n_gpus - 1} (got {self.n_infer_gpus}), or teach Trainer to shard.")

    def fairness_key(self) -> dict:
        """Everything that MUST match between arms. Compared post-run; a mismatch
        invalidates the comparison and is reported rather than quietly ignored.

        Includes the SOFTWARE ENVIRONMENT, not just hyperparameters. This was a real gap:
        the in-flight arm needs vLLM >= 0.18 for pause_generation/resume_generation, which
        do not exist in 0.11. Running one arm on a different engine version would confound
        "scheduling" with "entire inference engine upgraded" — and the key would have
        passed, because it only compared config fields. If the versions differ, the arms
        are not comparable and publish_run.py must refuse.
        """
        d = asdict(self)
        for k in ("arm", "run_id", "n_infer_gpus", "notes", "out_dir",
                  "max_lag", "inflight_updates"):
            d.pop(k, None)
        d["_env"] = self.env_fingerprint()
        return d

    @staticmethod
    def env_fingerprint() -> dict:
        """Versions that materially affect numerics or scheduling."""
        out = {}
        for mod, attr in (("vllm", "__version__"), ("torch", "__version__"),
                          ("transformers", "__version__")):
            try:
                out[mod] = getattr(__import__(mod), attr, "?")
            except Exception:
                out[mod] = "absent"
        try:
            import torch
            out["gpu"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu")
        except Exception:
            out["gpu"] = "?"
        return out

    def to_dict(self) -> dict:
        return asdict(self)
