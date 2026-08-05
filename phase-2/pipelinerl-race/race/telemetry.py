"""The event stream.

ONE append-only JSONL stream is the single source of truth for a run. Both the metric
figures and the Three.js visualizer are computed from it, so the picture the cohort sees
can never drift from what the GPUs actually did.

Design rules:
  * `t` is seconds since run start (float), monotonic — never wall-clock, so replays line up.
  * Every GPU is ALWAYS in exactly one state. A state change emits one event. The visualizer
    reconstructs occupancy by holding the last state per GPU until the next event.
  * `idle` is emitted explicitly and eagerly. The idle bubble is the entire pedagogical point
    of this lecture; a stream that omits idleness would make the two arms look identical.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

GpuState = Literal[
    "idle",            # allocated but doing nothing — THE BUBBLE
    "generating",      # vLLM decode
    "prefill",         # vLLM prefill
    "forward",         # trainer forward pass
    "backward",        # trainer backward pass
    "optimizer_step",  # trainer optimizer step
    "weight_sync",     # sending or receiving weights
    "waiting",         # blocked on a queue/barrier (distinct from idle: work exists elsewhere)
]
Role = Literal["infer", "train", "hybrid"]


@dataclass
class Telemetry:
    """Thread-safe JSONL event writer with an in-memory tail for the live endpoint."""

    path: Path
    arm: str
    n_gpus: int
    live_tail: int = 20_000
    _t0: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _fh: Any = None
    _tail: list[dict] = field(default_factory=list)
    _last_state: dict[int, str] = field(default_factory=dict)
    _closed: bool = False

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line-buffered: the viz can tail it live
        self.emit("run", detail={"arm": self.arm, "n_gpus": self.n_gpus})

    # ---- core ----------------------------------------------------------------
    def emit(self, kind: str, **fields: Any) -> None:
        # The timestamp MUST be taken inside the lock. The pipeline arm emits from two
        # threads (producer generating, consumer training); if t were sampled before
        # acquiring the lock, two threads could stamp t1 < t2 and then write in the
        # opposite order, producing a non-monotonic stream. Downstream consumers integrate
        # dt over that stream, so out-of-order lines silently inflate GPU-seconds — which
        # is exactly how this bug was caught (idle_gpu_s exceeded total GPU-seconds).
        with self._lock:
            if self._closed:
                # The pipeline arm's producer thread can outlive the consumer on an error
                # path and keep emitting after close(), which raised
                # "ValueError: I/O operation on closed file" and masked the REAL failure
                # underneath it. Dropping late events is correct: the run is already over.
                return
            ev = {"t": round(time.monotonic() - self._t0, 4), "kind": kind, **fields}
            line = json.dumps(ev, separators=(",", ":"))
            self._fh.write(line + "\n")
            self._tail.append(ev)
            if len(self._tail) > self.live_tail:
                del self._tail[: len(self._tail) // 4]

    # ---- typed helpers -------------------------------------------------------
    def gpu(self, gpu: int, role: Role, state: GpuState, **detail: Any) -> None:
        """Record a GPU state transition. Repeated identical states are suppressed."""
        with self._lock:
            if self._last_state.get(gpu) == state and not detail:
                return
            self._last_state[gpu] = state
        self.emit("gpu", gpu=gpu, role=role, state=state, detail=detail or None)

    def weights(self, step: int, src: list[int], dst: list[int], nbytes: int,
                ms: float, inflight: bool, transport: str = "ipc") -> None:
        """A weight transfer from trainer ranks to inference ranks.

        `inflight=True` means generation did NOT stop and the KV cache survived — the
        PipelineRL claim. `inflight=False` is a stop-the-world sync (conventional arm).
        """
        self.emit("weights", step=step, src=src, dst=dst, bytes=nbytes,
                  ms=round(ms, 2), inflight=inflight, transport=transport)

    def sample(self, reward: float, tokens: int, lag: int, prompt_id: str | None = None) -> None:
        """One finished rollout. `lag` = how many optimizer steps stale the weights were
        that produced it. lag=0 is perfectly on-policy; conventional RL is always 0."""
        self.emit("sample", reward=reward, tokens=tokens, lag=lag, prompt_id=prompt_id)

    def metric(self, step: int, **values: Any) -> None:
        self.emit("metric", step=step, **values)

    def phase(self, name: str, **detail: Any) -> None:
        """Coarse phase marker, e.g. conventional's generate/train alternation."""
        self.emit("phase", name=name, detail=detail or None)

    # ---- live endpoint support ----------------------------------------------
    def tail_since(self, t: float) -> list[dict]:
        with self._lock:
            return [e for e in self._tail if e["t"] > t]

    def checkpoint(self) -> None:
        """Flush to disk AND persist to the Modal volume.

        Modal Volumes only persist on an explicit commit() or a CLEAN container exit. A hard
        abort (e.g. vLLM's engine process calling std::terminate) loses everything written
        since the last commit — we lost ~45 minutes of a pipeline run this way, keeping only
        the 3 events that happened to land before the crash, while the driver log proved the
        run had gone much further.

        Called periodically from both arms so a crashed run still yields a partial, usable
        stream. Cheap: commit() is incremental.
        """
        with self._lock:
            if self._closed:
                return
            self._fh.flush()
        try:
            import modal
            modal.Volume.from_name("race-data").commit()
        except Exception:
            pass          # running outside Modal (local replay) — flushing is enough

    def close(self) -> None:
        self.emit("end")
        with self._lock:
            self._closed = True
            self._fh.close()
        try:
            import modal
            modal.Volume.from_name("race-data").commit()
        except Exception:
            pass


class GpuSpan:
    """Context manager so a block of work is always bracketed by a state and a return-to.

        with GpuSpan(tel, 0, "train", "backward"):
            loss.backward()

    Guarantees the GPU is marked `idle` on exit unless another span takes over, which is
    what keeps the bubble honest — you cannot forget to close a state.
    """

    def __init__(self, tel: Telemetry, gpu: int, role: Role, state: GpuState,
                 after: GpuState = "idle", **detail: Any):
        self.tel, self.gpu, self.role, self.state, self.after, self.detail = (
            tel, gpu, role, state, after, detail)

    def __enter__(self):
        self.tel.gpu(self.gpu, self.role, self.state, **self.detail)
        self._t = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.tel.gpu(self.gpu, self.role, self.after)
        return False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t
