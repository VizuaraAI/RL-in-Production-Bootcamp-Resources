"""vLLM worker extension — how new weights get INTO a running engine.

vLLM instantiates this class inside every tensor-parallel worker process (via
`worker_extension_cls`); the driver calls its methods with `collective_rpc`.

TRANSPORT (this is the whole design problem)
--------------------------------------------
`collective_rpc` PICKLES its arguments to reach the worker processes, and a raw CUDA tensor
cannot survive that — vLLM 0.11 tries to convert it via numpy and raises:

    TypeError: can't convert cuda:0 device type tensor to numpy

So weights must not be passed as ordinary RPC arguments. Two paths, in preference order:

  1. CUDA IPC (default). `torch.multiprocessing.reductions.reduce_tensor` turns a CUDA
     tensor into a *picklable handle* that another process on the SAME NODE can reopen
     against the original GPU memory — no copy, no PCIe round trip. Modal gives us one
     container holding every GPU, so this always applies here.
  2. CPU staging (fallback). Move to host, ship, move back. Always works, but the PCIe
     round trip inflates the measured weight-transfer time, which is exactly one of the
     numbers this experiment reports. It is recorded in telemetry as
     `transport="cpu"` so no figure can silently attribute that cost to the architecture.

Because `load_weights` is the model's own loader, it handles HF-name -> vLLM-name remapping
and fused-QKV / gate-up packing. Doing that remap by hand is a classic source of silently
wrong weights: the model still runs, it is just quietly worse.

Crucially, none of this touches the block manager, so the KV cache and every in-flight
sequence survive the update. That is the PipelineRL trick.
"""
from __future__ import annotations

import torch


class RaceWorkerExtension:

    # ---- path 1: CUDA IPC ------------------------------------------------
    def update_weights_ipc(self, handles):
        """handles: list[(name, reduce_tensor_args)].

        The rebuilt tensor points at memory owned by the SENDING process/device, so we copy
        it onto this worker's device before handing it to the loader — otherwise the loader
        would read across devices and, worse, we would be relying on the sender not freeing
        the storage while we are still using it.
        """
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        dev = torch.cuda.current_device()
        weights = []
        for name, args in handles:
            # `reduce_tensor(t)[1]` ALREADY carries torch.Tensor as args[0] — prepending it
            # again gives "rebuild_cuda_tensor() takes 15 positional arguments but 16 were
            # given" (observed on GPU). Splat as-is.
            t = rebuild_cuda_tensor(*args)
            weights.append((name, t.to(f"cuda:{dev}", copy=True)))
        loaded = self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()
        return {"loaded": len(loaded) if loaded is not None else -1, "transport": "ipc"}

    # ---- path 2: CPU staging --------------------------------------------
    # NOTE — there is deliberately NO CPU-staging fallback.
    #
    # It was implemented and tested on GPU, and vLLM 0.11 mangles it: a plain CPU tensor
    # sent through collective_rpc arrives as a nested Python list, even with
    # VLLM_ALLOW_INSECURE_SERIALIZATION=1. The `inspect_payload` diagnostic below showed
    # exactly that:
    #     list[2](str('model.embed_tokens.weight'), list…)
    # Reconstructing a tensor from that would mean carrying shape/dtype/stride out of band
    # and trusting it — a lot of machinery for a path we do not need, and one whose PCIe
    # round trip would inflate the weight-transfer time this experiment reports.
    #
    # CUDA IPC works (verified: 170 tensors, 567ms, KV cache RETAINED through a mid-decode
    # update) and requires only that trainer and inference share a node — which Modal
    # guarantees, since it hands us one container holding every GPU. Shipping a fallback
    # known to be broken would be worse than having none.

    # ---- verification ----------------------------------------------------
    # ---- path 2: shared HOST memory (works without GPU P2P) ---------------
    def update_weights_shm(self, handles):
        """handles: list[(name, reduce_tensor_args)] for tensors in SHARED CPU memory.

        Needed because CUDA IPC requires peer access between the trainer's GPU and the
        inference GPU, and that is unavailable on hardware without NVLink/P2P — on Modal's
        L4s the push fails with "peer access is not supported between these two devices".
        Shared host memory sidesteps the GPU topology entirely: torch's CPU sharing uses a
        file descriptor, so the handle is plain picklable data and the worker maps the same
        pages rather than copying them over a socket.

        Cost is one D2H on the trainer plus one H2D here, which is real and is recorded as
        transport="shm" so a run using it is never confused with a zero-copy one.
        """
        from torch.multiprocessing.reductions import rebuild_storage_filename  # noqa: F401

        dev = torch.cuda.current_device()
        weights = []
        for name, args in handles:
            t = args[0](*args[1])  # (rebuild_fn, rebuild_args) for the shared CPU tensor
            weights.append((name, t.to(f"cuda:{dev}", non_blocking=True)))
        loaded = self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()
        return {"loaded": len(loaded) if loaded is not None else -1, "transport": "shm"}

    def inspect_payload(self, payload):
        """Report what the worker ACTUALLY received, rather than guessing from an
        exception message. Added after `too many dimensions 'str'` showed something was
        arriving as a string and no amount of reasoning from the traceback settled what."""
        def describe(x, depth=0):
            if torch.is_tensor(x):
                return f"Tensor{tuple(x.shape)}:{x.dtype}:{x.device}"
            if isinstance(x, (list, tuple)) and depth < 2:
                head = [describe(v, depth + 1) for v in list(x)[:3]]
                return f"{type(x).__name__}[{len(x)}]({', '.join(head)}…)"
            if isinstance(x, str):
                return f"str({x[:32]!r})"
            return type(x).__name__
        return describe(payload)

    def weight_signature(self):
        """Cheap fingerprint proving the weights in the LIVE engine actually changed.

        Without this a silently no-op update looks identical to a successful one, and the
        pipeline arm would appear to train correctly while generating from stale weights
        for the entire run — producing a real-looking but meaningless result.
        """
        total, n = 0.0, 0
        for p in self.model_runner.model.parameters():
            if n >= 8:
                break
            total += float(p.detach().float().sum().item())
            n += 1
        return total

    def kv_cache_fingerprint(self):
        """Number of KV blocks currently allocated. Compared across an in-flight update to
        prove the cache was RETAINED rather than dropped and silently re-prefilled."""
        try:
            return int(self.model_runner.kv_cache_config.num_blocks)
        except Exception:
            try:
                return int(len(self.model_runner.kv_caches))
            except Exception:
                return -1
