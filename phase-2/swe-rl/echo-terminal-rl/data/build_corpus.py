"""Materialize the ECHO training corpus (Harbor tasks -> SkyRL parquet) onto a Modal Volume.

Sources (public, verified):
  - obiwan96/endless-terminals            : per-task dirs (Harbor format)
  - open-thoughts/OpenThoughts-Agent-v1-RL: tasks.parquet, each row a tarball of a task dir

Output (on Volume `echo-data` at /data/corpus/<tag>/):
  train.parquet, validation.parquet  with columns:
    prompt[list[msg]], env_class="terminal", dockerfile, copy_files_b64, test_files_b64,
    instruction, max_turns, reward_spec, data_source, extra_info

Build small for the P0 smoke, then full:
  modal run data/build_corpus.py --limit 40  --tag smoke
  modal run data/build_corpus.py --limit 0   --tag full     # 0 = no limit
"""
import modal

app = modal.App("echo-build-corpus")
vol = modal.Volume.from_name("echo-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.25", "datasets", "pyarrow", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "120"})
)

SYSTEM_PROMPT = (
    "You are a terminal agent. You solve the task by issuing shell commands in a Linux "
    "container. On each turn, first think briefly, then issue EXACTLY ONE bash command inside a "
    "fenced code block:\n\n```bash\n<your command here>\n```\n\nYou will then see the command's "
    "output (stdout, stderr, and exit code). Continue issuing one command per turn until the task "
    "is complete. When you are confident the task is fully done, respond with:\n\n<task_complete>\n\n"
    "Do not issue more than one command per turn. Keep commands non-interactive."
)


def task_dir_to_row(root, source, max_turns=16):
    """root: pathlib.Path of a Harbor task dir. Returns a row dict or None if unsuitable."""
    import base64, json
    from pathlib import Path
    root = Path(root)
    env_dir = root / "environment"
    tests_dir = root / "tests"
    dockerfile = env_dir / "Dockerfile"
    instruction = root / "instruction.md"
    test_sh = tests_dir / "test.sh"
    # must be well-formed + verifiable
    if not (dockerfile.exists() and instruction.exists() and test_sh.exists()):
        return None
    df_text = dockerfile.read_text(errors="replace")
    # v1 replay uses an ubuntu base; keep ubuntu-based tasks (skip exotic bases for now)
    import re as _re
    from_lines = _re.findall(r"(?im)^\s*FROM\s+(\S+)", df_text)
    if from_lines and not any("ubuntu" in f.lower() for f in from_lines):
        return None
    copy_files = {str(f.relative_to(env_dir)): base64.b64encode(f.read_bytes()).decode()
                  for f in env_dir.rglob("*") if f.is_file() and f.name != "Dockerfile"}
    test_files = {str(f.relative_to(tests_dir)): base64.b64encode(f.read_bytes()).decode()
                  for f in tests_dir.rglob("*") if f.is_file()}
    instr_text = instruction.read_text(errors="replace")
    return {
        "data_source": source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instr_text},
        ],
        "env_class": "terminal",
        "dockerfile": df_text,
        # JSON strings (NOT dicts) so pyarrow doesn't struct-unify variable keys -> None values
        "copy_files_b64": json.dumps(copy_files),
        "test_files_b64": json.dumps(test_files),
        "instruction": instr_text,
        "max_turns": max_turns,
        "reward_spec": {"method": "env"},
        "extra_info": {"task_id": root.name, "source": source},
    }


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=4, memory=16384)
def build(limit: int = 40, tag: str = "smoke"):
    import os, io, tarfile, json, random
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    import pyarrow as pa, pyarrow.parquet as pq

    workdir = Path("/data/_raw"); workdir.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- Source 1: endless-terminals (per-task dirs) ----
    et_ok = 0
    et_root = workdir / "et"
    if limit == 0 or limit > 200:
        # BATCHED: list all task dirs, download in groups (one giant request times out)
        api = HfApi()
        tree = api.list_repo_tree("obiwan96/endless-terminals", repo_type="dataset", recursive=False)
        task_names = sorted([t.path for t in tree if t.path.startswith("task_")])
        if limit > 0:
            task_names = task_names[:limit]
        print(f"ET: {len(task_names)} task dirs to fetch (batched)")
        import time as _time
        B = 100
        for bi in range(0, len(task_names), B):
            batch = task_names[bi:bi + B]
            for attempt in range(5):
                try:
                    snapshot_download("obiwan96/endless-terminals", repo_type="dataset",
                                      allow_patterns=[f"{t}/*" for t in batch],
                                      local_dir=str(et_root), max_workers=4)
                    break
                except Exception as e:
                    wait = 5 * (attempt + 1)  # backoff for HTTP 429 rate limits
                    print(f"ET batch@{bi} attempt {attempt} failed ({repr(e)[:80]}); sleep {wait}s")
                    _time.sleep(wait)
            _time.sleep(2)  # gentle pacing between batches to avoid 429s
            for t in batch:
                try:
                    row = task_dir_to_row(Path(et_root) / t, "endless-terminals")
                    if row:
                        rows.append(row); et_ok += 1
                except Exception as e:
                    print("ET skip", t, repr(e)[:100])
            if bi % 600 == 0:
                print(f"ET progress: {et_ok} ok / {bi + len(batch)} listed")
    else:
        api = HfApi()
        tree = api.list_repo_tree("obiwan96/endless-terminals", repo_type="dataset", recursive=False)
        task_dirs = [t.path for t in tree if t.path.startswith("task_")]
        for tp in sorted(task_dirs)[:limit]:
            try:
                snapshot_download("obiwan96/endless-terminals", repo_type="dataset",
                                  allow_patterns=[f"{tp}/*"], local_dir=str(et_root))
                row = task_dir_to_row(Path(et_root) / tp, "endless-terminals")
                if row:
                    rows.append(row); et_ok += 1
            except Exception as e:
                print("ET skip", tp, repr(e)[:120])

    # ---- Source 2: OpenThoughts-Agent-v1-RL (tarballs in parquet) ----
    ot_ok = 0
    try:
        pqpath = hf_hub_download("open-thoughts/OpenThoughts-Agent-v1-RL", "tasks.parquet",
                                 repo_type="dataset", local_dir=str(workdir / "ot"))
        t = pq.read_table(pqpath).to_pylist()
        n_ot = len(t) if limit == 0 else min(limit, len(t))
        for rec in t[:n_ot]:
            try:
                tb = rec["task_binary"]
                dest = workdir / "ot_x" / rec["path"]
                dest.mkdir(parents=True, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(tb), mode="r:*") as tf:
                    tf.extractall(dest)
                # the tar may nest one dir; find the task root (has environment/Dockerfile)
                root = dest
                for cand in [dest] + list(dest.rglob("*")):
                    if (cand / "environment" / "Dockerfile").exists():
                        root = cand; break
                row = task_dir_to_row(root, "openthoughts-agent-rl")
                if row:
                    rows.append(row); ot_ok += 1
            except Exception as e:
                print("OT skip", rec.get("path"), repr(e)[:120])
    except Exception as e:
        print("OT source failed:", repr(e)[:200])

    # ---- split: hold out 100 (or 10% for small) for val ----
    random.Random(42).shuffle(rows)
    n_val = 100 if len(rows) > 300 else max(1, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]

    outdir = Path(f"/data/corpus/{tag}"); outdir.mkdir(parents=True, exist_ok=True)
    def write(split_rows, name):
        # pyarrow needs consistent schema; store dict/list cols as JSON-safe via pa.array of structs
        pq.write_table(pa.Table.from_pylist(split_rows), str(outdir / name))
    write(train, "train.parquet")
    write(val, "validation.parquet")
    vol.commit()
    summary = {"tag": tag, "endless_terminals_ok": et_ok, "openthoughts_ok": ot_ok,
               "total": len(rows), "train": len(train), "val": len(val),
               "out": str(outdir)}
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(limit: int = 40, tag: str = "smoke"):
    print(build.remote(limit=limit, tag=tag))
