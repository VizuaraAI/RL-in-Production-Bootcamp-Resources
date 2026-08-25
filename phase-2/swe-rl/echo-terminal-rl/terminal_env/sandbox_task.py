"""Harbor-task rollout+verifier cycle on modal.Sandbox — SCALABLE (no per-task image builds).

Per-task `modal.Image.from_dockerfile` does not work at rollout time (context is read at
image-load, which happens inside the remote worker where the local Dockerfile is gone), and
pre-building thousands of task images doesn't scale. Instead we use ONE static base image and
REPLAY each task's Dockerfile setup (RUN/COPY/ENV/WORKDIR) inside the sandbox. Most
endless-terminals / OpenThoughts tasks are `FROM ubuntu:22.04` + simple setup, so this
reproduces the initial state faithfully and fast, with zero per-task builds.

    modal run terminal_env/sandbox_task.py   # reference solution -> reward 1 ; empty -> reward 0
"""
import modal
import re
from pathlib import Path

app = modal.App("echo-terminal-task-test")

# ONE static base image, built once. Superset of what typical tasks need so setup replay is fast.
BASE_IMAGE = (
    modal.Image.from_registry("ubuntu:22.04", add_python=None)
    .apt_install("python3", "python3-pip", "curl", "git", "ca-certificates", "coreutils",
                 "build-essential", "wget", "unzip", "jq", "tree")
    .run_commands("pip3 install --no-cache-dir pytest || true")
)

TASK = Path(__file__).parent.parent / "sample_tasks" / "et_0033979a"


# ---- Dockerfile setup parser (extracts the ops that build initial state) ----
def parse_dockerfile_setup(dockerfile_text: str):
    """Return an ordered list of setup ops: ('run', cmd) | ('copy', src, dst) | ('env', k, v)
    | ('workdir', path). Handles `\\` line-continuations and heredocs inside RUN. Skips
    FROM/LABEL/CMD/ENTRYPOINT/EXPOSE/comments."""
    ops = []
    lines = dockerfile_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        m = re.match(r"^(\w+)\s+(.*)$", stripped, re.DOTALL)
        if not m:
            i += 1
            continue
        instr = m.group(1).upper()
        rest = raw[raw.upper().find(instr) + len(instr):].lstrip()
        # gather line-continuations (\) — but a heredoc <<'EOF' pulls raw lines until its terminator
        body_lines = [rest]
        # detect heredoc terminator(s) in this instruction
        def heredoc_terms(s):
            return re.findall(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", s)
        pending_heredocs = heredoc_terms(rest)
        cont = rest.rstrip().endswith("\\")
        while (cont or pending_heredocs) and i + 1 < len(lines):
            i += 1
            nxt = lines[i]
            body_lines.append(nxt)
            if pending_heredocs:
                if nxt.strip() == pending_heredocs[0]:
                    pending_heredocs.pop(0)
                cont = (not pending_heredocs) and nxt.rstrip().endswith("\\")
            else:
                cont = nxt.rstrip().endswith("\\")
        body = "\n".join(body_lines)
        if instr == "RUN":
            ops.append(("run", body))
        elif instr == "COPY" or instr == "ADD":
            parts = body.split()
            parts = [p for p in parts if not p.startswith("--")]
            if len(parts) >= 2:
                ops.append(("copy", parts[0], parts[-1]))
        elif instr == "ENV":
            envm = re.match(r"(\S+)[=\s]+(.*)", body)
            if envm:
                ops.append(("env", envm.group(1), envm.group(2).strip().strip('"')))
        elif instr == "WORKDIR":
            ops.append(("workdir", body.strip()))
        # FROM/LABEL/CMD/ENTRYPOINT/EXPOSE/ARG/USER/HEALTHCHECK -> skip
        i += 1
    return ops


def apply_setup(sb: modal.Sandbox, ops, copy_files: dict):
    """Replay setup ops in the sandbox. copy_files: {src_rel: bytes} for COPY sources."""
    workdir = "/"
    env_prefix = ""
    warnings = []
    for op in ops:
        if op[0] == "workdir":
            workdir = op[1]
            sb.exec("bash", "-c", f"mkdir -p {workdir}").wait()
        elif op[0] == "env":
            env_prefix += f"export {op[1]}={op[2]!r}; "
        elif op[0] == "copy":
            _, src, dst = op
            data = copy_files.get(src) or copy_files.get(src.lstrip("./"))
            if data is None:
                warnings.append(f"COPY source not staged: {src}")
                continue
            # if dst ends with / it's a dir; else a file (basename may differ)
            target = dst if not dst.endswith("/") else f"{dst}/{Path(src).name}"
            sb.exec("bash", "-c", f"mkdir -p $(dirname {target})").wait()
            with sb.open(target, "wb") as fh:
                fh.write(data)
        elif op[0] == "run":
            script = f"{env_prefix}cd {workdir} 2>/dev/null; {op[1]}"
            p = sb.exec("bash", "-c", script)
            p.stdout.read(); p.stderr.read(); p.wait()
            if p.returncode != 0:
                warnings.append(f"setup RUN rc={p.returncode}: {op[1][:80]}")
    return warnings


def _stage_files(sb: modal.Sandbox, files: dict, remote_dir: str):
    sb.exec("bash", "-c", f"mkdir -p {remote_dir}").wait()
    for rel, data in files.items():
        remote = f"{remote_dir}/{rel}"
        sb.exec("bash", "-c", f"mkdir -p $(dirname {remote})").wait()
        with sb.open(remote, "wb") as fh:
            fh.write(data)


def run_task(dockerfile_text: str, copy_files: dict, agent_script: str,
             test_files: dict, timeout: int = 900) -> dict:
    sb = modal.Sandbox.create(image=BASE_IMAGE, app=app, timeout=timeout)
    log = {}
    try:
        ops = parse_dockerfile_setup(dockerfile_text)
        log["setup_warnings"] = apply_setup(sb, ops, copy_files)
        # agent acts
        p = sb.exec("bash", "-lc", agent_script)
        log["agent_stdout"] = p.stdout.read()[-1200:]; p.stderr.read(); p.wait()
        log["agent_rc"] = p.returncode
        # verify
        _stage_files(sb, test_files, "/tests")
        sb.exec("bash", "-c", "mkdir -p /logs/verifier").wait()
        v = sb.exec("bash", "-lc", "bash /tests/test.sh")
        log["verifier_stdout"] = v.stdout.read()[-1200:]; v.wait()
        log["verifier_rc"] = v.returncode
        r = sb.exec("bash", "-c", "cat /logs/verifier/reward.txt 2>/dev/null || echo MISSING")
        rr = r.stdout.read().strip(); r.wait()
        log["reward_raw"] = rr
        log["reward"] = 1.0 if rr == "1" else 0.0
    finally:
        sb.terminate()
    return log


@app.function(timeout=1800)
def validate(dockerfile_text: str, copy_files: dict, solve_script: str, test_files: dict):
    import json
    solved = run_task(dockerfile_text, copy_files, solve_script, test_files)
    empty = run_task(dockerfile_text, copy_files, "echo no-op", test_files)
    return json.dumps({
        "solved_reward": solved["reward"], "empty_reward": empty["reward"],
        "solved_verifier_rc": solved["verifier_rc"], "solved_agent_rc": solved["agent_rc"],
        "solved_reward_raw": solved["reward_raw"], "empty_reward_raw": empty["reward_raw"],
        "setup_warnings": solved["setup_warnings"],
        "solved_verifier_tail": solved["verifier_stdout"][-400:],
    }, indent=2)


@app.local_entrypoint()
def main():
    env_dir = TASK / "environment"
    dockerfile = (env_dir / "Dockerfile").read_text()
    copy_files = {str(f.relative_to(env_dir)): f.read_bytes()
                  for f in env_dir.rglob("*") if f.is_file() and f.name != "Dockerfile"}
    solve = (TASK / "solution" / "solve.sh").read_text()
    tests_dir = TASK / "tests"
    test_files = {str(f.relative_to(tests_dir)): f.read_bytes()
                  for f in tests_dir.rglob("*") if f.is_file()}
    print(validate.remote(dockerfile, copy_files, solve, test_files))
