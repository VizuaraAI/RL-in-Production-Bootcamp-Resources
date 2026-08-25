"""TerminalSession — framework-free terminal-task rollout on modal.Sandbox.

Owns one episode's sandbox: boots it (base image + Dockerfile-setup replay), execs agent bash
turn-by-turn, and runs the verifier for the binary reward. No SkyRL dependency, so it can be
unit-tested locally (creating real sandboxes from the driver). `TerminalEnv` (the SkyRL
BaseTextEnv adapter) is a thin wrapper over this.

Action format the agent is instructed to use (see SYSTEM_PROMPT):
  - a single bash command in a ```bash ... ``` fence  OR  <bash> ... </bash>
  - a completion signal <task_complete> (or "TASK_COMPLETE") when done
Unparseable actions yield a format WARNING observation (the paper's warning-prefix; §3.2).
"""
import modal
import re
from pathlib import Path

APP_NAME = "echo-terminal-rollout"

# ONE static base image (built once) — superset of typical task needs so setup-replay is fast.
BASE_IMAGE = (
    modal.Image.from_registry("ubuntu:22.04", add_python=None)
    .apt_install("python3", "python3-pip", "curl", "git", "ca-certificates", "coreutils",
                 "build-essential", "wget", "unzip", "jq", "tree", "bc")
    .run_commands("pip3 install --no-cache-dir pytest || true")
)

SYSTEM_PROMPT = """You are a terminal agent. You solve the task by issuing shell commands \
in a Linux container. On each turn, first think briefly, then issue EXACTLY ONE bash command \
inside a fenced code block:

```bash
<your command here>
```

You will then see the command's output (stdout, stderr, and exit code). Continue issuing one \
command per turn until the task is complete. When you are confident the task is fully done, \
respond with:

<task_complete>

Do not issue more than one command per turn. Keep commands non-interactive."""

# markers
_TASK_DONE = re.compile(r"<task_complete>|TASK_COMPLETE|<done>", re.IGNORECASE)
_FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)
_TAG = re.compile(r"<(?:bash|execute_bash|command)>(.*?)</(?:bash|execute_bash|command)>", re.DOTALL)

MAX_OBS_CHARS = 4000  # truncate long terminal output (keeps context bounded, like real harnesses)


def parse_action(action: str):
    """Return (command_or_None, done_bool, parse_ok_bool)."""
    done = bool(_TASK_DONE.search(action))
    cmd = None
    m = _FENCE.search(action) or _TAG.search(action)
    if m:
        cmd = m.group(1).strip()
    if cmd:
        return cmd, done, True
    if done:
        return None, True, True
    # no command and no done signal -> unparseable (format warning)
    return None, False, False


# ---- Dockerfile setup replay (validated in sandbox_task.py) ----
def parse_dockerfile_setup(dockerfile_text: str):
    ops, lines, i = [], dockerfile_text.splitlines(), 0
    def heredoc_terms(s):
        return re.findall(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", s)
    while i < len(lines):
        raw = lines[i]; stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1; continue
        m = re.match(r"^(\w+)\s+(.*)$", stripped, re.DOTALL)
        if not m:
            i += 1; continue
        instr = m.group(1).upper()
        rest = raw[raw.upper().find(instr) + len(instr):].lstrip()
        body_lines = [rest]
        pending = heredoc_terms(rest)
        cont = rest.rstrip().endswith("\\")
        while (cont or pending) and i + 1 < len(lines):
            i += 1; nxt = lines[i]; body_lines.append(nxt)
            if pending:
                if nxt.strip() == pending[0]:
                    pending.pop(0)
                cont = (not pending) and nxt.rstrip().endswith("\\")
            else:
                cont = nxt.rstrip().endswith("\\")
        body = "\n".join(body_lines)
        if instr == "RUN":
            ops.append(("run", body))
        elif instr in ("COPY", "ADD"):
            parts = [p for p in body.split() if not p.startswith("--")]
            if len(parts) >= 2:
                ops.append(("copy", parts[0], parts[-1]))
        elif instr == "ENV":
            em = re.match(r"(\S+)[=\s]+(.*)", body)
            if em:
                ops.append(("env", em.group(1), em.group(2).strip().strip('"')))
        elif instr == "WORKDIR":
            ops.append(("workdir", body.strip()))
        i += 1
    return ops


class TerminalSession:
    def __init__(self, dockerfile_text: str, copy_files: dict, test_files: dict,
                 instruction: str, max_turns: int = 16, timeout: int = 1200,
                 app=None, base_image=None):
        self.dockerfile_text = dockerfile_text or "FROM ubuntu:22.04\n"
        self.copy_files = copy_files or {}
        self.test_files = test_files or {}
        self.instruction = instruction
        self.max_turns = max_turns
        self.timeout = timeout
        self._app = app
        self._base_image = base_image or BASE_IMAGE
        self.sb = None
        self.turns = 0
        self.setup_warnings = []

    # -- lifecycle --
    def start(self):
        app = self._app or modal.App.lookup(APP_NAME, create_if_missing=True)
        self.sb = modal.Sandbox.create(image=self._base_image, app=app, timeout=self.timeout)
        self._replay_setup()
        return self

    def _write_file(self, remote: str, data: bytes):
        self.sb.exec("bash", "-c", f"mkdir -p $(dirname {remote})").wait()
        # filesystem API is the supported (non-deprecated) path
        try:
            self.sb.filesystem.write_bytes(remote, data)
        except Exception:
            with self.sb.open(remote, "wb") as fh:
                fh.write(data)

    def _replay_setup(self):
        ops = parse_dockerfile_setup(self.dockerfile_text)
        workdir, env_prefix = "/", ""
        for op in ops:
            if op[0] == "workdir":
                workdir = op[1]
                self.sb.exec("bash", "-c", f"mkdir -p {workdir}").wait()
            elif op[0] == "env":
                env_prefix += f"export {op[1]}={op[2]!r}; "
            elif op[0] == "copy":
                _, src, dst = op
                data = self.copy_files.get(src) or self.copy_files.get(src.lstrip("./"))
                if data is None:
                    self.setup_warnings.append(f"COPY src not staged: {src}"); continue
                target = dst if not dst.endswith("/") else f"{dst}/{Path(src).name}"
                self._write_file(target, data)
            elif op[0] == "run":
                p = self.sb.exec("bash", "-c", f"{env_prefix}cd {workdir} 2>/dev/null; {op[1]}")
                p.stdout.read(); p.stderr.read(); p.wait()
                if p.returncode != 0:
                    self.setup_warnings.append(f"setup rc={p.returncode}: {op[1][:60]}")

    # -- turn --
    def step(self, action: str):
        """Execute one agent action. Returns (observation_str, done_bool, is_warning_bool)."""
        self.turns += 1
        cmd, done, ok = parse_action(action)
        if done and cmd is None:
            return "", True, False
        if not ok:
            warn = ("WARNING: no runnable command found. Emit exactly one command inside a "
                    "```bash ... ``` fence, or <task_complete> when finished.")
            done = self.turns >= self.max_turns
            return warn, done, True
        p = self.sb.exec("bash", "-lc", cmd)
        out = p.stdout.read(); err = p.stderr.read(); p.wait()
        rc = p.returncode
        body = out + (("\n[stderr]\n" + err) if err.strip() else "")
        if len(body) > MAX_OBS_CHARS:
            body = body[:MAX_OBS_CHARS // 2] + "\n...[truncated]...\n" + body[-MAX_OBS_CHARS // 2:]
        # wrap real terminal output so a faithful O' mask can exclude warnings later (§3.2)
        obs = f"<command_output exit_code={rc}>\n{body}\n</command_output>"
        done = done or (self.turns >= self.max_turns)
        return obs, done, False

    # -- reward --
    def verify(self) -> float:
        if self.sb is None:
            return 0.0
        for rel, data in self.test_files.items():
            self._write_file(f"/tests/{rel}", data)
        self.sb.exec("bash", "-c", "mkdir -p /logs/verifier").wait()
        v = self.sb.exec("bash", "-lc", "bash /tests/test.sh")
        v.stdout.read(); v.wait()
        r = self.sb.exec("bash", "-c", "cat /logs/verifier/reward.txt 2>/dev/null || echo 0")
        rr = r.stdout.read().strip(); r.wait()
        return 1.0 if rr == "1" else 0.0

    def close(self):
        if self.sb is not None:
            try:
                self.sb.terminate()
            except Exception:
                pass
            self.sb = None
