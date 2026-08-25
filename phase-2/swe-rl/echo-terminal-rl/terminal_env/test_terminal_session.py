"""Live test of TerminalSession: parser asserts + a full multi-turn episode on a real sandbox.

Run locally (creates a real modal.Sandbox from the driver):
    python terminal_env/test_terminal_session.py
"""
from pathlib import Path
from terminal_session import parse_action, TerminalSession

TASK = Path(__file__).parent.parent / "sample_tasks" / "et_0033979a"


def test_parser():
    cmd, done, ok = parse_action("thinking...\n```bash\nls -la /home\n```")
    assert cmd == "ls -la /home" and not done and ok, (cmd, done, ok)
    cmd, done, ok = parse_action("I'll run <bash>echo hi</bash> now")
    assert cmd == "echo hi" and ok, (cmd, done, ok)
    cmd, done, ok = parse_action("All finished. <task_complete>")
    assert cmd is None and done and ok, (cmd, done, ok)
    cmd, done, ok = parse_action("no command here, just musing")
    assert cmd is None and not done and not ok, (cmd, done, ok)
    print("PASS parser (fence / tag / done / warning)")


def test_live_episode():
    env_dir = TASK / "environment"
    dockerfile = (env_dir / "Dockerfile").read_text()
    copy_files = {str(f.relative_to(env_dir)): f.read_bytes()
                  for f in env_dir.rglob("*") if f.is_file() and f.name != "Dockerfile"}
    tests_dir = TASK / "tests"
    test_files = {str(f.relative_to(tests_dir)): f.read_bytes()
                  for f in tests_dir.rglob("*") if f.is_file()}
    instruction = (TASK / "instruction.md").read_text()

    sess = TerminalSession(dockerfile, copy_files, test_files, instruction, max_turns=16)
    print("booting sandbox + replaying setup...")
    sess.start()
    print("setup_warnings:", sess.setup_warnings)

    # Turn 1: inspect initial state (proves setup replay produced it)
    obs, done, warn = sess.step("Let me look.\n```bash\ncat /home/user/insecure_config/sshd_config\n```")
    print("T1 obs:", obs[:200].replace("\n", " "))
    assert "PermitRootLogin yes" in obs, "initial state not reproduced!"
    assert not done and not warn

    # Turn 2: unparseable action -> format warning
    obs, done, warn = sess.step("hmm let me think about this")
    assert warn and "WARNING" in obs, obs
    print("T2 warning path OK")

    # Turn 3: solve the task in one command
    solve = ('mkdir -p /home/user/secure_config && '
             'printf "%s\\n" "# HARDENED BY AUTOMATION" "PermitRootLogin no" "PasswordAuthentication no" '
             '> /home/user/secure_config/sshd_config_hardened && '
             'chmod 600 /home/user/secure_config/sshd_config_hardened && '
             'printf "%s|UPDATED|/home/user/secure_config/sshd_config_hardened\\n" '
             '"$(date -u \'+%F_%H:%M:%S\')" >> /home/user/hardening.log')
    obs, done, warn = sess.step(f"Now I solve it.\n```bash\n{solve}\n```")
    print("T3 exit in obs:", "exit_code=0" in obs)
    assert "exit_code=0" in obs

    # Turn 4: signal completion
    obs, done, warn = sess.step("Task done. <task_complete>")
    assert done, "task_complete did not end episode"

    reward = sess.verify()
    print("REWARD:", reward)
    sess.close()
    assert reward == 1.0, f"expected reward 1.0, got {reward}"
    print("PASS live episode: multi-turn rollout + verifier -> reward 1.0")


if __name__ == "__main__":
    test_parser()
    test_live_episode()
    print("\nTERMINAL SESSION VALIDATED ✓")
