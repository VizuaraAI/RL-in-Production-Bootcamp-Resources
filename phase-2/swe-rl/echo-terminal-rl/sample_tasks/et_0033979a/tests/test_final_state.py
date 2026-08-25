# test_final_state.py
#
# Pytest suite that validates the **FINAL** state of the operating
# system after the hardening task has been performed.
#
# What we validate:
# 1. /home/user/secure_config/ exists and is a directory.
# 2. /home/user/secure_config/sshd_config_hardened exists, is a regular
#    file, has exact expected contents, and has permissions 600
#    (-rw-------).
# 3. /home/user/hardening.log exists and its *last* (and only) line
#    matches the required pipe-delimited audit-entry format:
#       YYYY-MM-DD_HH:MM:SS|UPDATED|/home/user/secure_config/sshd_config_hardened
#    where the timestamp component is 24-hour UTC time.
#
# If any check fails, the test will emit a clear, actionable message so
# that the student can quickly identify what is still wrong.

import os
import re
import stat
from pathlib import Path

SECURE_DIR = Path("/home/user/secure_config")
HARDENED_FILE = SECURE_DIR / "sshd_config_hardened"
LOG_FILE = Path("/home/user/hardening.log")

EXPECTED_HARDENED_LINES = [
    "# HARDENED BY AUTOMATION",
    "PermitRootLogin no",
    "PasswordAuthentication no",
]

AUDIT_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}\|UPDATED\|/home/user/secure_config/sshd_config_hardened$"
)


def test_secure_directory_exists():
    """The hardened configuration directory must exist."""
    assert SECURE_DIR.exists(), f"Directory {SECURE_DIR} is missing."
    assert SECURE_DIR.is_dir(), f"{SECURE_DIR} exists but is not a directory."


def test_hardened_file_exists_and_is_regular_file():
    """The hardened sshd_config_hardened file must exist and be a regular file."""
    assert HARDENED_FILE.exists(), f"File {HARDENED_FILE} is missing."
    assert HARDENED_FILE.is_file(), f"{HARDENED_FILE} exists but is not a regular file."


def test_hardened_file_contents_and_trailing_newline():
    """
    The hardened config file must contain exactly the three expected lines
    (in order) and end with one—and only one—trailing newline.
    """
    raw_content = HARDENED_FILE.read_text()
    assert raw_content.endswith(
        "\n"
    ), f"{HARDENED_FILE} must end with a single trailing newline."

    # Reject extra blank lines at the end (e.g., '\n\n')
    assert not raw_content.endswith(
        "\n\n"
    ), f"{HARDENED_FILE} contains extra blank lines at the end."

    lines = raw_content.splitlines()
    assert (
        lines == EXPECTED_HARDENED_LINES
    ), f"Unexpected content in {HARDENED_FILE}.\nExpected:\n" + "\n".join(
        f"    {l}" for l in EXPECTED_HARDENED_LINES
    ) + "\nGot:\n" + "\n".join(f"    {l}" for l in lines)


def test_hardened_file_permissions():
    """The hardened file must have permissions 600 (-rw-------)."""
    mode = HARDENED_FILE.stat().st_mode
    perms = stat.S_IMODE(mode)
    assert (
        perms == 0o600
    ), f"{HARDENED_FILE} permissions are {oct(perms)}; expected 0o600 (-rw-------)."


def test_hardening_log():
    """
    /home/user/hardening.log must exist and contain exactly one line that
    matches the required audit-entry regex.
    """
    assert LOG_FILE.exists(), f"Log file {LOG_FILE} is missing."
    assert LOG_FILE.is_file(), f"{LOG_FILE} exists but is not a regular file."

    # Read all non-empty lines to detect extra blank lines
    lines = [ln.rstrip("\n") for ln in LOG_FILE.read_text().splitlines() if ln.strip() != ""]
    assert (
        len(lines) == 1
    ), f"{LOG_FILE} should contain exactly one non-empty line; found {len(lines)}."

    last_line = lines[-1]
    assert AUDIT_REGEX.match(
        last_line
    ), f"The log entry in {LOG_FILE} does not match the required format.\nGot:\n    {last_line}\nExpected pattern:\n    {AUDIT_REGEX.pattern}"