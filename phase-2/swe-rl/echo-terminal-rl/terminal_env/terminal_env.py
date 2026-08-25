"""TerminalEnv — SkyRL BaseTextEnv adapter over TerminalSession.

Registered as env id `terminal`. The multi-turn SkyRLGymGenerator (use_conversation_multi_turn=
true) tokenizes each observation into the response span with loss_mask=0 — exactly the in-span
structure ECHO's env-CE loss needs.

Per-task data arrives via `extras` (the dataset row's env_extras):
  extras = {
    "dockerfile":     <str>,                     # task environment/Dockerfile text
    "copy_files_b64": {relpath: base64str},       # environment/ files referenced by COPY
    "test_files_b64": {relpath: base64str},       # tests/ (incl test.sh) for the verifier
    "instruction":    <str>,                      # (already in the prompt; kept for reference)
    "max_turns":      16,
    "timeout":        1200,
  }
Reward is 0 on intermediate turns and the binary verifier result on the final turn (paper: 1 iff
final unit tests pass).
"""
import base64
import json
from typing import Any, Dict, Union
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
try:
    from .terminal_session import TerminalSession   # when imported as the installed package
except ImportError:  # pragma: no cover - allows local `python test_*.py` from inside the dir
    from terminal_session import TerminalSession


def _decode(files_b64) -> Dict[str, bytes]:
    """Accept a JSON string {relpath: b64str} (parquet-safe) or a dict; return {relpath: bytes}.
    Skips None values defensively (pyarrow struct-unification can inject them)."""
    if isinstance(files_b64, str):
        files_b64 = json.loads(files_b64) if files_b64 else {}
    return {k: base64.b64decode(v) for k, v in (files_b64 or {}).items() if v is not None}


class TerminalEnv(BaseTextEnv):
    def __init__(self, env_config: Union[DictConfig, dict, None] = None, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = int(extras.get("max_turns", 16))
        self.session = TerminalSession(
            dockerfile_text=extras.get("dockerfile", "FROM ubuntu:22.04\n"),
            copy_files=_decode(extras.get("copy_files_b64", {})),
            test_files=_decode(extras.get("test_files_b64", {})),
            instruction=extras.get("instruction", ""),
            max_turns=self.max_turns,
            timeout=int(extras.get("timeout", 1200)),
        )
        self._started = False
        self._scored = False

    def init(self, prompt: ConversationType):
        # boot the sandbox + replay initial state at episode start
        self.session.start()
        self._started = True
        return prompt, {}

    def _ensure_started(self):
        if not self._started:
            self.session.start()
            self._started = True

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self._ensure_started()
        obs, done, is_warning = self.session.step(action)
        reward = 0.0
        if done and not self._scored:
            reward = self.session.verify()
            self._scored = True
        observations: ConversationType = [{"role": "user", "content": obs}] if obs else []
        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={"is_warning": is_warning, "turns": self.session.turns},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {"turns": self.session.turns, "setup_warnings": len(self.session.setup_warnings)}

    def close(self):
        self.session.close()
