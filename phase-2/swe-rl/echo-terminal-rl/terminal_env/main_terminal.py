"""ECHO training entrypoint: register the `terminal` env, then run SkyRL's PPO/GRPO trainer.

Run (inside the SkyRL venv, terminal_env importable):
    python -m terminal_env.main_terminal <hydra-style overrides...>

The `terminal` env is a multi-turn bash-in-modal.Sandbox environment; used with
generator.use_conversation_multi_turn=true it puts observation tokens in the response span
(loss_mask=0), which is what ECHO's env-CE term trains on. Enable ECHO via
trainer.algorithm.echo.enabled=true trainer.algorithm.echo.lambda_=0.05.
"""
import sys
import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # register inside the entrypoint task so it's available to the driver + workers
    register(id="terminal", entry_point="terminal_env.terminal_env:TerminalEnv")
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
