"""Prove the in-worker ECHO math (worker.py RL branch) equals the unit-tested kernel.

Reproduces EXACTLY the tensor ops added to `_forward_backward_micro`, deriving obs_mask
from the same (action_mask, loss_mask) the worker sees, and checks it against
`compute_env_prediction_loss` (which is validated in test_echo_loss.py).
"""
import torch
from echo_loss import compute_env_prediction_loss


def worker_inline_env_ce(action_log_probs, action_mask, loss_mask, env_output_mask=None, exclude_warn=False):
    """Verbatim copy of the math inserted into worker.py's RL branch."""
    obs_total_mask = ((action_mask > 0) & (loss_mask == 0)).to(action_log_probs.dtype)  # O
    obs_target_mask = obs_total_mask                                                     # O'
    if exclude_warn and env_output_mask is not None:
        obs_target_mask = obs_total_mask * env_output_mask.to(action_log_probs.dtype)
    seq_obs_len = obs_total_mask.sum(dim=-1)
    seq_nll = (-action_log_probs * obs_target_mask).sum(dim=-1)
    valid = seq_obs_len > 0
    env_ce = torch.tensor(0.0)
    if bool(valid.any()):
        seq_env = torch.where(valid, seq_nll / seq_obs_len.clamp(min=1.0), torch.zeros_like(seq_nll))
        env_ce = seq_env.sum() / valid.sum().clamp(min=1)
    return env_ce, obs_total_mask, obs_target_mask


def test_equiv_no_warning_exclusion():
    B, T = 3, 12
    torch.manual_seed(0)
    alp = torch.randn(B, T)
    # response span = last 9 tokens; within it: action/obs interleave; first 3 are prompt(pad)
    action_mask = torch.tensor([[0,0,0,1,1,1,1,1,1,1,1,1]]*B)          # response_mask (all resp tokens)
    loss_mask   = torch.tensor([[0,0,0,1,1,0,0,1,1,0,0,0]]*B)          # 1=action; 0 on obs (5,6,9,10,11)
    env_ce, obs_total, obs_tgt = worker_inline_env_ce(alp, action_mask, loss_mask)
    # kernel reference: env_target = obs_tgt, obs_total = obs_total
    ref, _ = compute_env_prediction_loss(alp, obs_tgt, obs_total, reduction="sequence_mean")
    assert torch.allclose(env_ce, ref, atol=1e-6), (env_ce.item(), ref.item())
    # obs tokens per sequence = positions {5,6,9,10,11} = 5
    assert int(obs_total[0].sum()) == 5
    print("PASS worker≡kernel (no warn-excl): env_ce=%.5f  obs/seq=%d" % (env_ce.item(), int(obs_total[0].sum())))


def test_equiv_with_warning_exclusion():
    B, T = 2, 10
    alp = torch.randn(B, T)
    action_mask = torch.tensor([[0,0,1,1,1,1,1,1,1,1]]*B)
    loss_mask   = torch.tensor([[0,0,1,1,0,0,0,0,1,1]]*B)     # obs = {4,5,6,7}
    # env_output_mask marks the terminal-output sub-span; say {4,5} are warnings, {6,7} real env
    env_output_mask = torch.tensor([[0,0,0,0,0,0,1,1,0,0]]*B)
    env_ce, obs_total, obs_tgt = worker_inline_env_ce(alp, action_mask, loss_mask,
                                                      env_output_mask=env_output_mask, exclude_warn=True)
    ref, _ = compute_env_prediction_loss(alp, obs_tgt, obs_total, reduction="sequence_mean")
    assert torch.allclose(env_ce, ref, atol=1e-6)
    # O' = {6,7} (2 tokens), but normalized by |O|=4 (the paper's rule)
    assert int(obs_tgt[0].sum()) == 2 and int(obs_total[0].sum()) == 4
    print("PASS worker≡kernel (warn-excl): |O'|=2 normalized by |O|=4, env_ce=%.5f" % env_ce.item())


def test_stepwise_mode_is_inert():
    # In step-wise mode obs tokens are NOT in the response span: response_mask==loss_mask.
    # ECHO must then be a no-op (obs_mask empty) — we detect this via obs_tokens==0.
    alp = torch.randn(1, 8)
    action_mask = torch.tensor([[0,0,1,1,1,1,1,1]])
    loss_mask   = torch.tensor([[0,0,1,1,1,1,1,1]])   # identical -> no obs tokens
    env_ce, obs_total, _ = worker_inline_env_ce(alp, action_mask, loss_mask)
    assert int(obs_total.sum()) == 0 and env_ce.item() == 0.0
    print("PASS step-wise detection: obs_tokens=0 -> ECHO inert (as designed; watch echo/obs_tokens)")


if __name__ == "__main__":
    test_equiv_no_warning_exclusion()
    test_equiv_with_warning_exclusion()
    test_stepwise_mode_is_inert()
    print("\nWORKER↔KERNEL EQUIVALENCE VERIFIED ✓")
