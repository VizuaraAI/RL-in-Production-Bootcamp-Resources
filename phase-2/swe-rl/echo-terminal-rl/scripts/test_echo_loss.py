"""Hand-computed validation of the ECHO env-prediction loss math (eq 3, §3.1-3.3)."""
import torch
from echo_loss import compute_env_prediction_loss, echo_total_loss


def test_hand_computed_sequence_mean():
    # One sequence, T=8. Token roles:
    #  pos: 0    1    2    3       4        5        6      7
    #  A/A/O: prompt prompt action WARN(obs) ENV(obs) ENV(obs) action pad
    # O  = observation tokens (warnings + env)  = positions {3,4,5}   -> |O| = 3
    # O' = env tokens only (warnings excluded)  = positions {4,5}     -> |O'| = 2
    logprobs = torch.tensor([[-0.1, -0.2, -0.5, -3.0, -0.4, -0.6, -0.7, 0.0]])
    obs_total_mask   = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0]])   # O  = warns+env
    env_target_mask  = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0]])   # O' = env only

    env_loss, m = compute_env_prediction_loss(logprobs, env_target_mask, obs_total_mask,
                                              reduction="sequence_mean")
    # eq 3: -(1/|O|) * sum_{t in O'} logprob = -(1/3)*(-0.4 + -0.6) = 1.0/3 = 0.33333
    expected = (0.4 + 0.6) / 3.0
    assert abs(env_loss.item() - expected) < 1e-6, (env_loss.item(), expected)
    # per-token CE on O' (nats) = (0.4+0.6)/2 = 0.5  (Fig 6 style diagnostic)
    assert abs(m["env_ce_per_token_nats"] - 0.5) < 1e-6, m
    assert m["env_target_tokens"] == 2 and m["obs_total_tokens"] == 3
    print("PASS sequence_mean:", env_loss.item(), "== expected", round(expected, 6))


def test_warning_exclusion_matters():
    # The huge -3.0 NLL warning token at pos 3 must NOT enter the loss (§3.2 exclusion).
    logprobs = torch.tensor([[0., 0., 0., -3.0, -0.4, -0.6, 0., 0.]])
    obs_total_mask  = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0]])
    env_only        = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0]])
    warn_included   = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0]])
    l_env_only, _ = compute_env_prediction_loss(logprobs, env_only, obs_total_mask)
    l_warn, _     = compute_env_prediction_loss(logprobs, warn_included, obs_total_mask)
    # Including the warning inflates the loss by 3.0/|O| = 1.0. Excluding it is the point.
    assert l_warn.item() > l_env_only.item() + 0.9, (l_warn.item(), l_env_only.item())
    print("PASS warning-exclusion: env_only=%.4f  warn_incl=%.4f" % (l_env_only.item(), l_warn.item()))


def test_normalize_by_O_not_Oprime():
    # Two configs with the SAME env tokens but DIFFERENT O' subsets must stay comparable
    # because we divide by |O| (total obs), not |O'|. (§3.1 rationale.)
    logprobs = torch.tensor([[0., -0.4, -0.6, -0.8, 0.]])   # 3 env tokens at 1,2,3
    obs_total = torch.tensor([[0, 1, 1, 1, 0]])             # |O| = 3
    subset_2  = torch.tensor([[0, 1, 1, 0, 0]])             # O' = {1,2}
    subset_3  = torch.tensor([[0, 1, 1, 1, 0]])             # O' = {1,2,3}
    l2, _ = compute_env_prediction_loss(logprobs, subset_2, obs_total)
    l3, _ = compute_env_prediction_loss(logprobs, subset_3, obs_total)
    # both divide by |O|=3, so they are on the same per-observation scale
    assert abs(l2.item() - (0.4 + 0.6) / 3) < 1e-6
    assert abs(l3.item() - (0.4 + 0.6 + 0.8) / 3) < 1e-6
    print("PASS |O|-normalization: subset2=%.4f  subset3=%.4f (same /3 scale)" % (l2.item(), l3.item()))


def test_batch_and_echo_total():
    # Batch of 2 sequences; one has no observations (all-action rollout) -> must be skipped.
    logprobs = torch.tensor([
        [-0.1, -0.4, -0.6, 0.0],   # seq0: env at 1,2 ; |O|=2
        [-0.2, -0.3, -0.5, -0.7],  # seq1: no obs at all
    ])
    obs_total = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]])
    env_tgt   = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]])
    env_loss, m = compute_env_prediction_loss(logprobs, env_tgt, obs_total, reduction="sequence_mean")
    # only seq0 counts: (0.4+0.6)/2 = 0.5 ; averaged over 1 valid seq = 0.5
    assert abs(env_loss.item() - 0.5) < 1e-6, env_loss.item()
    assert abs(m["frac_seqs_with_obs"] - 0.5) < 1e-6
    # echo total = grpo + lambda*env
    grpo = torch.tensor(2.0)
    total, mm = echo_total_loss(grpo, logprobs, env_tgt, obs_total, lam=0.05)
    assert abs(total.item() - (2.0 + 0.05 * 0.5)) < 1e-6, total.item()
    print("PASS batch+echo_total: env=%.4f total=%.4f" % (env_loss.item(), total.item()))


def test_gradient_flows_to_env_positions_only():
    logprobs = torch.zeros(1, 6, requires_grad=True)
    obs_total = torch.tensor([[0, 0, 1, 1, 1, 0]])
    env_tgt   = torch.tensor([[0, 0, 0, 1, 1, 0]])   # O' = {3,4}
    env_loss, _ = compute_env_prediction_loss(logprobs, env_tgt, obs_total)
    env_loss.backward()
    g = logprobs.grad[0]
    nonzero = (g != 0).nonzero().flatten().tolist()
    assert nonzero == [3, 4], nonzero   # gradient ONLY at env-target tokens
    print("PASS gradient-localization: nonzero grad at", nonzero)


if __name__ == "__main__":
    test_hand_computed_sequence_mean()
    test_warning_exclusion_matters()
    test_normalize_by_O_not_Oprime()
    test_batch_and_echo_total()
    test_gradient_flows_to_env_positions_only()
    print("\nALL ECHO LOSS TESTS PASSED ✓")
