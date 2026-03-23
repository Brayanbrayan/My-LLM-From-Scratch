"""dpo_loss.py — Direct Preference Optimisation loss for Part 10.

Loss:
    L_DPO = -E[ log σ( β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                       - β * (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

References
----------
Rafailov et al. 2023 "Direct Preference Optimization: Your Language Model is
Secretly a Reward Model"  https://arxiv.org/abs/2305.18290
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class DPOLossOutput:
    """Container for DPO loss and associated diagnostics."""
    loss: torch.Tensor          # scalar — the main training objective
    reward_margin: torch.Tensor # scalar — mean(chosen_ratio - rejected_ratio)
    accuracy: torch.Tensor      # scalar — fraction of samples where chosen > rejected


def dpo_loss(
    policy_chosen_logps: torch.Tensor,   # (B,) — log π_θ(y_w | x)
    policy_rejected_logps: torch.Tensor, # (B,) — log π_θ(y_l | x)
    ref_chosen_logps: torch.Tensor,      # (B,) — log π_ref(y_w | x)
    ref_rejected_logps: torch.Tensor,    # (B,) — log π_ref(y_l | x)
    beta: float = 0.1,
) -> DPOLossOutput:
    """Compute the DPO loss and auxiliary metrics.

    Parameters
    ----------
    policy_chosen_logps:   per-sequence sum log-prob of chosen response under policy
    policy_rejected_logps: per-sequence sum log-prob of rejected response under policy
    ref_chosen_logps:      per-sequence sum log-prob of chosen response under reference
    ref_rejected_logps:    per-sequence sum log-prob of rejected response under reference
    beta:                  temperature controlling how tightly we stay near the reference

    Returns
    -------
    DPOLossOutput with .loss, .reward_margin, .accuracy
    """
    # Log-ratios: log(π_θ / π_ref) for each response type
    chosen_log_ratios   = policy_chosen_logps   - ref_chosen_logps    # (B,)
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps  # (B,)

    # DPO margin: β * (log-ratio_chosen - log-ratio_rejected)
    # Positive margin → model assigns relatively higher prob to chosen than ref does
    logits = beta * (chosen_log_ratios - rejected_log_ratios)         # (B,)

    # Main loss: -log σ(margin), averaged over batch
    loss = -F.logsigmoid(logits).mean()

    # ---- Auxiliary metrics (detached — not used for gradients) ----

    # Reward margin: how much larger the chosen log-ratio is than the rejected one
    # Mirrors the "implicit reward" the DPO policy assigns to each response.
    # A positive, growing margin means the policy is learning the preference.
    reward_margin = (chosen_log_ratios - rejected_log_ratios).detach().mean()

    # Accuracy: fraction of samples where the model correctly prefers chosen over rejected
    # i.e. the margin is positive (model assigns higher implicit reward to chosen)
    accuracy = (logits.detach() > 0).float().mean()

    return DPOLossOutput(loss=loss, reward_margin=reward_margin, accuracy=accuracy)


# ---------------------------------------------------------------------------
# Sequence log-probability helper
# ---------------------------------------------------------------------------

def get_logps(
    model,
    input_ids: torch.Tensor,   # (B, T) — full sequence (prompt + response)
    response_mask: torch.Tensor,# (B, T) — True for response tokens only
) -> torch.Tensor:
    """Return per-sequence sum of log-probs over the response tokens only.

    Mirrors the shift logic in train_ppo.py:
        - logits are taken at positions [0 .. T-2]  (predicting next token)
        - labels  are taken at positions [1 .. T-1]  (the actual next tokens)
        - response_mask is also shifted by 1 so it aligns with the labels

    Parameters
    ----------
    model         : PolicyWithValue — must expose .lm(x, None) -> (logits, loss, _)
    input_ids     : (B, T) long tensor — the full token sequence
    response_mask : (B, T) bool tensor — True at response token positions

    Returns
    -------
    (B,) float tensor — sum of log-probs for each sequence's response
    """
    # Forward pass — we only need the logits
    logits, _, _ = model.lm(input_ids, None)  # (B, T, V)

    # Shift: predict position t+1 from position t
    # logits[:, :-1, :] predicts tokens 1..T-1
    # labels [:, 1:]    are tokens 1..T-1
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()   # (B, T-1)

    # Shift the mask the same way so it aligns with shift_labels
    # response_mask[:, 1:] marks which of the shifted positions are response tokens
    shift_mask = response_mask[:, 1:].contiguous() # (B, T-1)

    # Per-token log-probs
    log_probs = torch.log_softmax(shift_logits, dim=-1)          # (B, T-1, V)
    token_logps = log_probs.gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)                                                  # (B, T-1)

    # Zero out non-response positions and sum over response tokens
    token_logps = token_logps * shift_mask.float()                # (B, T-1)
    return token_logps.sum(dim=-1)                                 # (B,)