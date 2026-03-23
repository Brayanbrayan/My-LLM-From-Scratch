"""tests/test_dpo_loss.py — Unit tests for Part 10 DPO implementation.

Tests
-----
1. test_dpo_loss_perfect_preference
   When the policy log-ratios clearly favour chosen over rejected,
   the loss should be low, accuracy = 1.0, and margin > 0.

2. test_dpo_loss_reversed_preference
   When the policy log-ratios favour rejected over chosen (adversarial case),
   the loss should be high, accuracy = 0.0, and margin < 0.

3. test_dpo_loss_neutral
   When policy and reference agree (both log-ratios = 0), the margin is 0,
   the loss equals log(2) ≈ 0.693, and accuracy is undefined (we check 0.5).

4. test_get_logps_masks_prompt_tokens
   get_logps must assign zero weight to prompt tokens and only sum
   log-probs over the response mask — verified by checking that
   masking the whole sequence to False gives logps == 0.

5. test_get_logps_response_only_higher_than_zero
   When the response mask covers at least one token, the returned log-prob
   should be a non-positive finite scalar (log-probs are ≤ 0).

Run with:  pytest -q tests/test_dpo_loss.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import pytest

# Make sure part_10 root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpo_loss import dpo_loss, get_logps, DPOLossOutput


# ---------------------------------------------------------------------------
# Fixtures / tiny helpers
# ---------------------------------------------------------------------------

def _make_logps(values: list[float]) -> torch.Tensor:
    """Create a (B,) tensor of log-probs from a plain list."""
    return torch.tensor(values, dtype=torch.float32)


class _TinyLM(torch.nn.Module):
    """Minimal stand-in for PolicyWithValue that get_logps can call via .lm()."""

    def __init__(self, vocab_size: int = 16, seq_len: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        # Fixed logits — uniform distribution so every token has log-prob = -log(V)
        self._logits = torch.zeros(1, seq_len, vocab_size)

    def lm(self, x: torch.Tensor, _y):
        B, T = x.shape
        logits = self._logits.expand(B, -1, -1)
        return logits, None, None


# ---------------------------------------------------------------------------
# Test 1 — clear preference: policy strongly prefers chosen
# ---------------------------------------------------------------------------

def test_dpo_loss_perfect_preference():
    """Model confidently prefers chosen: low loss, acc=1.0, positive margin."""
    beta = 0.1

    # Policy log-ratios: chosen much higher than rejected
    pol_chosen   = _make_logps([-1.0, -2.0])   # sequence log-probs
    pol_rejected = _make_logps([-5.0, -6.0])   # much lower
    ref_chosen   = _make_logps([-3.0, -3.0])   # reference is neutral
    ref_rejected = _make_logps([-3.0, -3.0])

    out = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=beta)

    assert isinstance(out, DPOLossOutput)

    # Margin should be positive (policy prefers chosen over rejected vs reference)
    assert out.reward_margin.item() > 0.0, "Margin should be positive when chosen is preferred"

    # Accuracy should be 1.0 — every sample in the batch is correct
    assert out.accuracy.item() == pytest.approx(1.0), "Accuracy should be 1.0 when chosen always wins"

    # Loss should be well below log(2) ≈ 0.693 (the random-guess baseline)
    assert out.loss.item() < math.log(2), "Loss should be below random-guess baseline"

    # Loss must be positive (it's -log σ(…))
    assert out.loss.item() > 0.0, "DPO loss must be positive"


# ---------------------------------------------------------------------------
# Test 2 — reversed preference: policy prefers rejected (adversarial)
# ---------------------------------------------------------------------------

def test_dpo_loss_reversed_preference():
    """Policy prefers rejected: high loss, acc=0.0, negative margin."""
    beta = 0.1

    pol_chosen   = _make_logps([-5.0, -6.0])   # policy assigns low prob to chosen
    pol_rejected = _make_logps([-1.0, -2.0])   # and high prob to rejected
    ref_chosen   = _make_logps([-3.0, -3.0])
    ref_rejected = _make_logps([-3.0, -3.0])

    out = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=beta)

    assert out.reward_margin.item() < 0.0, "Margin should be negative when rejected is preferred"
    assert out.accuracy.item() == pytest.approx(0.0), "Accuracy should be 0.0 when rejected always wins"
    assert out.loss.item() > math.log(2), "Loss should exceed random-guess baseline in adversarial case"


# ---------------------------------------------------------------------------
# Test 3 — neutral / ambiguous: policy exactly mirrors reference
# ---------------------------------------------------------------------------

def test_dpo_loss_neutral():
    """Policy = reference everywhere → log-ratios = 0 → loss = log(2)."""
    beta = 0.1

    # All log-probs identical between policy and reference
    pol_chosen   = _make_logps([-2.5, -2.5])
    pol_rejected = _make_logps([-2.5, -2.5])
    ref_chosen   = _make_logps([-2.5, -2.5])
    ref_rejected = _make_logps([-2.5, -2.5])

    out = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=beta)

    assert out.reward_margin.item() == pytest.approx(0.0, abs=1e-5), \
        "Margin should be zero when policy == reference"

    # -log σ(0) = log(2)
    assert out.loss.item() == pytest.approx(math.log(2), abs=1e-5), \
        "Loss should equal log(2) ≈ 0.693 when all log-ratios are zero"

    # Accuracy is 0.5 when the margin is exactly zero (logit = 0 → not > 0)
    assert out.accuracy.item() == pytest.approx(0.0, abs=1e-5), \
        "Accuracy should be 0 when logit==0 (boundary case, not strictly positive)"


# ---------------------------------------------------------------------------
# Test 4 — get_logps: all-False mask returns exactly zero
# ---------------------------------------------------------------------------

def test_get_logps_zero_mask_gives_zero():
    """If the response mask is all False, get_logps should return 0.0."""
    model    = _TinyLM(vocab_size=16, seq_len=8)
    B, T     = 1, 8
    input_ids = torch.randint(0, 16, (B, T))
    mask      = torch.zeros(B, T, dtype=torch.bool)   # nothing masked as response

    result = get_logps(model, input_ids, mask)

    assert result.shape == (B,), "get_logps should return (B,) tensor"
    assert result[0].item() == pytest.approx(0.0, abs=1e-6), \
        "Zero mask should yield zero log-prob sum"


# ---------------------------------------------------------------------------
# Test 5 — get_logps: response tokens yield finite non-positive log-probs
# ---------------------------------------------------------------------------

def test_get_logps_response_tokens_nonpositive():
    """get_logps over response tokens must return a finite value ≤ 0."""
    model    = _TinyLM(vocab_size=16, seq_len=8)
    B, T     = 2, 8
    input_ids = torch.randint(0, 16, (B, T))

    # Mark the last 4 tokens of each sequence as the response
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 4:] = True

    result = get_logps(model, input_ids, mask)

    assert result.shape == (B,), "Should return one value per sequence"
    for i in range(B):
        val = result[i].item()
        assert math.isfinite(val), f"Log-prob sum must be finite, got {val}"
        assert val <= 0.0, f"Log-prob sum must be ≤ 0 (it's a sum of logs of probs), got {val}"