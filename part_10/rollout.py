"""rollout.py — Tokenizer, logprob utilities, and dataset helpers.

Extended for Part 10 (DPO) with two new public helpers:
  - load_alpaca()          : loads tatsu-lab/alpaca and filters empty outputs
  - build_pair_tensors()   : concatenates prompt+response and builds the response mask
"""
from __future__ import annotations
import torch
from typing import List

# tokenizer pref: BPE from part 4 -> fallback to ByteTokenizer from part 3
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

from part_6.formatters import Example, format_example, format_prompt_only

# ---------- tokenizer helpers --------
class RLHFTokenizer:
    def __init__(self, block_size: int, bpe_dir: str | None = None, vocab_size: int = 8000):
        self.block_size = block_size
        self.tok = None
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size)
                if bpe_dir:
                    self.tok.load(bpe_dir)
            except Exception:
                self.tok.load(bpe_dir)
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("No tokenizer available for RHLF")

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    def decode(self, ids: List[int]) -> str:
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        return bytes(ids).decode('utf-8', errors='ignore')


# ---------------- logprob utilities ----------------

def shift_labels(x: torch.Tensor) -> torch.Tensor:
    # for causal LM: predict x[t+1] from x[:t]
    return x[:, 1:].contiguous()

def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs of the given labels.
    logits: (B,T,V), labels: (B,T) over same T
    returns: (B,T) log p(labels)
    """
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def model_logprobs(model, X: torch.Tensor) -> torch.Tensor:
    # compute log p(x[t+1]) | x[:t] for t
    logits, _, _ = model.lm(X, None) if hasattr(model, 'lm') else model(X, None)
    labels = shift_labels(X)
    lp = gather_logprobs(logits[:, :-1, :], labels)
    return lp  # (B, T-1)


# ---------- KL -------------
def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    # Mean over tokens: KL(pi||ref) ≈ (logp_pi - logp_ref).mean()
    return (policy_logp - ref_logp).mean()


# ---------------------------------------------------------------------------
# Dataset helper — used by train_dpo.py
# ---------------------------------------------------------------------------

try:
    from datasets import load_dataset as _load_ds
except Exception:
    _load_ds = None


def load_alpaca() -> list[dict]:
    """Load tatsu-lab/alpaca and return rows that have a non-empty output.

    Returns a plain list of dicts with keys: instruction, input, output.
    Raises RuntimeError if the datasets library is not installed.
    """
    if _load_ds is None:
        raise RuntimeError("pip install datasets  — required to load tatsu-lab/alpaca")
    ds = _load_ds("tatsu-lab/alpaca", split="train")
    filtered = [r for r in ds if (r.get('output') or '').strip()]
    return filtered


# ---------------------------------------------------------------------------
# Tensor-pair builder — used by train_dpo.py and dpo_logger.py
# ---------------------------------------------------------------------------

def build_pair_tensors(
    prompt_ids: List[int],
    response_ids: List[int],
    block_size: int,
    device: torch.device,
):
    """Concatenate prompt + response into a single sequence tensor and build a
    boolean response mask.  Sequences longer than block_size are truncated from
    the left of the prompt so the full response is always preserved.

    Parameters
    ----------
    prompt_ids   : token IDs for the formatted prompt
    response_ids : token IDs for the response (chosen or rejected)
    block_size   : maximum sequence length
    device       : target device

    Returns
    -------
    input_ids     : (1, T) long tensor  — full token sequence
    response_mask : (1, T) bool tensor  — True only at response positions
    """
    # Case 1: response alone exceeds block_size — clip the response, keep a
    #         minimal prompt prefix (at least 1 token) so the model has context.
    if len(response_ids) >= block_size:
        response_ids = response_ids[: block_size - 1]

    # Case 2: prompt + response together exceed block_size — trim the prompt
    #         from the left so the full (clipped) response is always preserved.
    full = prompt_ids + response_ids
    if len(full) > block_size:
        overflow   = len(full) - block_size
        prompt_ids = prompt_ids[overflow:]
        full       = prompt_ids + response_ids

    # Safety net: hard clip to block_size (guards against any remaining edge cases)
    full       = full[:block_size]
    prompt_len = min(len(prompt_ids), len(full))    # prompt_len can't exceed full

    T = len(full)
    input_ids     = torch.tensor([full], dtype=torch.long, device=device)
    response_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    response_mask[0, prompt_len:] = True            # supervise response tokens only

    return input_ids, response_mask


# ---------------------------------------------------------------------------
# Small prompt source — used by eval / logger scripts
# ---------------------------------------------------------------------------

def sample_prompts(n: int) -> List[str]:
    if _load_ds is not None:
        try:
            ds = _load_ds("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                inst = (r.get('instruction') or '').strip()
                inp  = (r.get('input')       or '').strip()
                if inp:
                    inst = inst + "\n" + inp
                if inst:
                    arr.append(inst)
                if len(arr) >= n:
                    break
            if arr:
                return arr
        except Exception:
            pass
    # fallback: tiny hardcoded prompts(Lol how can the fallback actually be what is in the actual dataset wont that cause confusion)
    base = [
        "What is the capital of France?",
        "Write a haiku about the ocean.",
        "How do I make a peanut butter sandwich?",
        "What are the benefits of exercise?",
        "Explain quantum computing in simple terms.",
        "What is the meaning of life?",
        "How does photosynthesis work?",
        "What is the best way to learn programming?",
    ]
    return (base * ((n + len(base) - 1) // len(base)))[:n]