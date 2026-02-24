from __future__ import annotations
import torch
from typing import List, Tuple


#tokenizer pref: BPE from part 4 -> fallback to ByteTokenizer from part 3
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
    #for causal LM: predict x[t+1] fro x[:t]
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
    return lp #(B, T-1)

# ----------KL-------------
def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    # Mean over tokens: KL(pi||ref) ≈ (logp_pi - logp_ref).mean()
    return (policy_logp - ref_logp).mean()

# ---------------small prompt source ------------
try:
    from datasets import load_dataset as _load_ds
except Exception:
    _load_ds = None

def sample_prompts(n: int) -> List[str]:
    if _load_ds is not None:
        try:
            ds = _load_ds("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                inst = (r.get('instruction') or '').strip()
                inp = (r.get('input') or '').strip()
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
        #fallback: tiny hardcoded prompts
    base = [
        "What is the capital of France?",
        "Write a haiku about the ocean.",
        "How do I make a peanut butter sandwich?",
        "What are the benefits of exercise?",
        "Explain quantum computing in simple terms.",
        "What is the meaning of life?",
        "How does photosynthesis work?",
        "What is the best way to learn programming?"
    ]
    return (base * ((n + len(base) -1)//len(base)))[:n]