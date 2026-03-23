"""eval_dpo.py — Standalone evaluation script for a DPO checkpoint.

Mirrors eval_ppo.py in structure and output so results are directly comparable.
Generates responses from the DPO policy, scores them with the Part 7 Reward
Model, and prints the average RM reward over --n prompts.

Usage
-----
python eval_dpo.py \
    --policy_ckpt runs/dpo-demo/model_last.pt \
    --reward_ckpt ../part_7/runs/rm-demo/model_last.pt \
    --sft_ckpt    ../part_6/runs/sft-demo/model_last.pt \
    --bpe_dir     ../part_4/runs/part4-demo/tokenizer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _sub in ['part_10', 'part_8', 'part_6', 'part_4', 'part_3', 'part_7']:
    sys.path.insert(0, str(_HERE.parent / _sub))
sys.path.insert(0, str(_HERE))

from policy       import PolicyWithValue
from rollout      import RLHFTokenizer, sample_prompts, format_prompt_only
from model_reward import RewardModel

try:
    from part_6.formatters import Example, format_example
except ModuleNotFoundError:
    from formatters import Example, format_example


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_policy(ckpt_path: str, device: torch.device, tok: RLHFTokenizer):
    """Load PolicyWithValue — handles both SFT and DPO/PPO checkpoint formats."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt.get('config', {})
    model = PolicyWithValue(
        cfg.get('vocab_size', tok.vocab_size),
        cfg.get('block_size', tok.block_size),
        cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128),
    ).to(device)
    state = ckpt['model']
    if any(k.startswith('lm.') for k in state):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict({f'lm.{k}': v for k, v in state.items()}, strict=False)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def score_policy(
    policy_ckpt: str,
    reward_ckpt: str,
    sft_ckpt:    str,
    bpe_dir:     str | None,
    n:           int = 16,
) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok    = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    # DPO policy being evaluated
    pol, _ = _load_policy(policy_ckpt, device, tok)

    # SFT reference — side-by-side comparison (same pattern as eval_ppo.py)
    ref, _ = _load_policy(sft_ckpt, device, tok)
    for p_ in ref.parameters():
        p_.requires_grad_(False)

    # Reward model
    rckpt = torch.load(reward_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size  = rckpt['config'].get('vocab_size', tok.vocab_size),
        block_size  = rckpt['config'].get('block_size', tok.block_size),
        n_layer     = rckpt['config'].get('n_layer',    4),
        n_head      = rckpt['config'].get('n_head',     4),
        n_embd      = rckpt['config'].get('n_embd',   256),
    ).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    prompts = sample_prompts(n)
    rewards: list[float] = []

    for prompt in prompts:
        prefix = format_prompt_only(prompt).replace('</s>', '')
        ids    = tok.encode(prefix)
        x      = torch.tensor([ids[: tok.block_size]], dtype=torch.long, device=device)

        with torch.no_grad():
            y     = pol.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)
            y_ref = ref.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)  # noqa: F841

        resp = tok.decode(y[0].tolist()[len(ids[-tok.block_size:]):])

        text = format_example(Example(prompt, resp))
        z    = torch.tensor([tok.encode(text)[: tok.block_size]], dtype=torch.long, device=device)
        with torch.no_grad():
            r = rm(z)[0].item()
        rewards.append(r)

    return sum(rewards) / max(1, len(rewards))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate a DPO checkpoint with the Reward Model')
    p.add_argument('--policy_ckpt', type=str, required=True, help='DPO checkpoint')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--sft_ckpt',    type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--bpe_dir',     type=str, default=None)
    p.add_argument('--n',           type=int, default=16,    help='Number of prompts to evaluate')
    args = p.parse_args()

    avg_r = score_policy(args.policy_ckpt, args.reward_ckpt, args.sft_ckpt, args.bpe_dir, n=args.n)
    print(f"Average RM reward (DPO): {avg_r:.4f}")