"""phase5_sft_eval.py — Phase 5 SFT re-evaluation with improved sampling.

Changes from original sft_logger.py:
  - temperature:    0.7  → 0.3   (less randomness, exploits existing representations)
  - top_k:          50   → 20    (tighter sampling, more focused outputs)
  - max_new_tokens: 64   → 96    (allows more complete responses)

No retraining — SFT is a frozen checkpoint. This tests whether better
sampling parameters alone can raise the SFT ceiling.

Output: sft_phase5_metrics.json  (same folder as sft_phase1 results)

Usage
-----
python phase5_sft_eval.py \
    --sft_ckpt  ../part_6/runs/sft-demo/model_last.pt \
    --rm_ckpt   ../part_7/runs/rm-demo/model_last.pt \
    --bpe_dir   ../part_4/runs/part4-demo/tokenizer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

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

# ── Phase 5 sampling parameters ─────────────────────────────────────────────
TEMPERATURE    = 0.3   # was 0.7
TOP_K          = 20    # was 50
MAX_NEW_TOKENS = 96    # was 64


def run(sft_ckpt: str, rm_ckpt: str, bpe_dir: str | None, n_samples: int = 16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Phase 5 SFT params → temp={TEMPERATURE}  top_k={TOP_K}  max_tokens={MAX_NEW_TOKENS}")

    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    s_ckpt = torch.load(sft_ckpt, map_location=device)
    cfg    = s_ckpt.get('config', {})
    sft    = PolicyWithValue(
        cfg.get('vocab_size', tok.vocab_size),
        cfg.get('block_size', tok.block_size),
        cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128),
    ).to(device)
    state = s_ckpt['model']
    if any(k.startswith('lm.') for k in state):
        sft.load_state_dict(state, strict=False)
    else:
        sft.load_state_dict({f'lm.{k}': v for k, v in state.items()}, strict=False)
    sft.eval()

    r_ckpt = torch.load(rm_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size  = r_ckpt['config'].get('vocab_size', tok.vocab_size),
        block_size  = r_ckpt['config'].get('block_size', tok.block_size),
        n_layer     = r_ckpt['config'].get('n_layer',    4),
        n_head      = r_ckpt['config'].get('n_head',     4),
        n_embd      = r_ckpt['config'].get('n_embd',   256),
    ).to(device)
    rm.load_state_dict(r_ckpt['model'])
    rm.eval()

    prompts = sample_prompts(n_samples)
    logs: list[dict] = []

    print(f"\n--- Phase 5 SFT Evaluation ({n_samples} prompts) ---")

    for i, prompt in enumerate(prompts):
        prefix = format_prompt_only(prompt).replace('</s>', '')
        ids    = tok.encode(prefix)
        x      = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)

        with torch.no_grad():
            out       = sft.generate(x, max_new_tokens=MAX_NEW_TOKENS,
                                     temperature=TEMPERATURE, top_k=TOP_K)
            resp_ids  = out[0].tolist()[len(ids):]
            resp_text = tok.decode(resp_ids)

            full_text = format_example(Example(prompt, resp_text))
            z         = torch.tensor(
                [tok.encode(full_text)[:tok.block_size]], dtype=torch.long, device=device
            )
            reward = rm(z)[0].item()

        logs.append({'prompt': prompt, 'response': resp_text, 'reward': round(reward, 4)})
        print(f"[{i+1:>2}/{n_samples}] Reward: {reward:+.3f} | {resp_text[:60].strip()!r}")

    avg = sum(l['reward'] for l in logs) / n_samples
    output = {
        'phase': 5,
        'tweaks': {
            'temperature':    TEMPERATURE,
            'top_k':          TOP_K,
            'max_new_tokens': MAX_NEW_TOKENS,
        },
        'average_reward': round(avg, 4),
        'samples': logs,
    }

    # Save to same folder as original SFT results
    out_path = Path(__file__).resolve().parent / 'sft_phase5_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\nPhase 5 avg reward: {avg:.4f}  (Phase 1 baseline: 3.009)")
    print(f"Saved → {out_path.absolute()}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Phase 5 SFT evaluation')
    p.add_argument('--sft_ckpt', type=str, required=True)
    p.add_argument('--rm_ckpt',  type=str, required=True)
    p.add_argument('--bpe_dir',  type=str, default=None)
    p.add_argument('--n',        type=int, default=16)
    args = p.parse_args()
    run(args.sft_ckpt, args.rm_ckpt, args.bpe_dir, args.n)