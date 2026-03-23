"""dpo_logger.py — Phase 1 Benchmark Evaluator for DPO (Part 10).

Loads the trained DPO checkpoint, runs inference on the 16 standard prompts,
scores each response with the Part 7 Reward Model, and computes KL divergence
from the SFT base — mirroring the schema of ppo_phase1_metrics.json exactly
so results can be compared side-by-side.

Deliverable
-----------
dpo_phase1_metrics.json  (list of per-prompt dicts + summary averages)

Usage
-----
python dpo_logger.py \
    --dpo_ckpt  runs/dpo-demo/model_last.pt \
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

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'part_8'))
sys.path.insert(0, str(_HERE.parent / 'part_7'))
sys.path.insert(0, str(_HERE.parent / 'part_6'))
sys.path.insert(0, str(_HERE.parent / 'part_4'))
sys.path.insert(0, str(_HERE.parent / 'part_3'))

from policy import PolicyWithValue
from rollout import RLHFTokenizer, format_prompt_only, sample_prompts, model_logprobs
from model_reward import RewardModel

try:
    from part_6.formatters import Example, format_example
except ModuleNotFoundError:
    from formatters import Example, format_example


# ---------------------------------------------------------------------------
# Loader helper (shared pattern from ppo_logger.py)
# ---------------------------------------------------------------------------

def _load_policy(
    ckpt_path: str,
    device: torch.device,
    tok: RLHFTokenizer,
) -> tuple[PolicyWithValue, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('config', {})

    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer    = cfg.get('n_layer', 2)
    n_head     = cfg.get('n_head', 2)
    n_embd     = cfg.get('n_embd', 128)

    model = PolicyWithValue(
        vocab_size, block_size, n_layer, n_head, n_embd
    ).to(device)

    state = ckpt['model']

    # Detect whether state dict has PolicyWithValue keys or bare LM keys
    # (mirrors the pattern established in ppo_logger.py)
    has_lm_prefix = any(k.startswith('lm.') for k in state)
    if has_lm_prefix:
        model.load_state_dict(state, strict=False)
    else:
        remapped = {f'lm.{k}': v for k, v in state.items()}
        model.load_state_dict(remapped, strict=False)

    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_dpo_eval(
    dpo_ckpt:  str,
    sft_ckpt:  str,
    rm_ckpt:   str,
    bpe_dir:   str | None,
    n_samples: int = 16,
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    # 1. DPO policy (the model being evaluated)
    dpo_model, cfg = _load_policy(dpo_ckpt, device, tok)

    # 2. SFT reference (the frozen anchor — used for KL)
    sft_model, _ = _load_policy(sft_ckpt, device, tok)
    for p in sft_model.parameters():
        p.requires_grad_(False)

    # 3. Reward model
    rckpt = torch.load(rm_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size  = rckpt['config'].get('vocab_size', tok.vocab_size),
        block_size  = rckpt['config'].get('block_size', tok.block_size),
        n_layer     = rckpt['config'].get('n_layer',    4),
        n_head      = rckpt['config'].get('n_head',     4),
        n_embd      = rckpt['config'].get('n_embd',   256),
    ).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    prompts = sample_prompts(n_samples)
    logs: list[dict] = []

    print(f"\n--- DPO Phase 1 Benchmark ({n_samples} prompts) ---")

    for i, prompt in enumerate(prompts):
        prefix = format_prompt_only(prompt).replace('</s>', '')
        ids    = tok.encode(prefix)
        x      = torch.tensor(
            [ids[-tok.block_size:]], dtype=torch.long, device=device
        )

        with torch.no_grad():
            # Generate response using the DPO policy
            out      = dpo_model.generate(x, max_new_tokens=64, temperature=0.7)
            resp_ids  = out[0].tolist()[len(ids):]
            resp_text = tok.decode(resp_ids)

            # KL divergence: KL(DPO || SFT) on the generated tokens
            # model_logprobs returns (B, T-1) — same convention as ppo_logger.py
            dpo_lp = model_logprobs(dpo_model, out)  # (1, T-1)
            sft_lp = model_logprobs(sft_model, out)  # (1, T-1)

            # Slice to the response portion only (response starts at len(ids))
            # Shift by -1 because model_logprobs already removed one position
            resp_start   = max(0, len(ids) - 1)
            kl_per_token = dpo_lp[0, resp_start:] - sft_lp[0, resp_start:]
            avg_kl       = kl_per_token.mean().item()

            # Reward model score
            full_text = format_example(Example(prompt, resp_text))
            z = torch.tensor(
                [tok.encode(full_text)[: tok.block_size]],
                dtype=torch.long, device=device,
            )
            reward = rm(z)[0].item()

        entry = {
            'prompt':            prompt,
            'response':          resp_text,
            'reward':            round(reward, 4),
            'avg_kl_divergence': round(avg_kl, 6),
        }
        logs.append(entry)

        print(
            f"[{i+1:>2}/{n_samples}] "
            f"Reward: {reward:+.3f} | KL: {avg_kl:.4f} | "
            f"Response: {resp_text[:60].strip()!r}…"
        )

    # ---- Summary ----------------------------------------------------------
    avg_reward = sum(l['reward']            for l in logs) / n_samples
    avg_kl     = sum(l['avg_kl_divergence'] for l in logs) / n_samples

    summary = {
        'avg_reward':       round(avg_reward, 4),
        'avg_kl_divergence': round(avg_kl,   6),
        'n_samples':         n_samples,
    }
    output = {'summary': summary, 'per_prompt': logs}

    # ---- Save deliverable -------------------------------------------------
    out_path = Path('dpo_phase1_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\n✅ DPO Phase 1 complete.")
    print(f"   Avg Reward : {avg_reward:.4f}")
    print(f"   Avg KL     : {avg_kl:.6f}")
    print(f"   Saved to   : {out_path.absolute()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo_ckpt',  default='runs/dpo-demo/model_last.pt')
    parser.add_argument('--sft_ckpt',  default='../part_6/runs/sft-demo/model_last.pt')
    parser.add_argument('--rm_ckpt',   default='../part_7/runs/rm-demo/model_last.pt')
    parser.add_argument('--bpe_dir',   default='../part_4/runs/part4-demo/tokenizer')
    parser.add_argument('--n_samples', type=int, default=16)
    args = parser.parse_args()

    run_dpo_eval(
        dpo_ckpt  = args.dpo_ckpt,
        sft_ckpt  = args.sft_ckpt,
        rm_ckpt   = args.rm_ckpt,
        bpe_dir   = args.bpe_dir,
        n_samples = args.n_samples,
    )