"""phase5_ppo_train.py — Phase 5 PPO retraining with improved hyperparameters.

Changes from original train_ppo.py:
  - kl_coef:        0.01  → 0.1   (10× stronger KL anchor to reference)
  - lr:             1e-5  → 5e-6  (slower updates, less aggressive policy drift)
  - resp_len:       64    → 96    (more complete rollout responses for RM scoring)
  - eval temp:      0.7   → 0.3   (more deterministic evaluation)

What each tweak aims to fix:
  kl_coef ×10  — The original KL penalty was too weak. PPO scored identically
                 to SFT on several prompts (capital of France: -7.10 for both),
                 meaning the policy had drifted without learning. A stronger
                 penalty keeps the policy anchored while still allowing movement
                 toward high-reward regions.
  lr ×0.5      — Smaller steps mean each rollout batch causes less disruption
                 to the existing representations. Reduces catastrophic forgetting
                 on prompts where SFT was already performing well.
  resp_len +32 — Longer responses give the reward model more signal to work with.
                 Many of the original PPO responses were truncated fragments that
                 scored based on opening tokens rather than full content.

Output: ppo_phase5_metrics.json  saved to part_8/ alongside ppo_phase1_metrics.json

Usage (run from part_8/)
-----
python ../part_10/phase5_ppo_train.py \
    --policy_ckpt ../part_6/runs/sft-demo/model_last.pt \
    --reward_ckpt ../part_7/runs/rm-demo/model_last.pt \
    --bpe_dir     ../part_4/runs/part4-demo/tokenizer \
    --out         runs/ppo-phase5
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

from policy import PolicyWithValue
from rollout import (RLHFTokenizer, format_prompt_only, format_example,
                          sample_prompts, model_logprobs)
from model_reward import RewardModel
from ppo_loss import ppo_losses

try:
    from part_6.formatters import Example
except ModuleNotFoundError:
    from formatters import Example

# ── Phase 5 hyperparameters ──────────────────────────────────────────────────
KL_COEF        = 0.1    # was 0.01
LR             = 5e-6   # was 1e-5
RESP_LEN       = 96     # was 64
EVAL_TEMP      = 0.3    # was 0.7
EVAL_TOP_K     = 20     # was 50
EVAL_MAX_TOK   = 96


def compute_reward(rm, tok, prompt, response, device):
    text = format_example(Example(prompt, response))
    ids  = tok.encode(text)
    x    = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    with torch.no_grad():
        r = rm(x)
    return float(r[0].item())


def load_policy(ckpt_path, device, tok):
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt.get('config', {})
    vs    = cfg.get('vocab_size', tok.vocab_size)
    bs    = cfg.get('block_size', tok.block_size)
    model = PolicyWithValue(vs, bs, cfg.get('n_layer', 2),
                            cfg.get('n_head', 2), cfg.get('n_embd', 128)).to(device)
    state = ckpt['model']
    if any(k.startswith('lm.') for k in state):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict({f'lm.{k}': v for k, v in state.items()}, strict=False)
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description='Phase 5 PPO retraining')
    ap.add_argument('--policy_ckpt', type=str, required=True)
    ap.add_argument('--reward_ckpt', type=str, required=True)
    ap.add_argument('--bpe_dir',     type=str, default=None)
    ap.add_argument('--out',         type=str, default='runs/ppo-phase5')
    ap.add_argument('--steps',       type=int, default=200)
    ap.add_argument('--batch_size',  type=int, default=4)
    ap.add_argument('--block_size',  type=int, default=256)
    ap.add_argument('--cpu',         action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    print(f"Phase 5 PPO → kl_coef={KL_COEF}  lr={LR}  resp_len={RESP_LEN}")

    tok = RLHFTokenizer(args.block_size, bpe_dir=args.bpe_dir)

    policy, cfg = load_policy(args.policy_ckpt, device, tok)
    vocab_size  = cfg.get('vocab_size', tok.vocab_size)
    block_size  = cfg.get('block_size', tok.block_size)
    n_layer     = cfg.get('n_layer', 2)
    n_head      = cfg.get('n_head', 2)
    n_embd      = cfg.get('n_embd', 128)

    ref, _ = load_policy(args.policy_ckpt, device, tok)
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size  = rckpt['config'].get('vocab_size', tok.vocab_size),
        block_size  = rckpt['config'].get('block_size', tok.block_size),
        n_layer     = rckpt['config'].get('n_layer',    4),
        n_head      = rckpt['config'].get('n_head',     4),
        n_embd      = rckpt['config'].get('n_embd',   256),
    ).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt     = torch.optim.AdamW(policy.parameters(), lr=LR, betas=(0.9, 0.999))
    prompts = sample_prompts(16)

    print(f"\n--- Phase 5 PPO Training ({args.steps} steps) ---")
    for step in range(1, args.steps + 1):
        batch_prompts = prompts[(step * args.batch_size) % len(prompts) :
                                ((step + 1) * args.batch_size) % len(prompts)]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size - len(batch_prompts)]

        texts  = [format_prompt_only(p).replace('</s>', '') for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]

        with torch.no_grad():
            out_ids = []
            for x in in_ids:
                idx = torch.tensor([x], dtype=torch.long, device=device)
                out = policy.generate(idx, max_new_tokens=RESP_LEN, temperature=0.7, top_k=50)
                out_ids.append(out[0].tolist())

        data = []
        for i, prompt in enumerate(batch_prompts):
            full     = out_ids[i]
            p_ids    = in_ids[i][-block_size:]
            boundary = len(p_ids)
            resp_ids = full[boundary:]
            resp_txt = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_txt, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))

        max_len = min(block_size, max(t[0].numel() for t in data))
        B       = len(data)
        seq     = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask    = torch.zeros(B, max_len, dtype=torch.bool,  device=device)
        rewards = torch.zeros(B, max_len, dtype=torch.float, device=device)

        for i, (ids, boundary, r_scalar) in enumerate(data):
            L_full = ids.numel()
            L      = min(L_full, max_len)
            drop   = L_full - L
            b      = max(0, boundary - drop)
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2
            mask[i, b:L]    = True
            rewards[i, L-1] = r_scalar

        pol_lp = model_logprobs(policy, seq)
        ref_lp = model_logprobs(ref,    seq)
        with torch.no_grad():
            logits, values, _ = policy(seq, None)
        values = values[:, :-1]

        act_mask  = mask[:, 1:]
        old_logp  = pol_lp[act_mask].detach()
        ref_logp  = ref_lp[act_mask].detach()
        old_values = values[act_mask].detach()

        kl       = old_logp - ref_logp
        shaped_r = rewards[:, 1:][act_mask] - KL_COEF * kl
        returns  = shaped_r
        adv      = returns - old_values
        adv      = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

        policy.train()
        logits_new, values_new_full, _ = policy(seq, None)
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels    = seq[:, 1:]
        new_logp  = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[act_mask]
        new_values = values_new_full[:, :-1][act_mask]

        out_loss = ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
                              clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0)
        loss = out_loss.total_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()

        if step % 20 == 0:
            print(f"step {step:>4} | loss {loss.item():.4f}")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size, 'block_size': block_size,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
    }}, str(out_dir / 'model_last.pt'))
    print(f"\nCheckpoint saved → {out_dir / 'model_last.pt'}")

    # ── Evaluate on 16 prompts ───────────────────────────────────────────────
    print(f"\n--- Phase 5 PPO Evaluation (temp={EVAL_TEMP}  top_k={EVAL_TOP_K}) ---")
    logs: list[dict] = []

    for i, prompt in enumerate(prompts):
        prefix = format_prompt_only(prompt).replace('</s>', '')
        ids    = tok.encode(prefix)
        x      = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)

        with torch.no_grad():
            out       = policy.generate(x, max_new_tokens=EVAL_MAX_TOK,
                                        temperature=EVAL_TEMP, top_k=EVAL_TOP_K)
            resp_ids  = out[0].tolist()[len(ids):]
            resp_text = tok.decode(resp_ids)

            full_text = format_example(Example(prompt, resp_text))
            z         = torch.tensor(
                [tok.encode(full_text)[:tok.block_size]], dtype=torch.long, device=device
            )
            reward = rm(z)[0].item()

        logs.append({'prompt': prompt, 'response': resp_text,
                     'reward': round(reward, 4)})
        print(f"[{i+1:>2}/16] {reward:+.3f} | {resp_text[:60].strip()!r}")

    avg = sum(l['reward'] for l in logs) / len(logs)
    output = {
        'phase': 5,
        'tweaks': {
            'kl_coef':        KL_COEF,
            'lr':             LR,
            'resp_len':       RESP_LEN,
            'eval_temp':      EVAL_TEMP,
            'eval_top_k':     EVAL_TOP_K,
        },
        'average_reward': round(avg, 4),
        'samples': logs,
    }

    # Save alongside ppo_phase1_metrics.json in part_8/
    save_dir = Path(__file__).resolve().parents[1] / 'part_8'
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / 'ppo_phase5_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\nPhase 5 avg PPO reward: {avg:.4f}  (Phase 1 baseline: 3.992)")
    print(f"Saved → {out_path.absolute()}")


if __name__ == '__main__':
    main()