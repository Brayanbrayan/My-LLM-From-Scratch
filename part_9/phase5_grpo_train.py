"""phase5_grpo_train.py — Phase 5 GRPO retraining with improved hyperparameters.

Changes from original train_grpo.py:
  - group_size:     4    → 8     (doubles samples per group, reduces collapse risk)
  - gen_temperature:0.8  → 1.0   (more diverse samples → better advantage signal)
  - lr:             1e-5 → 5e-6  (slower updates, reduces overfitting per batch)
  - eval_temp:      0.7  → 0.3   (more deterministic evaluation)
  - eval_top_k:     50   → 20    (tighter eval sampling)

What each tweak aims to fix:
  group_size ×2    — The primary GRPO failure was group collapse: with k=4 on a
                     low-entropy model, all samples often became identical, zeroing
                     the advantage. k=8 gives more chances for at least some samples
                     to diverge, keeping the gradient signal alive.
  gen_temp → 1.0   — Higher generation temperature increases output diversity during
                     rollout. More diverse groups mean larger standard deviation,
                     which means more meaningful advantage normalisation. This works
                     together with the larger group size.
  lr ×0.5          — With a larger group and noisier generations, smaller learning
                     rate steps prevent the policy from overcorrecting on any single
                     batch and maintains more stable training.

Output: grpo_phase5_metrics.json  saved to part_9/ alongside grpo_phase1_metrics.json

Usage (run from part_9/)
-----
python ../part_10/phase5_grpo_train.py \
    --sft_ckpt    ../part_6/runs/sft-demo/model_last.pt \
    --reward_ckpt ../part_7/runs/rm-demo/model_last.pt \
    --bpe_dir     ../part_4/runs/part4-demo/tokenizer \
    --out         runs/grpo-phase5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
for _sub in ['part_10', 'part_9', 'part_8', 'part_6', 'part_4', 'part_3', 'part_7']:
    sys.path.insert(0, str(_HERE.parent / _sub))
sys.path.insert(0, str(_HERE))

from policy import PolicyWithValue
from rollout import RLHFTokenizer, format_prompt_only, sample_prompts
from model_reward import RewardModel

try:
    from part_6.formatters import Example, format_example
except ModuleNotFoundError:
    from formatters import Example, format_example

# ── Phase 5 hyperparameters ──────────────────────────────────────────────────
GROUP_SIZE    = 8      # was 4
GEN_TEMP      = 1.0    # was 0.8  — more diverse rollout samples
LR            = 5e-6   # was 1e-5
EVAL_TEMP     = 0.3    # was 0.7
EVAL_TOP_K    = 20     # was 50
EVAL_MAX_TOK  = 96     # was 64
RESP_LEN      = 64     # rollout length (unchanged)
KL_COEF       = 0.01   # unchanged


def load_policy(ckpt_path, device, tok):
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
    return model, cfg


def get_logps(model, input_ids, response_mask):
    """Per-sequence log-prob sum over response tokens only."""
    logits, _, _ = model.lm(input_ids, None)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask   = response_mask[:, 1:].contiguous()
    log_probs    = torch.log_softmax(shift_logits, dim=-1)
    token_logps  = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * shift_mask.float()).sum(dim=-1)


def build_pair(prompt_ids, resp_ids, block_size, device):
    full = prompt_ids + resp_ids
    if len(full) > block_size:
        overflow   = len(full) - block_size
        prompt_ids = prompt_ids[overflow:]
        full       = prompt_ids + resp_ids
    full       = full[:block_size]
    prompt_len = min(len(prompt_ids), len(full))
    ids        = torch.tensor([full], dtype=torch.long, device=device)
    mask       = torch.zeros(1, len(full), dtype=torch.bool, device=device)
    mask[0, prompt_len:] = True
    return ids, mask


def main():
    ap = argparse.ArgumentParser(description='Phase 5 GRPO retraining')
    ap.add_argument('--sft_ckpt',    type=str, required=True)
    ap.add_argument('--reward_ckpt', type=str, required=True)
    ap.add_argument('--bpe_dir',     type=str, default=None)
    ap.add_argument('--out',         type=str, default='runs/grpo-phase5')
    ap.add_argument('--steps',       type=int, default=200)
    ap.add_argument('--block_size',  type=int, default=256)
    ap.add_argument('--cpu',         action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    print(f"Phase 5 GRPO → group_size={GROUP_SIZE}  gen_temp={GEN_TEMP}  lr={LR}")

    tok    = RLHFTokenizer(args.block_size, bpe_dir=args.bpe_dir)
    policy, cfg = load_policy(args.sft_ckpt, device, tok)

    vocab_size  = cfg.get('vocab_size', tok.vocab_size)
    block_size  = cfg.get('block_size', tok.block_size)
    n_layer     = cfg.get('n_layer', 2)
    n_head      = cfg.get('n_head', 2)
    n_embd      = cfg.get('n_embd', 128)

    ref, _ = load_policy(args.sft_ckpt, device, tok)
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

    print(f"\n--- Phase 5 GRPO Training ({args.steps} steps, k={GROUP_SIZE}) ---")

    for step in range(1, args.steps + 1):
        prompt = prompts[(step - 1) % len(prompts)]
        prefix = format_prompt_only(prompt).replace('</s>', '')
        p_ids  = tok.encode(prefix)
        x      = torch.tensor([p_ids[-block_size:]], dtype=torch.long, device=device)

        # Generate GROUP_SIZE responses
        group_rewards = []
        group_logps   = []
        group_ref_lps = []

        for _ in range(GROUP_SIZE):
            with torch.no_grad():
                out      = policy.generate(x, max_new_tokens=RESP_LEN,
                                           temperature=GEN_TEMP, top_k=50)
            resp_ids  = out[0].tolist()[len(p_ids):]
            resp_text = tok.decode(resp_ids)

            full_text = format_example(Example(prompt, resp_text))
            z         = torch.tensor(
                [tok.encode(full_text)[:tok.block_size]], dtype=torch.long, device=device
            )
            with torch.no_grad():
                reward = rm(z)[0].item()

            ids_t, mask_t = build_pair(p_ids, tok.encode(resp_text), block_size, device)
            policy_lp = get_logps(policy, ids_t, mask_t)
            with torch.no_grad():
                ref_lp = get_logps(ref, ids_t, mask_t)

            group_rewards.append(reward)
            group_logps.append(policy_lp)
            group_ref_lps.append(ref_lp)

        rewards_t = torch.tensor(group_rewards, dtype=torch.float, device=device)
        mean_r    = rewards_t.mean()
        std_r     = rewards_t.std().clamp_min(1e-6)
        advantages = (rewards_t - mean_r) / std_r

        # GRPO loss — policy gradient weighted by group-relative advantage
        logps_stack = torch.stack(group_logps)      # (k,)
        ref_stack   = torch.stack(group_ref_lps)    # (k,)
        kl_penalty  = KL_COEF * (logps_stack - ref_stack.detach())
        loss = -(advantages.detach() * (logps_stack - kl_penalty)).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            print(f"step {step:>4} | loss {loss.item():.4f} | mean_r {mean_r.item():.3f} | std {std_r.item():.3f}")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size, 'block_size': block_size,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
    }}, str(out_dir / 'model_last.pt'))
    print(f"\nCheckpoint saved → {out_dir / 'model_last.pt'}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\n--- Phase 5 GRPO Evaluation (temp={EVAL_TEMP}  top_k={EVAL_TOP_K}) ---")
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
            'group_size':  GROUP_SIZE,
            'gen_temp':    GEN_TEMP,
            'lr':          LR,
            'eval_temp':   EVAL_TEMP,
            'eval_top_k':  EVAL_TOP_K,
        },
        'average_reward': round(avg, 4),
        'samples': logs,
    }

    save_dir = Path(__file__).resolve().parents[1] / 'part_9'
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / 'grpo_phase5_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\nPhase 5 avg GRPO reward: {avg:.4f}  (Phase 1 baseline: -0.116)")
    print(f"Saved → {out_path.absolute()}")


if __name__ == '__main__':
    main()