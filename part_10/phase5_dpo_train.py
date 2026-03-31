"""phase5_dpo_train.py — Phase 5 DPO retraining with improved hyperparameters.

Changes from original train_dpo.py:
  - beta:           0.1  → 0.3   (3× stronger KL constraint, slows margin explosion)
  - lr:             1e-5 → 5e-6  (slower updates, less per-step overfitting)
  - rej_temp:       0.9  → 1.1   (more diverse rejected responses, cleaner preference signal)
  - eval_temp:      0.7  → 0.3   (more deterministic evaluation)
  - eval_top_k:     50   → 20    (tighter eval sampling)

What each tweak aims to fix:
  beta × 3     — The original reward margin exploded to 599 by step 150, meaning
                 the policy had drifted catastrophically far from the reference.
                 Beta controls how strongly the loss penalises large log-ratio gaps.
                 Tripling it makes large margins much more costly, keeping the
                 policy anchored and preventing per-prompt catastrophic forgetting.
  lr × 0.5     — Combined with stronger beta, a smaller learning rate means each
                 step moves the policy less aggressively. The two work together:
                 beta limits how far the loss tolerates drift, lr limits how fast
                 each update moves toward that limit.
  rej_temp 1.1 — Higher temperature for the rejected response generation increases
                 the quality gap between chosen (human text) and rejected (model
                 generation). A cleaner gap means a cleaner preference signal and
                 less chance of training on noisy pairs where rejected happens to
                 be coherent.

Output: dpo_phase5_metrics.json  saved to part_10/ alongside dpo_phase1_metrics.json

Usage (run from part_10/)
-----
python phase5_dpo_train.py \
    --sft_ckpt  ../part_6/runs/sft-demo/model_last.pt \
    --bpe_dir   ../part_4/runs/part4-demo/tokenizer \
    --out       runs/dpo-phase5
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
from rollout import RLHFTokenizer, format_prompt_only, load_alpaca, build_pair_tensors
from dpo_loss import dpo_loss, get_logps

try:
    from part_6.formatters import Example, format_example
except ModuleNotFoundError:
    from formatters import Example, format_example

# ── Phase 5 hyperparameters ──────────────────────────────────────────────────
BETA          = 0.3    # was 0.1  — stronger KL constraint
LR            = 5e-6   # was 1e-5 — slower updates
REJ_TEMP      = 1.1    # was 0.9  — more diverse rejected responses
EVAL_TEMP     = 0.3    # was 0.7
EVAL_TOP_K    = 20     # was 50
EVAL_MAX_TOK  = 96     # was 64
RESP_LEN      = 64     # rollout length (unchanged)


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


def main():
    ap = argparse.ArgumentParser(description='Phase 5 DPO retraining')
    ap.add_argument('--sft_ckpt',  type=str, required=True)
    ap.add_argument('--bpe_dir',   type=str, default=None)
    ap.add_argument('--out',       type=str, default='runs/dpo-phase5')
    ap.add_argument('--steps',     type=int, default=200)
    ap.add_argument('--block_size',type=int, default=256)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--cpu',       action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    print(f"Phase 5 DPO → beta={BETA}  lr={LR}  rej_temp={REJ_TEMP}")

    tok = RLHFTokenizer(block_size=256, bpe_dir=args.bpe_dir)

    policy, cfg = load_policy(args.sft_ckpt, device, tok)
    policy.train()

    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer    = cfg.get('n_layer', 2)
    n_head     = cfg.get('n_head', 2)
    n_embd     = cfg.get('n_embd', 128)

    for param in policy.val_head.parameters():
        param.requires_grad_(False)

    ref, _ = load_policy(args.sft_ckpt, device, tok)
    for param in ref.parameters():
        param.requires_grad_(False)
    ref.eval()

    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.999),
    )

    dataset = load_alpaca()
    print(f"Alpaca dataset: {len(dataset)} examples")

    training_logs: list[dict] = []

    print(f"\n--- Phase 5 DPO Training ({args.steps} steps) ---")

    for step in range(1, args.steps + 1):
        ex = dataset[(step - 1) % len(dataset)]

        instruction = (ex.get('instruction') or '').strip()
        inp         = (ex.get('input')       or '').strip()
        prompt      = f"{instruction}\n{inp}".strip() if inp else instruction
        chosen_text = (ex.get('output')      or '').strip()
        if not chosen_text:
            continue

        prompt_ids         = tok.encode(format_prompt_only(prompt).replace('</s>', ''))
        clipped_prompt_ids = prompt_ids[-block_size:]

        with torch.no_grad():
            p_t          = torch.tensor([clipped_prompt_ids], dtype=torch.long, device=device)
            rejected_out = ref.generate(p_t, max_new_tokens=RESP_LEN,
                                        temperature=REJ_TEMP, top_k=50)

        rejected_ids  = rejected_out[0].tolist()[len(clipped_prompt_ids):]
        rejected_text = tok.decode(rejected_ids)

        c_input, c_mask = build_pair_tensors(
            prompt_ids, tok.encode(chosen_text),   block_size, device)
        r_input, r_mask = build_pair_tensors(
            prompt_ids, tok.encode(rejected_text), block_size, device)

        policy.train()
        pol_c_lp = get_logps(policy, c_input, c_mask)
        pol_r_lp = get_logps(policy, r_input, r_mask)

        with torch.no_grad():
            ref_c_lp = get_logps(ref, c_input, c_mask)
            ref_r_lp = get_logps(ref, r_input, r_mask)

        result = dpo_loss(pol_c_lp, pol_r_lp, ref_c_lp, ref_r_lp, beta=BETA)

        opt.zero_grad(set_to_none=True)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            entry = {
                'step':          step,
                'loss':          round(result.loss.item(),          6),
                'accuracy':      round(result.accuracy.item(),      4),
                'reward_margin': round(result.reward_margin.item(), 6),
            }
            training_logs.append(entry)
            print(f"step {step:>5} | loss {entry['loss']:.4f}"
                  f" | acc {entry['accuracy']:.3f}"
                  f" | margin {entry['reward_margin']:.4f}")

    # ── Save checkpoint and training logs ────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'dpo_phase5_training_logs.json', 'w') as f:
        json.dump(training_logs, f, indent=4)

    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size, 'block_size': block_size,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
    }}, str(out_dir / 'model_last.pt'))
    print(f"\nCheckpoint → {out_dir / 'model_last.pt'}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    from rollout import sample_prompts
    prompts = sample_prompts(16)

    print(f"\n--- Phase 5 DPO Evaluation (temp={EVAL_TEMP}  top_k={EVAL_TOP_K}) ---")
    logs: list[dict] = []

    # Load reward model for eval
    from model_reward import RewardModel
    rm_path = _HERE.parent / 'part_7' / 'runs' / 'rm-demo' / 'model_last.pt'
    if not rm_path.exists():
        print(f"WARNING: reward model not found at {rm_path}, skipping eval scoring")
        return

    r_ckpt = torch.load(str(rm_path), map_location=device)
    rm = RewardModel(
        vocab_size  = r_ckpt['config'].get('vocab_size', tok.vocab_size),
        block_size  = r_ckpt['config'].get('block_size', tok.block_size),
        n_layer     = r_ckpt['config'].get('n_layer',    4),
        n_head      = r_ckpt['config'].get('n_head',     4),
        n_embd      = r_ckpt['config'].get('n_embd',   256),
    ).to(device)
    rm.load_state_dict(r_ckpt['model'])
    rm.eval()

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
            'beta':        BETA,
            'lr':          LR,
            'rej_temp':    REJ_TEMP,
            'eval_temp':   EVAL_TEMP,
            'eval_top_k':  EVAL_TOP_K,
        },
        'average_reward': round(avg, 4),
        'samples': logs,
    }

    # Save to part_10/ alongside dpo_phase1_metrics.json
    out_path = Path(__file__).resolve().parent / 'dpo_phase5_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\nPhase 5 avg DPO reward: {avg:.4f}  (Phase 1 baseline: 2.403)")
    print(f"Saved → {out_path.absolute()}")


if __name__ == '__main__':
    main()