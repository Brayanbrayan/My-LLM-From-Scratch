"""train_dpo.py — DPO training loop for Part 10.

DPO Philosophy
--------------
Unlike PPO/GRPO, DPO requires NO reward model during training.
The preference signal is fully encoded in the (chosen, rejected) pairs:
  - Chosen  (y_w): the dataset's human-written output → always trusted
  - Rejected (y_l): generated on-the-fly by the frozen reference model

The standard assumption "human output > model generation" is well-established
and removes any need for an RM verification step inside the training loop.
The reward model is reserved for post-hoc evaluation only
(see dpo_logger.py and eval_dpo.py).

Pipeline
--------
1. Load tatsu-lab/alpaca via rollout.load_alpaca()
2. For each step:
   a. Sample an example → prompt + chosen text (from dataset)
   b. Generate rejected text from the frozen reference model
   c. Build (input_ids, response_mask) via rollout.build_pair_tensors()
   d. Compute log-probs under policy and reference
   e. Compute DPO loss and backprop (val_head frozen throughout)
3. Log every --log_every steps → <out>/dpo_training_logs.json
4. Save checkpoint            → <out>/model_last.pt

Usage
-----
python train_dpo.py \
    --sft_ckpt ../part_6/runs/sft-demo/model_last.pt \
    --bpe_dir  ../part_4/runs/part4-demo/tokenizer \
    --steps 200 --out runs/dpo-demo
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
for _sub in ['part_10', 'part_8', 'part_6', 'part_4', 'part_3']:
    sys.path.insert(0, str(_HERE.parent / _sub))
sys.path.insert(0, str(_HERE))

from policy   import PolicyWithValue
from rollout  import RLHFTokenizer, format_prompt_only, load_alpaca, build_pair_tensors
from dpo_loss import dpo_loss, get_logps


# ---------------------------------------------------------------------------
# Policy loader (shared pattern from ppo_logger.py / eval_ppo.py)
# ---------------------------------------------------------------------------

def load_policy(ckpt_path: str, device: torch.device, tok: RLHFTokenizer):
    """Load a PolicyWithValue — handles both SFT and PPO checkpoint formats."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='DPO training — Part 10')
    ap.add_argument('--sft_ckpt',  required=True, help='SFT checkpoint (Part 6)')
    ap.add_argument('--bpe_dir',   default=None)
    ap.add_argument('--out',       default='runs/dpo-demo')
    ap.add_argument('--steps',     type=int,   default=200)
    ap.add_argument('--beta',      type=float, default=0.1,  help='DPO temperature β')
    ap.add_argument('--lr',        type=float, default=1e-5)
    ap.add_argument('--resp_len',  type=int,   default=64)
    ap.add_argument('--log_every', type=int,   default=10)
    ap.add_argument('--cpu',       action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    tok = RLHFTokenizer(block_size=256, bpe_dir=args.bpe_dir)

    # ---- Policy (trainable LM, frozen val_head) ---------------------------
    policy, cfg = load_policy(args.sft_ckpt, device, tok)
    policy.train()

    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer    = cfg.get('n_layer', 2)
    n_head     = cfg.get('n_head', 2)
    n_embd     = cfg.get('n_embd', 128)

    for param in policy.val_head.parameters():   # kept for checkpoint compat, not trained
        param.requires_grad_(False)

    # ---- Reference — frozen SFT copy --------------------------------------
    ref, _ = load_policy(args.sft_ckpt, device, tok)
    for param in ref.parameters():
        param.requires_grad_(False)
    ref.eval()

    # ---- Optimiser --------------------------------------------------------
    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999),
    )

    # ---- Dataset ----------------------------------------------------------
    dataset = load_alpaca()
    print(f"Alpaca dataset: {len(dataset)} examples")

    # ---- Training loop ----------------------------------------------------
    training_logs: list[dict] = []

    for step in range(1, args.steps + 1):
        ex = dataset[(step - 1) % len(dataset)]

        instruction = (ex.get('instruction') or '').strip()
        inp         = (ex.get('input')       or '').strip()
        prompt      = f"{instruction}\n{inp}".strip() if inp else instruction
        chosen_text = (ex.get('output')      or '').strip()
        if not chosen_text:
            continue

        prompt_ids = tok.encode(format_prompt_only(prompt).replace('</s>', ''))

        # Generate rejected response on-the-fly from the frozen reference.
        # Clip prompt to block_size before feeding the model; track the
        # clipped length so we correctly slice off only the generated tokens.
        clipped_prompt_ids = prompt_ids[-block_size:]
        with torch.no_grad():
            p_t          = torch.tensor([clipped_prompt_ids], dtype=torch.long, device=device)
            rejected_out = ref.generate(p_t, max_new_tokens=args.resp_len, temperature=0.9, top_k=50)

        # Slice from clipped length, not original prompt length
        rejected_ids  = rejected_out[0].tolist()[len(clipped_prompt_ids):]
        rejected_text = tok.decode(rejected_ids)

        # Build tensor pairs (prompt+response, response_mask)
        c_input, c_mask = build_pair_tensors(prompt_ids, tok.encode(chosen_text),   block_size, device)
        r_input, r_mask = build_pair_tensors(prompt_ids, tok.encode(rejected_text), block_size, device)

        # Log-probs: policy needs grad; reference does not
        policy.train()
        pol_c_lp = get_logps(policy, c_input, c_mask)
        pol_r_lp = get_logps(policy, r_input, r_mask)

        with torch.no_grad():
            ref_c_lp = get_logps(ref, c_input, c_mask)
            ref_r_lp = get_logps(ref, r_input, r_mask)

        result = dpo_loss(pol_c_lp, pol_r_lp, ref_c_lp, ref_r_lp, beta=args.beta)

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

    # ---- Save artefacts ---------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'dpo_training_logs.json', 'w') as f:
        json.dump(training_logs, f, indent=4)
    print(f"\nTraining logs  → {(out_dir / 'dpo_training_logs.json').absolute()}")

    torch.save(
        {'model':  policy.state_dict(),
         'config': {'vocab_size': vocab_size, 'block_size': block_size,
                    'n_layer':    n_layer,    'n_head':     n_head, 'n_embd': n_embd}},
        str(out_dir / 'model_last.pt'),
    )
    print(f"DPO checkpoint → {(out_dir / 'model_last.pt').absolute()}")


if __name__ == '__main__':
    main()