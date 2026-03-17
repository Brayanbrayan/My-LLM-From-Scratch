import torch
import json
import argparse
from pathlib import Path
import sys

# Aligning paths to reach your utilities
sys.path.append(str(Path(__file__).resolve().parents[1]))

from part_9.policy import PolicyWithValue
from part_9.rollout import RLHFTokenizer, sample_prompts, format_prompt_only, model_logprobs
from part_7.model_reward import RewardModel 
from part_6.formatters import Example, format_example

def instrumented_grpo_eval(grpo_ckpt, sft_ckpt, rm_ckpt, bpe_dir, n_prompts=16, group_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    # 1. Load GRPO Policy
    g_ckpt = torch.load(grpo_ckpt, map_location=device)
    cfg = g_ckpt.get('config', {})
    grpo_model = PolicyWithValue(
        cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
        cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)
    ).to(device)
    grpo_model.load_state_dict(g_ckpt['model'])
    grpo_model.eval()

    # 2. Load SFT Reference (The Anchor)
    s_ckpt = torch.load(sft_ckpt, map_location=device)
    sft_model = PolicyWithValue(
        cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
        cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)
    ).to(device)
    
    # Remapping SFT weights
    sft_weights = { (f"lm.{k}" if k.startswith("tok") or k.startswith("block") or k.startswith("head") else k): v 
                   for k, v in s_ckpt['model'].items() }
    sft_model.load_state_dict(sft_weights, strict=False)
    sft_model.eval()

    # 3. Load Reward Model
    r_ckpt = torch.load(rm_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size=r_ckpt['config'].get('vocab_size', tok.vocab_size),
        block_size=r_ckpt['config'].get('block_size', tok.block_size),
        n_layer=r_ckpt['config'].get('n_layer', 4), n_head=r_ckpt['config'].get('n_head', 4), n_embd=r_ckpt['config'].get('n_embd', 256)
    ).to(device)
    rm.load_state_dict(r_ckpt['model'])
    rm.eval()

    prompts = sample_prompts(n_prompts)
    all_logs = []

    print(f"--- Starting GRPO Phase 1 Logging ({n_prompts} prompts, Group Size {group_size}) ---")
    
    for p_idx, p in enumerate(prompts):
        prefix = format_prompt_only(p).replace('</s>', '')
        ids = tok.encode(prefix)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)

        group_data = []
        
        # GRPO generates a group of responses for the SAME prompt
        for g_idx in range(group_size):
            with torch.no_grad():
                out = grpo_model.generate(x, max_new_tokens=64, temperature=0.8)
                resp_ids = out[0].tolist()[len(ids):]
                resp_text = tok.decode(resp_ids)

                # Calculate KL from SFT Anchor
                grpo_lp = model_logprobs(grpo_model, out)
                sft_lp = model_logprobs(sft_model, out)
                kl_val = (grpo_lp[0, len(ids)-1:] - sft_lp[0, len(ids)-1:]).mean().item()

                # Get Reward
                full_text = format_example(Example(p, resp_text))
                z = torch.tensor([tok.encode(full_text)[:tok.block_size]], dtype=torch.long, device=device)
                reward = rm(z)[0].item()

                group_data.append({
                    "response": resp_text,
                    "reward": reward,
                    "kl_divergence": kl_val
                })

        # Calculate Group Advantages (The core of GRPO logic)
        rewards = torch.tensor([d['reward'] for d in group_data])
        mean_r = rewards.mean().item()
        std_r = rewards.std().item() + 1e-6
        
        for i, d in enumerate(group_data):
            d["advantage"] = (d["reward"] - mean_r) / std_r

        all_logs.append({
            "prompt": p,
            "group_mean_reward": round(mean_r, 4),
            "group_std_reward": round(std_r, 4),
            "samples": group_data
        })
        print(f"Prompt {p_idx+1}: Avg Reward {mean_r:.2f} | Std {std_r:.2f}")

    # Save to file
    output_path = Path("grpo_phase1_metrics.json")
    with open(output_path, "w") as f:
        json.dump(all_logs, f, indent=4)
    
    print(f"\n✅ GRPO Phase 1 Logging Complete. Results saved to: {output_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grpo_ckpt', default='runs/grpo-demo/model_last.pt')
    parser.add_argument('--sft_ckpt', default='../part_6/runs/sft-demo/model_last.pt')
    parser.add_argument('--rm_ckpt', default='../part_7/runs/rm-demo/model_last.pt')
    parser.add_argument('--bpe_dir', default='../part_4/runs/part4-demo/tokenizer')
    args = parser.parse_args()

    instrumented_grpo_eval(args.grpo_ckpt, args.sft_ckpt, args.rm_ckpt, args.bpe_dir)