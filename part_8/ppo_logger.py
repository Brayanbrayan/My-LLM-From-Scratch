import torch
import json
import argparse
from pathlib import Path
import sys

# Ensure we can find the modules across your folder structure
sys.path.append(str(Path(__file__).resolve().parents[1]))

from part_8.policy import PolicyWithValue
from part_8.rollout import RLHFTokenizer, sample_prompts, format_prompt_only, model_logprobs
from part_7.model_reward import RewardModel 
from part_6.formatters import Example, format_example

def instrumented_ppo_eval(ppo_ckpt, sft_ckpt, rm_ckpt, bpe_dir, n_samples=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    # 1. Load PPO Policy (The Actor)
    p_ckpt = torch.load(ppo_ckpt, map_location=device)
    cfg = p_ckpt.get('config', {})
    ppo_model = PolicyWithValue(
        cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
        cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)
    ).to(device)
    ppo_model.load_state_dict(p_ckpt['model'])
    ppo_model.eval()

    # 2. Load SFT Policy (The Anchor/Reference)
    s_ckpt = torch.load(sft_ckpt, map_location=device)
    sft_model = PolicyWithValue(
        cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
        cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)
    ).to(device)
    
    # Remapping SFT weights into the .lm container
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

    prompts = sample_prompts(n_samples)
    logs = []

    print(f"--- Starting Phase 1 Instrumented Logging for PPO ---")
    
    for i, p in enumerate(prompts):
        prefix = format_prompt_only(p).replace('</s>', '')
        ids = tok.encode(prefix)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)

        with torch.no_grad():
            # Generate response using PPO Policy
            out = ppo_model.generate(x, max_new_tokens=64, temperature=0.7)
            resp_ids = out[0].tolist()[len(ids):]
            resp_text = tok.decode(resp_ids)

            # Get Log-Probs for KL calculation
            # model_logprobs returns (B, T-1) log p(x_t | x_{<t})
            ppo_lp = model_logprobs(ppo_model, out)
            sft_lp = model_logprobs(sft_model, out)
            
            # Slice only the response part for KL
            kl_per_token = (ppo_lp[0, len(ids)-1:] - sft_lp[0, len(ids)-1:])
            avg_kl = kl_per_token.mean().item()

            # Get RM Reward
            full_text = format_example(Example(p, resp_text))
            z = torch.tensor([tok.encode(full_text)[:tok.block_size]], dtype=torch.long, device=device)
            reward = rm(z)[0].item()

            # Get Value Estimate from the PPO Critic
            _, values, _ = ppo_model(out)
            # The value of the last token generated is our estimate for the total reward
            value_est = values[0, -1].item()

        logs.append({
            "prompt": p,
            "reward": round(reward, 4),
            "value_estimate": round(value_est, 4),
            "avg_kl_divergence": round(avg_kl, 6),
            "response": resp_text
        })
        print(f"[{i+1}/{n_samples}] Reward: {reward:.2f} | KL: {avg_kl:.4f}")

    # Save Phase 1 Deliverables
    output_file = Path("ppo_phase1_metrics.json")
    with open(output_file, "w") as f:
        json.dump(logs, f, indent=4)
    
    avg_r = sum(l['reward'] for l in logs) / n_samples
    print(f"\n✅ Phase 1 Complete. Avg PPO Reward: {avg_r:.4f}")
    print(f"Detailed logs saved to {output_file.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppo_ckpt', default='runs/ppo-demo/model_last.pt')
    parser.add_argument('--sft_ckpt', default='../part_6/runs/sft-demo/model_last.pt')
    parser.add_argument('--rm_ckpt', default='../part_7/runs/rm-demo/model_last.pt')
    parser.add_argument('--bpe_dir', default='../part_4/runs/part4-demo/tokenizer')
    args = parser.parse_args()

    instrumented_ppo_eval(args.ppo_ckpt, args.sft_ckpt, args.rm_ckpt, args.bpe_dir)