import json
import torch
import argparse
from pathlib import Path
import sys

# Add parent directory to sys.path to allow imports from other parts
sys.path.append(str(Path(__file__).resolve().parents[1]))

from part_9.policy import PolicyWithValue
from part_9.rollout import RLHFTokenizer, sample_prompts, format_prompt_only
from part_7.model_reward import RewardModel 
from part_6.formatters import Example, format_example

def run_baseline(sft_ckpt: str, rm_ckpt: str, bpe_dir: str, n_prompts: int = 16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

# 1. Load SFT Model
    print(f"--- Loading SFT Model from: {sft_ckpt} ---")
    ckpt = torch.load(sft_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    model = PolicyWithValue(
        cfg.get('vocab_size', tok.vocab_size), 
        cfg.get('block_size', tok.block_size),
        cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)
    ).to(device)

    # REMAPPING LOGIC: Move weights from 'key' to 'lm.key'
    sft_weights = ckpt['model']
    remapped_state_dict = {}
    for key, value in sft_weights.items():
        # Check if the weight is a standard transformer weight
        if key.startswith("tok_emb") or key.startswith("blocks") or key.startswith("head"):
            remapped_state_dict[f"lm.{key}"] = value
        else:
            remapped_state_dict[key] = value

    # Load the remapped weights; strict=False ignores the missing 'val_head' 
    # which is fine because SFT doesn't have a value function anyway.
    model.load_state_dict(remapped_state_dict, strict=False)
    model.eval()
    print("✅ SFT weights remapped and loaded successfully.")

    # 2. Load Reward Model
    print(f"--- Loading Reward Model: {rm_ckpt} ---")
    rckpt = torch.load(rm_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), 
        block_size=rckpt['config'].get('block_size', tok.block_size),
        n_layer=rckpt['config'].get('n_layer', 4), 
        n_head=rckpt['config'].get('n_head', 4), 
        n_embd=rckpt['config'].get('n_embd', 256)
    ).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    # 3. Evaluation Loop
    prompts = sample_prompts(n_prompts)
    results = []
    total_reward = 0

    print(f"\n--- Running Baseline Evaluation ({n_prompts} prompts) ---")
    for i, p in enumerate(prompts):
        prefix = format_prompt_only(p).replace('</s>', '')
        ids = tok.encode(prefix)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            # Use low temperature for a stable baseline
            y = model.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)
            
        resp = tok.decode(y[0].tolist()[len(ids[-tok.block_size:]):])
        
        # Calculate Reward
        text = format_example(Example(p, resp))
        z = torch.tensor([tok.encode(text)[:tok.block_size]], dtype=torch.long, device=device)
        with torch.no_grad():
            reward = rm(z)[0].item()
        
        total_reward += reward
        results.append({
            "prompt": p,
            "response": resp,
            "reward": round(reward, 4)
        })
        print(f"[{i+1}/{n_prompts}] Reward: {reward:.4f}")

    # 4. Save and Summarize
    avg_reward = total_reward / n_prompts
    output_data = {
        "average_reward": avg_reward,
        "samples": results
    }
    
    output_path = Path("sft_baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\n✅ Done! Average Baseline Reward: {avg_reward:.4f}")
    print(f"Results saved to: {output_path.absolute()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_ckpt', type=str, default='runs/sft-demo/model_last.pt')
    parser.add_argument('--rm_ckpt', type=str, default='../part_7/runs/rm-demo/model_last.pt')
    parser.add_argument('--bpe_dir', type=str, default='../part_4/runs/part4-demo/tokenizer')
    args = parser.parse_args()

    run_baseline(args.sft_ckpt, args.rm_ckpt, args.bpe_dir)