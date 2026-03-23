# Part 10 — DPO
#
#   part_10/
#     orchestrator.py      ← you are here  (run unit tests + optional DPO demo)
#     policy.py            ← copy of part_8/policy.py (unchanged)
#     rollout.py           ← extended copy of part_8/rollout.py (+load_alpaca, +build_pair_tensors)
#     dpo_loss.py          ← DPO objective, reward margin, accuracy
#     train_dpo.py         ← single-GPU DPO training loop
#     eval_dpo.py          ← standalone RM scoring for a DPO checkpoint
#     dpo_logger.py        ← Phase 1 benchmark: reward + KL vs SFT on 16 prompts
#     tests/
#       test_dpo_loss.py
#
# Run from inside `part_10/`:
#   cd part_10
#   python orchestrator.py           # unit tests only
#   python orchestrator.py --demo    # tests + full training + evaluation

import argparse
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def run(cmd: str):
    print(f"\n>>> {cmd}")
    parts = shlex.split(cmd)
    if parts[0] == 'python':
        parts[0] = sys.executable
    res = subprocess.run(parts, cwd=ROOT)
    if res.returncode != 0:
        print(f"FAILED: command exited with code {res.returncode}")
        sys.exit(res.returncode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Run full DPO training + evaluation demo")
    args = p.parse_args()

    # 1) Unit tests — always run
    run("python -m pytest -q tests/test_dpo_loss.py")

    # 2) Optional demo — requires SFT + RM checkpoints from Parts 6 & 7
    if args.demo:
        # Step 1: Train DPO policy
        run("python train_dpo.py"
            " --sft_ckpt ../part_6/runs/sft-demo/model_last.pt"
            " --bpe_dir ../part_4/runs/part4-demo/tokenizer"
            " --steps 200 --out runs/dpo-demo")

        # Step 2: Quick RM score (mirrors eval_ppo.py call in Part 8)
        run("python eval_dpo.py"
            " --policy_ckpt runs/dpo-demo/model_last.pt"
            " --reward_ckpt ../part_7/runs/rm-demo/model_last.pt"
            " --sft_ckpt ../part_6/runs/sft-demo/model_last.pt"
            " --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        # Step 3: Phase 1 benchmark — reward + KL → dpo_phase1_metrics.json
        run("python dpo_logger.py"
            " --dpo_ckpt runs/dpo-demo/model_last.pt"
            " --sft_ckpt ../part_6/runs/sft-demo/model_last.pt"
            " --rm_ckpt ../part_7/runs/rm-demo/model_last.pt"
            " --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 10 checks complete. ✅")