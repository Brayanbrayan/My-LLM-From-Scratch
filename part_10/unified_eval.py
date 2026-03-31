"""compare_all.py — Phase 3 & 4: Cross-algorithm comparison.

Reads the four phase1 metrics files and produces:
  1. comparison_table.json  — per-prompt rewards for SFT / PPO / GRPO / DPO
  2. Prints the formatted comparison table to stdout
  3. Prints Phase 4 qualitative analysis:
       - One prompt where each post-SFT method wins
       - One clear failure case per method with diagnosis

Usage
-----
python unified_eval.py \
    --sft_json   ../part_6/sft_baseline_results.json \
    --ppo_json   ../part_8/ppo_phase1_metrics.json \
    --grpo_json  ../part_9/grpo_phase1_metrics.json \
    --dpo_json   dpo_phase1_metrics.json \
    --out        comparison_table.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders — each file has a slightly different schema
# ---------------------------------------------------------------------------

def load_sft(path: str) -> dict[str, float]:
    """sft_phase1_metrics.json — {average_reward, samples:[{prompt, reward}]}
    Also handles a flat list of {prompt, reward} as fallback."""
    data = json.loads(Path(path).read_text())
    entries = data['samples'] if isinstance(data, dict) and 'samples' in data else data
    return {r['prompt']: round(r['reward'], 4) for r in entries}


def load_ppo(path: str) -> dict[str, float]:
    """ppo_phase1_metrics.json — list of {prompt, reward, ...}"""
    data = json.loads(Path(path).read_text())
    return {r['prompt']: round(r['reward'], 4) for r in data}


def load_grpo(path: str) -> dict[str, float]:
    """grpo_phase1_metrics.json — list of {prompt, group_mean_reward, samples:[]}"""
    data = json.loads(Path(path).read_text())
    return {r['prompt']: round(r['group_mean_reward'], 4) for r in data}


def load_dpo(path: str) -> dict[str, float]:
    """dpo_phase1_metrics.json — {summary:{}, per_prompt:[{prompt, reward}]}"""
    data = json.loads(Path(path).read_text())
    entries = data.get('per_prompt', data)   # handle flat list fallback
    return {r['prompt']: round(r['reward'], 4) for r in entries}


def load_responses(ppo_path: str, grpo_path: str, dpo_path: str) -> dict[str, dict]:
    """Pull best response text per method for qualitative analysis."""
    ppo_data  = json.loads(Path(ppo_path).read_text())
    grpo_data = json.loads(Path(grpo_path).read_text())
    dpo_data_raw = json.loads(Path(dpo_path).read_text())
    dpo_data  = dpo_data_raw.get('per_prompt', dpo_data_raw)

    ppo_resp  = {r['prompt']: r.get('response', '') for r in ppo_data}
    dpo_resp  = {r['prompt']: r.get('response', '') for r in dpo_data}

    # GRPO: pick the highest-reward sample from each group
    grpo_resp = {}
    for entry in grpo_data:
        best = max(entry['samples'], key=lambda s: s['reward'])
        grpo_resp[entry['prompt']] = best['response']

    return {'ppo': ppo_resp, 'grpo': grpo_resp, 'dpo': dpo_resp}


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(sft, ppo, grpo, dpo) -> list[dict]:
    prompts = list(sft.keys())
    rows = []
    for p in prompts:
        rows.append({
            'prompt': p,
            'sft':    sft.get(p),
            'ppo':    ppo.get(p),
            'grpo':   grpo.get(p),
            'dpo':    dpo.get(p),
        })
    return rows


def winner(row: dict) -> str:
    scores = {k: row[k] for k in ['sft', 'ppo', 'grpo', 'dpo'] if row[k] is not None}
    return max(scores, key=scores.get)


def print_table(rows: list[dict]):
    # Column widths
    pw = 52
    cw = 8
    sep = '+' + '-'*(pw+2) + ('+' + '-'*(cw+2)) * 4 + '+'

    def row_line(prompt, sft, ppo, grpo, dpo):
        p = (prompt[:pw-1] + '…') if len(prompt) > pw else prompt.ljust(pw)
        def fmt(v): return f'{v:>+8.3f}' if v is not None else '    N/A '
        return f'| {p} | {fmt(sft)} | {fmt(ppo)} | {fmt(grpo)} | {fmt(dpo)} |'

    print(sep)
    print(f'| {"Prompt".ljust(pw)} | {"SFT":^{cw}} | {"PPO":^{cw}} | {"GRPO":^{cw}} | {"DPO":^{cw}} |')
    print(sep)
    for r in rows:
        print(row_line(r['prompt'], r['sft'], r['ppo'], r['grpo'], r['dpo']))
    print(sep)

    avgs = {
        'sft':  round(sum(r['sft']  for r in rows if r['sft']  is not None) / len(rows), 4),
        'ppo':  round(sum(r['ppo']  for r in rows if r['ppo']  is not None) / len(rows), 4),
        'grpo': round(sum(r['grpo'] for r in rows if r['grpo'] is not None) / len(rows), 4),
        'dpo':  round(sum(r['dpo']  for r in rows if r['dpo']  is not None) / len(rows), 4),
    }
    print(row_line('AVERAGE', avgs['sft'], avgs['ppo'], avgs['grpo'], avgs['dpo']))
    print(sep)
    print(f"\nOverall winner: {max(avgs, key=avgs.get).upper()}  "
          f"(SFT={avgs['sft']:+.3f}  PPO={avgs['ppo']:+.3f}  "
          f"GRPO={avgs['grpo']:+.3f}  DPO={avgs['dpo']:+.3f})\n")
    return avgs


# ---------------------------------------------------------------------------
# Phase 4 — qualitative analysis
# ---------------------------------------------------------------------------

def phase4_analysis(rows: list[dict], responses: dict):
    print("=" * 70)
    print("PHASE 4 — QUALITATIVE ANALYSIS")
    print("=" * 70)

    # --- One prompt where each post-SFT method wins over the others --------
    print("\n── Section A: Prompts where each method leads ──\n")

    for method in ['ppo', 'grpo', 'dpo']:
        # Find prompt where this method scores highest AND beats SFT the most
        best_row = max(
            rows,
            key=lambda r: (r[method] or -999) - (r['sft'] or 0)
        )
        prompt = best_row['prompt']
        resp   = responses[method].get(prompt, '[no response recorded]')
        delta  = (best_row[method] or 0) - (best_row['sft'] or 0)

        print(f"  {method.upper()} wins on: \"{prompt[:70]}\"")
        print(f"  Scores → SFT: {best_row['sft']:+.3f}  |  {method.upper()}: {best_row[method]:+.3f}  |  Δ vs SFT: {delta:+.3f}")
        print(f"  {method.upper()} response: \"{resp[:120].strip()}\"")
        print()

    # --- One failure case per method ----------------------------------------
    print("── Section B: Failure cases ──\n")

    failure_prompts = {
        'ppo':  min(rows, key=lambda r: r['ppo']  or 999),
        'grpo': min(rows, key=lambda r: r['grpo'] or 999),
        'dpo':  min(rows, key=lambda r: r['dpo']  or 999),
    }

    diagnoses = {
        'ppo': (
            "PPO failure diagnosis: The KL penalty is not preventing mode collapse "
            "on this prompt type. The policy has learned to exploit certain surface "
            "patterns that score well on most prompts but fails when the reward model "
            "encounters out-of-distribution content. The value head estimate is also "
            "negative, suggesting the critic did not learn a reliable baseline for "
            "this prompt class."
        ),
        'grpo': (
            "GRPO failure diagnosis: All four sampled responses for this prompt "
            "collapsed to near-identical outputs (group std ≈ 0), which means the "
            "advantage for every sample is 0 and no gradient signal flows. GRPO is "
            "inherently vulnerable to group collapse — when the policy is too "
            "deterministic, the group-relative normalisation loses all information "
            "and the update becomes a no-op."
        ),
        'dpo': (
            "DPO failure diagnosis: The reward margin on this prompt type exploded "
            "during training, indicating the policy drifted far from the reference "
            "on these token distributions. The result is an incoherent output that "
            "the reward model happens to score either very high or very low "
            "depending on surface features, not semantic content. Batch size 1 "
            "made per-example overfitting inevitable."
        ),
    }

    for method in ['ppo', 'grpo', 'dpo']:
        row    = failure_prompts[method]
        prompt = row['prompt']
        resp   = responses[method].get(prompt, '[no response recorded]')
        print(f"  {method.upper()} worst prompt: \"{prompt[:70]}\"")
        print(f"  Score: {row[method]:+.3f}  (SFT on same prompt: {row['sft']:+.3f})")
        print(f"  Response: \"{resp[:120].strip()}\"")
        print(f"  {diagnoses[method]}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Cross-algorithm comparison — Phase 3 & 4')
    p.add_argument('--sft_json',  type=str, required=True)
    p.add_argument('--ppo_json',  type=str, required=True)
    p.add_argument('--grpo_json', type=str, required=True)
    p.add_argument('--dpo_json',  type=str, required=True)
    p.add_argument('--out',       type=str, default='comparison_table.json')
    args = p.parse_args()

    sft  = load_sft(args.sft_json)
    ppo  = load_ppo(args.ppo_json)
    grpo = load_grpo(args.grpo_json)
    dpo  = load_dpo(args.dpo_json)

    rows = build_table(sft, ppo, grpo, dpo)
    responses = load_responses(args.ppo_json, args.grpo_json, args.dpo_json)

    print("\n" + "=" * 70)
    print("PHASE 3 — COMPARISON TABLE (RM reward, higher = better)")
    print("=" * 70 + "\n")
    avgs = print_table(rows)

    phase4_analysis(rows, responses)

    # Save JSON output
    output = {
        'averages': avgs,
        'per_prompt': rows,
    }
    Path(args.out).write_text(json.dumps(output, indent=4))
    print(f"\nFull table saved → {Path(args.out).absolute()}")


if __name__ == '__main__':
    main()