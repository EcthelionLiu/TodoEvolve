import json
import os
import math
import random
from glob import glob
import difflib
import argparse

def degrade_rejected_sample(rejected_text):
    """
    Artificially degrades the quality of the rejected sample if it is too similar 
    to the chosen sample. It introduces bad formatting or dismissive responses.
    """
    degraded = rejected_text
    rand_val = random.random()
    
    # Randomly inject bad thought processes if python code blocks exist
    if "<<<PYTHON>>>" in degraded and rand_val < 0.5:
        parts = degraded.split("<<<PYTHON>>>")
        if len(parts[0]) > 50: 
            bad_thoughts = [
                "I will generate the code directly.",
                "Skipping analysis.",
                "### Analysis\n(No detailed analysis provided)"
            ]
            new_pre = random.choice(bad_thoughts) + "\n\n"
            degraded = new_pre + "<<<PYTHON>>>" + parts[1]
            return degraded

    # Randomly mess up formatting tags
    if rand_val < 0.7:
        if random.random() < 0.5:
            degraded = degraded.replace("<<<PYTHON>>>", "[PYTHON]")
            degraded = degraded.replace("<<<END_PYTHON>>>", "[/PYTHON]")
        else:
            degraded = degraded.replace("<<<YAML>>>", "```yaml")
            degraded = degraded.replace("<<<END_YAML>>>", "```")
    else:
        # Comment out executable code to make it non-functional
        degraded = degraded.replace("import ", "# import ")
        degraded = degraded.replace("try:", "# try:")

    return degraded

# ==========================================
# Cognitive Impedance
# ==========================================

def calculate_cognitive_impedance(entry):
    """
    Calculates a score representing the 'cost' or 'impedance' of the model's reasoning.
    Lower scores are better. It considers financial cost, failures, and plan stability.
    """
    metrics = entry.get('metrics', {})
    usage = entry.get('usage', {})
    plan_usage = entry.get('planning_system_usage', {})
    worker_usage = entry.get('worker_usage', {})
    
    cost_total = float(usage.get('total_cost_usd', 0.0)) + 1e-9
    
    n_fail = int(metrics.get('failed_tool_calls', 0))
    
    p_stab = float(metrics.get('plan_stability', 1.0))
    turbulence = max(0.0, 1.0 - p_stab)
    
    c_plan = float(plan_usage.get('cost_usd', 0.0))
    c_work = float(worker_usage.get('cost_usd', 0.0))
    bureaucracy_ratio = (c_plan / c_work) if c_work > 1e-6 else 1.0

    exponent = (0.5 * n_fail) + (2.0 * turbulence) + (0.5 * bureaucracy_ratio)
    
    return cost_total * math.exp(exponent)

def parse_instruction(data):
    """
    Parses the instruction from various formats (List, Dict, String).
    """
    raw_input = data.get('plan_model_input', None)
    if isinstance(raw_input, list):
        contents = [str(m['content']) for m in raw_input if isinstance(m, dict) and 'content' in m]
        return "\n\n".join(contents).strip()
    elif isinstance(raw_input, dict):
        return str(raw_input.get('content', '')).strip()
    return str(raw_input).strip()

def is_correct(entry):
    """
    Checks if the data entry is marked as correct.
    """
    judgement = str(entry.get('judgement', '')).lower()
    return 'correct' in judgement and 'incorrect' not in judgement

def process_pairwise_dpo(input_source, output_file):
    """
    Processes raw JSONL files into a DPO (Direct Preference Optimization) dataset.
    It compares pairs of outputs and determines the 'chosen' and 'rejected' samples
    based on correctness or cognitive impedance scores.
    """
    
    dpo_data = []
    stats = {
        "processed_pairs": 0,
        "kept_correct_vs_incorrect": 0,
        "kept_fancy_battle": 0,
        "skipped_both_incorrect": 0,
        "skipped_same_content": 0,
        "augmented_too_similar": 0 
    }

    # Determine file list
    if os.path.isdir(input_source):
        files = glob(os.path.join(input_source, "*.jsonl"))
    else:
        files = [input_source]
        
    for file_path in files:
        if not os.path.exists(file_path): 
            continue
        print(f" - Processing: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        # Process lines in pairs (Step = 2)
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines): 
                break
            stats["processed_pairs"] += 1
            
            try:
                A = json.loads(lines[i])
                B = json.loads(lines[i+1])
            except: 
                continue

            A_correct = is_correct(A)
            B_correct = is_correct(B)
            winner, loser = None, None
            
            # Logic 1: Correct vs Incorrect
            if A_correct and not B_correct:
                winner, loser = A, B
                stats["kept_correct_vs_incorrect"] += 1
            elif not A_correct and B_correct:
                winner, loser = B, A
                stats["kept_correct_vs_incorrect"] += 1
            # Logic 2: Both Correct (Compare Impedance)
            elif A_correct and B_correct:
                score_A = calculate_cognitive_impedance(A)
                score_B = calculate_cognitive_impedance(B)
                # Lower impedance is better
                if score_A < score_B: 
                    winner, loser = A, B
                else: 
                    winner, loser = B, A
                stats["kept_fancy_battle"] += 1
            else:
                stats["skipped_both_incorrect"] += 1
                continue
                
            if winner and loser:
                instr = parse_instruction(winner)
                chosen = str(winner.get('plan_model_output', '')).strip()
                rejected = str(loser.get('plan_model_output', '')).strip()
                
                if not chosen or not rejected: 
                    continue
                if chosen == rejected:
                    stats["skipped_same_content"] += 1
                    continue
                
                # Check text similarity
                seq = difflib.SequenceMatcher(None, chosen, rejected)
                similarity_ratio = seq.ratio()
                
                # If too similar, artificially degrade the rejected sample
                if similarity_ratio > 0.8:
                    rejected = degrade_rejected_sample(rejected)
                    stats["augmented_too_similar"] += 1
                
                dpo_data.append({
                    "instruction": instr,
                    "input": "",
                    "chosen": chosen,
                    "rejected": rejected
                })

    print(f"\nFinal Statistics:")
    print(f" - Total Pairs Scanned: {stats['processed_pairs']}")
    print(f" - Total DPO Samples Generated: {len(dpo_data)}")
    print(f" - Augmented (Too Similar): {stats['augmented_too_similar']}")
    
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {output_file}")
    except Exception as e:
        print(f"[Error] Failed to save output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw JSONL pairwise data to DPO JSON format.")
    
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to input folder containing .jsonl files or a single .jsonl file."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to save the output .json file."
    )
    
    args = parser.parse_args()
    
    process_pairwise_dpo(args.input_path, args.output_file)