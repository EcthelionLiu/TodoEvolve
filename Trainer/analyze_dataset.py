import json
import os
import numpy as np
import argparse

# Check for tiktoken availability
# We keep this single try-except because it's an optional dependency logic.
try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    USE_TIKTOKEN = True
except ImportError:
    USE_TIKTOKEN = False
    print("[Info] 'tiktoken' not found. Using approximate word count rule.")

def count_tokens(text):
    """
    Counts tokens using tiktoken if available, otherwise estimates based on word count.
    """
    if not text: 
        return 0
    
    if USE_TIKTOKEN:
        return len(ENC.encode(text))
    else:
        # Approximate: 1 word ~ 1.3 tokens
        return int(len(str(text).split()) * 1.3)

def split_reasoning_and_code(full_text):
    """
    Splits the content into Reasoning (CoT) and Code parts based on delimiters.
    """
    full_text = str(full_text)
    delimiters = ["<<PYTHON>>", "<<<PYTHON>>>"]
    
    for delimiter in delimiters:
        if delimiter in full_text:
            parts = full_text.split(delimiter, 1)
            return parts[0].strip(), delimiter + parts[1].strip()
    
    # Simple heuristic: if no delimiter, check for Python keywords
    if "def " in full_text or "import " in full_text:
        return "", full_text
        
    return full_text, ""

def load_dataset_file(file_path):
    """
    Loads data based on file extension (.json or .jsonl).
    Raises errors directly if file is missing or JSON is invalid.
    """
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        # Strict handling: .jsonl reads lines, .json reads whole file
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

def analyze_dataset(file_path, stage_name, data_type="sft"):
    """
    Analyzes token lengths for inputs, reasoning, and code outputs.
    """
    print(f"Processing {stage_name} ({file_path})...")
    
    data = load_dataset_file(file_path)
    if not data:
        return "N/A"

    n_sample = len(data)
    len_inputs = []
    len_reasons = []
    len_codes = []

    for entry in data:
        # 1. Parse Input
        instruction = entry.get('instruction', '')
        inp = entry.get('input', '')
        full_input = f"{instruction}\n{inp}".strip()
        len_inputs.append(count_tokens(full_input))

        # 2. Parse Target (SFT uses 'output', DPO uses 'chosen')
        target_key = 'chosen' if data_type == 'dpo' else 'output'
        target_content = entry.get(target_key, '')

        # 3. Split CoT and Code
        reason_text, code_text = split_reasoning_and_code(target_content)
        len_reasons.append(count_tokens(reason_text))
        len_codes.append(count_tokens(code_text))

    # Calculate statistics
    avg_input = int(np.mean(len_inputs)) if len_inputs else 0
    avg_reason = int(np.mean(len_reasons)) if len_reasons else 0
    avg_code = int(np.mean(len_codes)) if len_codes else 0

    print(f" -> Samples: {n_sample} | Input: {avg_input} | Reason: {avg_reason} | Code: {avg_code}")
    
    # Generate LaTeX Row
    return f"\\textbf{{{stage_name}}} & {n_sample} & {avg_input} & {avg_reason} & {avg_code} \\\\"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token statistics for datasets.")
    parser.add_argument("--sft_file", type=str, default="sft_data_final.json", help="Path to SFT data")
    parser.add_argument("--dpo_file", type=str, default="dpo_data_final.json", help="Path to DPO data")
    
    args = parser.parse_args()

    print("="*60)
    row_sft = analyze_dataset(args.sft_file, "Stage 1: SFT", data_type="sft")
    print("-" * 40)
    row_dpo = analyze_dataset(args.dpo_file, "Stage 2: DPO", data_type="dpo")
    print("="*60)
    
    print("\nLaTeX Table Rows:")
    print(row_sft)
    print(row_dpo)