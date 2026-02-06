import json
import argparse
import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def load_dataset(data_path):
    """
    Loads dataset from a JSON or JSONL file.
    Raises FileNotFoundError if the path does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")
    
    print(f"[INFO] Loading dataset from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON file: {e}")

def calculate_statistics(data_list):
    """
    Computes basic statistical metrics (min, mean, max, p95, p99).
    """
    if not data_list:
        return None
    
    arr = np.array(data_list)
    return {
        "count": len(arr),
        "min": int(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": int(np.max(arr)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99))
    }

def print_statistics(stats, label):
    """
    Prints formatted statistics to stdout.
    """
    if not stats:
        print(f"[WARNING] No data found for {label}.")
        return

    print(f"\n--- {label} ---")
    print(f"{'Metric':<10} | {'Value'}")
    print("-" * 25)
    print(f"{'Count':<10} | {stats['count']}")
    print(f"{'Min':<10} | {stats['min']}")
    print(f"{'Mean':<10} | {stats['mean']:.1f}")
    print(f"{'Max':<10} | {stats['max']}")
    print(f"{'P95':<10} | {stats['p95']}")
    print(f"{'P99':<10} | {stats['p99']}")

def analyze_token_distribution(data_path, model_path, mode="sft"):
    """
    Main function to analyze token length distribution for SFT or DPO datasets.
    """
    # 1. Initialize Tokenizer
    print(f"[INFO] Loading tokenizer from: {model_path}")
    # Using trust_remote_code=True is often necessary for custom models (e.g., Qwen, ChatGLM)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. Load Data
    dataset = load_dataset(data_path)
    print(f"[INFO] Total samples loaded: {len(dataset)}")

    effective_lengths = []
    
    # Buffer for chat template overhead (special tokens like <|im_start|>, newlines, etc.)
    # Increasing this ensures safety against OOM or truncation.
    TEMPLATE_BUFFER = 32 

    print(f"[INFO] Starting tokenization (Mode: {mode.upper()})...")
    
    for entry in tqdm(dataset, desc="Processing"):
        # Extract Instruction / Prompt
        # Fallback keys are handled to support various dataset schemas
        instruction = entry.get('instruction', "") or entry.get('prompt', "")
        user_input = entry.get('input', "") or entry.get('query', "")
        
        if user_input:
            full_prompt = f"{instruction}\n{user_input}"
        else:
            full_prompt = instruction

        # Calculate Prompt Tokens
        # add_special_tokens=False ensures we count raw text tokens without BOS/EOS injection twice
        prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        p_len = len(prompt_ids)
        
        current_len = 0

        if mode == "sft":
            # SFT Mode: Length = Prompt + Output
            output_text = entry.get('output', "") or entry.get('response', "")
            o_ids = tokenizer.encode(output_text, add_special_tokens=False)
            current_len = p_len + len(o_ids) + TEMPLATE_BUFFER

        elif mode == "dpo":
            # DPO Mode: Length = Prompt + Max(Chosen, Rejected)
            chosen_text = entry.get('chosen', "")
            rejected_text = entry.get('rejected', "")
            
            c_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
            r_ids = tokenizer.encode(rejected_text, add_special_tokens=False)
            
            max_response_len = max(len(c_ids), len(r_ids))
            current_len = p_len + max_response_len + TEMPLATE_BUFFER
        
        effective_lengths.append(current_len)

    # 3. Calculate and Print Results
    stats = calculate_statistics(effective_lengths)
    print_statistics(stats, f"{mode.upper()} Effective Token Length")

    # 4. Configuration Recommendation
    if stats:
        max_req = stats['max']
        p99_req = stats['p99']
        
        print("\n[RECOMMENDATION] Suggested 'cutoff_len' / 'max_seq_len':")
        
        if max_req <= 2048:
            print(f" -> 2048 (Sufficient for max length: {max_req})")
        elif max_req <= 4096:
            print(f" -> 4096 (Sufficient for max length: {max_req})")
        elif max_req <= 8192:
            print(f" -> 8192 (Sufficient for max length: {max_req})")
        else:
            print(f" -> {p99_req} (Covers 99% of samples)")
            print(f" -> {max_req} (Covers 100% of samples - Warning: Long context detected)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token length requirements for LLM training datasets.")
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="Path to the dataset file (.json or .jsonl)."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Path to the model or tokenizer (e.g., 'gpt2', '/path/to/llama')."
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="sft", 
        choices=["sft", "dpo"], 
        help="Analysis mode: 'sft' (Supervised Fine-Tuning) or 'dpo' (Direct Preference Optimization)."
    )
    
    args = parser.parse_args()
    
    analyze_token_distribution(args.data, args.model, args.mode)