import json
import os
import sys
from glob import glob
import argparse

def convert_to_sft_dataset(input_source, output_file):
    """
    Converts raw data to SFT (Supervised Fine-Tuning) format dataset.
    
    Args:
        input_source (str): Path to the input file or directory containing .jsonl files.
        output_file (str): Path to the output .json file.
    """
    sft_data = []
    correct_count = 0
    skipped_count = 0

    # Check if the input source is a directory or a single file
    if os.path.isdir(input_source):
        files = glob(os.path.join(input_source, "*.jsonl"))
    else:
        files = [input_source]

    print(f"Processing {len(files)} data files...")

    for file_path in files:
        if not os.path.exists(file_path): 
            continue
        
        print(f" - Reading: {os.path.basename(file_path)}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: 
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"[Warning] Failed to parse JSON at line {line_num} in {file_path}")
                    continue

                # Filter logic: Only keep data where judgement contains 'correct' and does not contain 'incorrect'
                judgement = str(data.get('judgement', '')).lower()
                if 'correct' not in judgement or 'incorrect' in judgement:
                    skipped_count += 1
                    continue

                plan_input_obj = data.get('plan_model_input', None)
                instruction = ""
                
                # Handle different input types (list, dict, str) for 'plan_model_input'
                if isinstance(plan_input_obj, list):
                    contents = []
                    for msg in plan_input_obj:
                        if isinstance(msg, dict) and 'content' in msg:
                            contents.append(str(msg['content']))
                    instruction = "\n\n".join(contents)
                    
                elif isinstance(plan_input_obj, dict):
                    instruction = plan_input_obj.get('content', '')
                    
                elif isinstance(plan_input_obj, str):
                    instruction = plan_input_obj
                
                else:
                    print(f"[Warning] Unexpected type {type(plan_input_obj)} for 'plan_model_input' at line {line_num}")
                    continue

                output = data.get('plan_model_output', '')
                
                if not instruction or not output:
                    skipped_count += 1
                    continue

                # Construct the data entry in SFT format
                entry = {
                    "instruction": str(instruction).strip(), 
                    "input": "",                     
                    "output": str(output).strip(),
                }
                
                sft_data.append(entry)
                correct_count += 1

    print(f" - Total Correct Samples Extracted: {correct_count}")
    print(f" - Skipped Samples (Incorrect/Empty): {skipped_count}")
    print(f" - Output File: {output_file}")
    
    try:
        # Create output directory if it does not exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print("Conversion Done Successfully!")
    except Exception as e:
        print(f"[Error] saving output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw JSONL data to SFT JSON format.")
    
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
    
    convert_to_sft_dataset(args.input_path, args.output_file)