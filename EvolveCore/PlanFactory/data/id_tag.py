#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IDAssigner")

def add_ids_to_jsonl(input_file: str, output_file: str, start_id: int = 1):
    """
    Reads a JSONL file and assigns a sequential 'id' field to each record.
    The 'id' field is inserted as the first key for better readability.
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output will be saved to: {output_file}")

    processed_count = 0
    error_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for idx, line in enumerate(f_in, start=start_id):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    original_data = json.loads(line)
                    
                    # Create a new dict with 'id' as the first key
                    # This preserves visual order in the output file
                    new_record = {"id": idx}
                    new_record.update(original_data)
                    
                    f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at line {idx}")
                    error_count += 1
                    
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        return

    logger.info("Processing complete.")
    logger.info(f"Total records processed: {processed_count}")
    if error_count > 0:
        logger.warning(f"Total errors skipped: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add sequential IDs to a JSONL dataset for inference tracking.")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the source JSONL file."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to save the processed JSONL file."
    )
    
    parser.add_argument(
        "--start_id", 
        type=int, 
        default=1, 
        help="The starting number for the ID sequence (default: 1)."
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    add_ids_to_jsonl(args.input_file, args.output_file, args.start_id)