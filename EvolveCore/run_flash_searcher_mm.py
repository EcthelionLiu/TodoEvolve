#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
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

import os
import random
import argparse
import json
import logging
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from FlashOAgents import OpenAIServerModel
from FlashOAgents import VisualInspectorTool, TextInspectorTool, AudioInspectorTool, get_zip_description, get_single_file_description
from base_agent import MMSearchAgent
from utils import read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)



def process_item(item, model, summary_interval, prompts_type, max_steps, visual_tool, text_tool, audio_tool):

    search_agent = MMSearchAgent(
        model, 
        summary_interval=summary_interval, 
        prompts_type=prompts_type, 
        max_steps=max_steps
    )

    question = item["question"]
    golden_answer = item["answer"]

    if item["file_name"]:
        item["file_name"] = f"data/gaia/validation/" + item["file_name"]
        if ".zip" in item["file_name"]:
            question += "\n\nTo solve the task above, you will have to use these attached files:\n"
            question += get_zip_description(
                item["file_name"], item["question"], visual_tool, text_tool, audio_tool,
            )
        else:
            question += "\n\nTo solve the task above, you will have to use this attached file:"
            question += get_single_file_description(
                item["file_name"], item["question"], visual_tool, text_tool, audio_tool,
            )
    
    try:
        result = search_agent(question)
    except Exception as e:
        logger.error(f"Exception occurred while calling multi_agent: {str(e)}")
        return None

    return {
        "question": question,
        "golden_answer": golden_answer,
        **result,
    }


def main(args):
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    # Use EXECUTE_MODEL, fallback to PLANNING_MODEL
    model_id = os.environ.get("EXECUTE_MODEL") or os.environ.get("PLANNING_MODEL")
    if not model_id:
        raise ValueError("Please set EXECUTE_MODEL or PLANNING_MODEL in .env file")

    model = OpenAIServerModel(
        model_id,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=32768,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )

    visual_tool = VisualInspectorTool(model, 100000)
    text_tool = TextInspectorTool(model, 100000)
    audio_tool = AudioInspectorTool(model, 100000)

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
            data = json.load(f)
    else:
        data = read_jsonl(args.infile)

    if args.sample_num is not None:
        data = data[:args.sample_num]
    
    try:
        out_data = read_jsonl(args.outfile)
    except Exception:
        out_data = []
    done_questions = set([item.get("question") for item in out_data])
    data_to_run = [item for item in data if item.get("question") not in done_questions]
    logger.info(f"Total data: {len(data)}, Completed: {len(done_questions)}, Remaining: {len(data_to_run)}")

    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        with file_lock:
            write_jsonl(args.outfile, [result], "a")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        summary_interval = random.randint(args.summary_interval - 1, args.summary_interval + 1)

        futures = [
            executor.submit(
                process_item, 
                item, 
                model, 
                summary_interval, 
                args.prompts_type, 
                args.max_steps, 
                visual_tool, 
                text_tool, 
                audio_tool
            ) for item in data_to_run
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
                safe_write(result)

    logger.info(f"Processing completed. Newly added: {len(results)}, Total completed: {len(done_questions) + len(results)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal data generation')

    parser.add_argument('--infile', type=str, default="./data/<example.json>", help='input path')
    parser.add_argument('--outfile', type=str, default="./output/<example.jsonl>", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=15, help='Number of concurrency')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')

    args = parser.parse_args()
    
    main(args)
    