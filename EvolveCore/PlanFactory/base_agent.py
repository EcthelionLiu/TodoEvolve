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
import sys
# Add project root to sys.path to find utils and FlashOAgents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from utils import safe_json_loads

from FlashOAgents import ToolCallingAgent
from FlashOAgents import ActionStep, PlanningStep, TaskStep, SummaryStep
from FlashOAgents import WebSearchTool, CrawlPageTool, VisualInspectorTool, AudioInspectorTool, TextInspectorTool
from FlashOAgents import (
    VectorSimilarityRetrieve, Reasoning, Process, EndProcess, 
    DeleteMemory, VoteTool, EnsembleTool, Executor, Refine,
    UpdatePlanStatus, CheckPlanProgress
)
from FlashOAgents.cosight_tool import ExpertParallelTool, CAMVTool

from typing import Any, Dict, Optional, List
import re
import json
import textwrap
import time
from jinja2 import Template, StrictUndefined

load_dotenv(override=True)

class PlannerBaseAgent:
    def __init__(self, model):
        self.model = model
        self.agent_fn = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def capture_trajectory(self, ):
        if not hasattr(self, 'agent_fn'):
            raise ValueError("[capture_trajectory] agent_fn is not defined.")
        if not isinstance(self.agent_fn, ToolCallingAgent):
            raise ValueError("[capture_trajectory] agent_fn must be an instance of ToolCallingAgent.")
        
        trajectory = []
        total_input_tokens = 0
        total_output_tokens = 0
        planner_input_tokens = 0
        planner_output_tokens = 0
        worker_input_tokens = 0
        worker_output_tokens = 0

        # Define pricing for different models
        # api price for planning
        PLANNER_PRICE_INPUT = 0.50
        PLANNER_PRICE_OUTPUT = 3.0
        # api price for execution
        WORKER_PRICE_INPUT = 0.25
        WORKER_PRICE_OUTPUT = 2.0

        for step_num, step in enumerate(self.agent_fn.memory.steps):
            step_data = step.dict()
            in_t = step_data.get("input_tokens", 0) or 0
            out_t = step_data.get("output_tokens", 0) or 0
            
            total_input_tokens += in_t
            total_output_tokens += out_t

            if isinstance(step, TaskStep):
                continue
            elif isinstance(step, PlanningStep):
                planner_input_tokens += in_t
                planner_output_tokens += out_t
                traj = {
                    "name": "plan", 
                    "value": step.plan, 
                    "think": step.plan_think, 
                    "cot_think": step.plan_reasoning,
                    "input_messages": step.model_input_messages,
                    "model_output_raw": step.model_output_raw,
                    "input_tokens": in_t,
                    "output_tokens": out_t
                }
                trajectory.append(traj)
            elif isinstance(step, SummaryStep):
                planner_input_tokens += in_t
                planner_output_tokens += out_t
                traj = {
                    "name": "summary", 
                    "value": step.summary, 
                    "cot_think": step.summary_reasoning,
                    "input_messages": step.model_input_messages,
                    "model_output_raw": step.model_output_raw,
                    "input_tokens": in_t,
                    "output_tokens": out_t
                }
                trajectory.append(traj)
            elif isinstance(step, ActionStep):
                worker_input_tokens += in_t
                worker_output_tokens += out_t
                safe_tool_calls = step.tool_calls if step.tool_calls is not None else []
                traj = {
                    "name": "action", 
                    "tool_calls": [st.dict() for st in safe_tool_calls], 
                    "obs": step.observations,
                    "think": step.action_think, 
                    "cot_think": step.action_reasoning,
                    "input_messages": step.model_input_messages,
                    "input_tokens": in_t,
                    "output_tokens": out_t
                }
                trajectory.append(traj)

        planner_cost = (planner_input_tokens / 1_000_000 * PLANNER_PRICE_INPUT) + (planner_output_tokens / 1_000_000 * PLANNER_PRICE_OUTPUT)
        worker_cost = (worker_input_tokens / 1_000_000 * WORKER_PRICE_INPUT) + (worker_output_tokens / 1_000_000 * WORKER_PRICE_OUTPUT)
        total_cost = planner_cost + worker_cost

        return {
            "agent_trajectory": trajectory,
            "usage": {
                "step_count": len(trajectory),
                "total_tokens": total_input_tokens + total_output_tokens,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_cost_usd": round(total_cost, 6),
            },
            "planning_system_usage": {
                "total_tokens": planner_input_tokens + planner_output_tokens,
                "input_tokens": planner_input_tokens,
                "output_tokens": planner_output_tokens,
                "cost_usd": round(planner_cost, 6),
            },
            "worker_usage": {
                "total_tokens": worker_input_tokens + worker_output_tokens,
                "input_tokens": worker_input_tokens,
                "output_tokens": worker_output_tokens,
                "cost_usd": round(worker_cost, 6),
            }
        }

    def forward(self, task, answer=None, return_json=False, max_retries=3):
        last_error = None
        for _ in range(max_retries):
            try:
                start_time = time.time()
                if answer is not None:
                    result = self.agent_fn.run(task, answer=answer)
                else:
                    result = self.agent_fn.run(task)
                    
                end_time = time.time()
                latency = end_time - start_time
                
                if return_json and isinstance(result, str):
                    result = safe_json_loads(result)
                elif not return_json and isinstance(result, dict):
                    result = str(result)
                return {
                    "agent_result": result, **self.capture_trajectory(),
                    "latency": round(latency, 4)
                }
            except Exception as e:
                last_error = e
                print(f"[PlannerBaseAgent] error: {e}")
                continue
        return {"error": str(last_error)}


class SearchAgent(PlannerBaseAgent):
    def __init__(
        self,
        model,
        summary_interval,
        prompts_type=None,
        max_steps=40,
        planning_system=None,
        execute_model=None,
        **kwargs
    ):
        super().__init__(model)

        # Use execute_model for workers if provided, otherwise fallback to planning model
        worker_model = execute_model if execute_model is not None else model

        # Instantiate ALL available tools
        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=worker_model)
        vector_tool = VectorSimilarityRetrieve(memory=None, model=worker_model)
        reasoning_tool = Reasoning()
        process_tool = Process(agent=None)
        end_process_tool = EndProcess(agent=None)
        delete_memory_tool = DeleteMemory(agent=None)

        executor_tool = Executor(agent=None)
        refine_tool = Refine(agent=None)

        update_tool = UpdatePlanStatus(agent=None)
        check_tool = CheckPlanProgress(agent=None)

        vote_tool = VoteTool(model=worker_model)
        
        worker_tools = [web_tool, crawl_tool, reasoning_tool]

        # create specialized workers to match the expected names in EnsembleTool (PE vs ReAct)
        react_worker_1 = ToolCallingAgent(
            model=worker_model,
            tools=worker_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            planning_system=planning_system,
            name="ReAct_Expert_1",
            prompts_type="default"
        )

        react_worker_2 = ToolCallingAgent(
            model=worker_model,
            tools=worker_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            planning_system=planning_system,
            name="ReAct_Expert_2",
            prompts_type="default"
        )
        pe_worker = ToolCallingAgent(
            model=worker_model,
            tools=worker_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            planning_system=planning_system,
            name="PE_Expert",
            prompts_type="default"
        )

        # ExpertParallelTool: 1 ReAct + 1 PE as requested
        expert_parallel_tool = ExpertParallelTool(model=worker_model, agents=[react_worker_1, pe_worker])
        camv_tool = CAMVTool(model=worker_model)
        
        # EnsembleTool: 2 ReActs as requested
        ensemble_tool = EnsembleTool(pe_worker=pe_worker, react_workers=[react_worker_1, react_worker_2])

        # Collect all tools into a single list
        all_tools = [
            web_tool, crawl_tool, reasoning_tool, vector_tool,
            process_tool, end_process_tool, delete_memory_tool,
            executor_tool, refine_tool,
            update_tool, check_tool,
            vote_tool, ensemble_tool,
            expert_parallel_tool, camv_tool
        ]

        # Use model for planning and execute_model for actions
        self.agent_fn = ToolCallingAgent(
            model=model,
            execute_model=execute_model,
            tools=all_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            planning_system=planning_system
        )

        # Post-initialization wiring
        vector_tool.memory = self.agent_fn.memory
        
        process_tool.agent = self.agent_fn
        end_process_tool.agent = self.agent_fn
        delete_memory_tool.agent = self.agent_fn
        
        executor_tool.agent = self.agent_fn
        refine_tool.agent = self.agent_fn
        
        update_tool.agent = self.agent_fn
        check_tool.agent = self.agent_fn
        
        vote_tool.agent = self.agent_fn
        ensemble_tool.agent = self.agent_fn
        
        expert_parallel_tool.set_prompt_templates(self.agent_fn.prompt_templates)
        camv_tool.set_prompt_templates(self.agent_fn.prompt_templates)

        # Initialize sub-memory spaces if needed (Standard practice for some tools)
        if not hasattr(self.agent_fn, "web_memory") or self.agent_fn.web_memory is None:
            self.agent_fn.web_memory = []
        if not hasattr(self.agent_fn, "reasoning_memory") or self.agent_fn.reasoning_memory is None:
            self.agent_fn.reasoning_memory = []

class MMSearchAgent(PlannerBaseAgent):
    def __init__(
        self,
        model,
        summary_interval,
        prompts_type=None,
        max_steps=40,
        planning_system=None,
        execute_model=None,
        **kwargs
    ):
        super().__init__(model)

        # Use execute_model for workers if provided, otherwise fallback to planning model
        worker_model = execute_model if execute_model is not None else model

        # Instantiate ALL available tools including MM tools
        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=worker_model)
        vector_tool = VectorSimilarityRetrieve(memory=None, model=worker_model)
        reasoning_tool = Reasoning()
        process_tool = Process(agent=None)
        end_process_tool = EndProcess(agent=None)
        delete_memory_tool = DeleteMemory(agent=None)

        executor_tool = Executor(agent=None)
        refine_tool = Refine(agent=None)

        update_tool = UpdatePlanStatus(agent=None)
        check_tool = CheckPlanProgress(agent=None)

        vote_tool = VoteTool(model=worker_model)
        
        # Multimedia tools
        visual_tool = VisualInspectorTool(worker_model, 100000)
        text_tool = TextInspectorTool(worker_model, 100000)
        audio_tool = AudioInspectorTool(worker_model, 100000)
        
        worker_tools = [web_tool, crawl_tool, reasoning_tool, visual_tool, text_tool, audio_tool]

        # We create specialized workers to match the expected names in EnsembleTool (PE vs ReAct)
        react_worker_1 = ToolCallingAgent(
            model=worker_model,
            tools=worker_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            planning_system=planning_system,
            name="ReAct_Expert_1",
            prompts_type="default"
        )

        react_worker_2 = ToolCallingAgent(
            model=worker_model,
            tools=worker_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            planning_system=planning_system,
            name="ReAct_Expert_2",
            prompts_type="default"
        )
        pe_worker = ToolCallingAgent(
            model=worker_model,
            tools=worker_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            planning_system=planning_system,
            name="PE_Expert",
            prompts_type="default"
        )

        # ExpertParallelTool: 1 ReAct + 1 PE as requested
        expert_parallel_tool = ExpertParallelTool(model=worker_model, agents=[react_worker_1, pe_worker])
        camv_tool = CAMVTool(model=worker_model)
        
        # EnsembleTool: 1 PE + 1 ReActs as requested
        ensemble_tool = EnsembleTool(pe_worker=pe_worker, react_workers=[react_worker_1, react_worker_2])

        # Collect all tools into a single list 
        all_tools = [
            web_tool, crawl_tool, reasoning_tool, vector_tool,
            process_tool, end_process_tool, delete_memory_tool,
            executor_tool, refine_tool,
            update_tool, check_tool,
            vote_tool, ensemble_tool,
            expert_parallel_tool, camv_tool,
            visual_tool, text_tool, audio_tool 
        ]

        # Initialize ToolCallingAgent with ALL tools
        self.agent_fn = ToolCallingAgent(
            model=model,
            execute_model=execute_model,
            tools=all_tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            planning_system=planning_system
        )

        # Post-initialization wiring
        vector_tool.memory = self.agent_fn.memory
        
        process_tool.agent = self.agent_fn
        end_process_tool.agent = self.agent_fn
        delete_memory_tool.agent = self.agent_fn
        
        executor_tool.agent = self.agent_fn
        refine_tool.agent = self.agent_fn
        
        update_tool.agent = self.agent_fn
        check_tool.agent = self.agent_fn
        
        vote_tool.agent = self.agent_fn
        ensemble_tool.agent = self.agent_fn

        expert_parallel_tool.set_prompt_templates(self.agent_fn.prompt_templates)
        camv_tool.set_prompt_templates(self.agent_fn.prompt_templates)

        # Initialize sub-memory spaces if needed (Standard practice for some tools)
        if not hasattr(self.agent_fn, "web_memory") or self.agent_fn.web_memory is None:
            self.agent_fn.web_memory = []
        if not hasattr(self.agent_fn, "reasoning_memory") or self.agent_fn.reasoning_memory is None:
            self.agent_fn.reasoning_memory = []
