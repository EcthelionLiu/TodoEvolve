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
from jinja2 import Template, StrictUndefined

load_dotenv(override=True)

class BaseAgent:
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
        for step_num, step in enumerate(self.agent_fn.memory.steps):
            if isinstance(step, TaskStep):
                continue
            elif isinstance(step, PlanningStep):
                traj = {"name": "plan", "value": step.plan, "think": step.plan_think, "cot_think": step.plan_reasoning}
                trajectory.append(traj)
            elif isinstance(step, SummaryStep):
                traj = {"name": "summary", "value": step.summary, "cot_think": step.summary_reasoning}
                trajectory.append(traj)
            elif isinstance(step, ActionStep):
                safe_tool_calls = step.tool_calls if step.tool_calls is not None else []
                traj = {"name": "action", "tool_calls": [st.dict() for st in safe_tool_calls], "obs": step.observations,
                        "think": step.action_think, "cot_think": step.action_reasoning}
                trajectory.append(traj)
            else:
                raise ValueError("[capture_trajectory] Unknown Step:", step)

        return {
            "agent_trajectory": trajectory,
        }

    def forward(self, task, answer=None, return_json=False, max_retries=3):
        last_error = None
        for _ in range(max_retries):
            try:
                if answer is not None:
                    result = self.agent_fn.run(task, answer=answer)
                else:
                    result = self.agent_fn.run(task)
                if return_json and isinstance(result, str):
                    result = safe_json_loads(result)
                elif not return_json and isinstance(result, dict):
                    result = str(result)
                return {
                    "agent_result": result, **self.capture_trajectory()
                }
            except Exception as e:
                last_error = e
                print(f"[BaseAgent] error: {e}")
                continue
        return {"error": str(last_error)}


class SearchAgent(BaseAgent):
    def __init__(
        self,
        model,
        summary_interval,
        prompts_type=None,
        max_steps=40,
        planning_system="flash_searcher",
        **kwargs
    ):
        super().__init__(model)

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        vector_tool = VectorSimilarityRetrieve(
            memory=None,
            model=model
        )
        
        reasoning_tool = Reasoning()
        process_tool = Process(agent=None)
        end_process_tool = EndProcess(agent=None)
        delete_memory_tool = DeleteMemory(agent=None)
        
        expert_parallel_tool = ExpertParallelTool(model=model, agents=[])
        camv_tool = CAMVTool(model=model)
        
        executor_tool = Executor(agent=None)
        refine_tool = Refine(agent=None)

        if planning_system == "joy_agent":
            import yaml
            import importlib.resources
            
            prompt_path = importlib.resources.files("FlashOAgents.prompts.joy_agent").joinpath("toolcalling_agent.yaml")
            joy_prompts = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            
            base_tools = [web_tool, crawl_tool, reasoning_tool]
            
            pe_worker = ToolCallingAgent(
                model=model, tools=base_tools,
                planning_system="joy_agent",
                prompt_templates=joy_prompts['pe_worker'],
                name="pe_expert",
                summary_interval=max_steps + 1, # Disable periodic re-planning to stay on roadmap
                description="Expert at structured logic and high-reliability reports. Follows Plan-Execute paradigm."
            )
            
            react_workers = [
                ToolCallingAgent(
                    model=model, tools=base_tools,
                    planning_system="joy_agent",
                    prompt_templates=joy_prompts['react_worker'],
                    name=f"react_expert_{i}",
                    summary_interval=max_steps + 1,
                    description="Fast reactive expert for exploratory search. Follows ReAct paradigm."
                ) for i in range(1, 4)
            ]
            
            ensemble_tool = EnsembleTool(pe_worker=pe_worker, react_workers=react_workers)
            vote_tool = VoteTool(model=model)
            
            tools = [ensemble_tool, vote_tool, vector_tool]

            self.agent_fn = ToolCallingAgent(
                model=model, tools=tools,
                planning_system="joy_agent",
                prompt_templates=joy_prompts, 
                summary_interval=summary_interval,
                max_steps=max_steps,
                prompts_type="joy_agent"
            )
            
            self.agent_fn.managed_agents = {w.name: w for w in [pe_worker] + react_workers}
            
            ensemble_tool.agent = self.agent_fn
            vote_tool.agent = self.agent_fn
        
        elif planning_system == "co-sight":
            import yaml
            import importlib.resources
            
            prompt_path = importlib.resources.files("FlashOAgents.prompts.co-sight").joinpath("toolcalling_agent.yaml")
            cosight_prompts = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            
            base_tools = [web_tool, crawl_tool, reasoning_tool]
            
            experts = [
                ToolCallingAgent(
                    model=model, tools=base_tools,
                    planning_system="co-sight",
                    prompt_templates=cosight_prompts.get('expert_internal'),
                    name=f"expert_{i+1}",
                    summary_interval=max_steps + 1,
                    description="Autonomous research expert."
                ) for i in range(4) 
            ]
            
            expert_parallel_tool.agents = experts
            tools = [expert_parallel_tool, camv_tool]

            self.agent_fn = ToolCallingAgent(
                model=model,
                tools=tools,
                summary_interval=summary_interval,
                max_steps=max_steps,
                prompts_type=prompts_type,
                planning_system=planning_system,
                prompt_templates=cosight_prompts
            )
            
        elif planning_system == "flowsearch":
            tools = [executor_tool, refine_tool]
            self.agent_fn = ToolCallingAgent(
                model=model,
                tools=tools,
                summary_interval=summary_interval,
                max_steps=max_steps,
                prompts_type=prompts_type,
                planning_system=planning_system
            )
            # Inject hidden tools for Executor's internal use
            self.agent_fn.tools["web_search"] = web_tool
            self.agent_fn.tools["crawl_page"] = crawl_tool
            self.agent_fn.tools["reasoning"] = reasoning_tool
            
        elif planning_system == "agentorchestra":
            import yaml
            import importlib.resources
            
            prompt_path = importlib.resources.files("FlashOAgents.prompts.agentorchestra").joinpath("toolcalling_agent.yaml")
            orchestra_prompts = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            
            update_tool = UpdatePlanStatus(agent=None)
            check_tool = CheckPlanProgress(agent=None)
            
            # AgentOrchestra uses sub-agents as tools (search, reason)
            tools = [
                web_tool, crawl_tool, reasoning_tool,
                update_tool, check_tool,
            ]
            
            self.agent_fn = ToolCallingAgent(
                model=model,
                tools=tools,
                summary_interval=summary_interval,
                max_steps=max_steps,
                prompts_type="agentorchestra",
                planning_system="agentorchestra",
                prompt_templates=orchestra_prompts
            )
            
            update_tool.agent = self.agent_fn
            check_tool.agent = self.agent_fn
        else:
            tools = [
                web_tool, crawl_tool, vector_tool, reasoning_tool,
                process_tool, end_process_tool, delete_memory_tool
            ]
            self.agent_fn = ToolCallingAgent(
                model=model,
                tools=tools,
                summary_interval=summary_interval,
                max_steps=max_steps,
                prompts_type=prompts_type,
                planning_system=planning_system
            )

        # Manually inject prompt templates to CoSightTool to avoid modifying agents.py
        if planning_system == "co-sight":
            expert_parallel_tool.set_prompt_templates(self.agent_fn.prompt_templates)
            camv_tool.set_prompt_templates(self.agent_fn.prompt_templates)

        # Set memory reference after agent initialization
        vector_tool.memory = self.agent_fn.memory
        process_tool.agent = self.agent_fn
        end_process_tool.agent = self.agent_fn
        delete_memory_tool.agent = self.agent_fn
        executor_tool.agent = self.agent_fn
        refine_tool.agent = self.agent_fn

        # sub memory space for owl planning_system
        if getattr(self.agent_fn, "planning_system", None) == "owl":
            if not hasattr(self.agent_fn, "web_memory") or self.agent_fn.web_memory is None:
                self.agent_fn.web_memory = []
            if not hasattr(self.agent_fn, "reasoning_memory") or self.agent_fn.reasoning_memory is None:
                self.agent_fn.reasoning_memory = []


class MMSearchAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type=None, max_steps=40, planning_system="flash_searcher", **kwargs):
        super().__init__(model)

        from FlashOAgents import VectorSimilarityRetrieve, Reasoning

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        visual_tool = VisualInspectorTool(model, 100000)
        text_tool = TextInspectorTool(model, 100000)
        audio_tool = AudioInspectorTool(model, 100000)
        vector_tool = VectorSimilarityRetrieve(
            memory=None,  # Will be set after ToolCallingAgent initialization
            model=model
        )
        reasoning_tool = Reasoning()
        
        process_tool = Process(agent=None)
        end_process_tool = EndProcess(agent=None)
        delete_memory_tool = DeleteMemory(agent=None)
        # tools = [web_tool, crawl_tool, visual_tool] text or audio tool may not useful during agent execution.
        tools = [web_tool, crawl_tool, visual_tool, text_tool, audio_tool, vector_tool, reasoning_tool,
                 process_tool, end_process_tool, delete_memory_tool]

        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            planning_system=planning_system
        )

        # Set memory reference after agent initialization
        vector_tool.memory = self.agent_fn.memory
        process_tool.agent = self.agent_fn
        end_process_tool.agent = self.agent_fn
        delete_memory_tool.agent = self.agent_fn
        
        # sub memory space for owl planning_system
        if getattr(self.agent_fn, "planning_system", None) == "owl":
            if not hasattr(self.agent_fn, "web_memory") or self.agent_fn.web_memory is None:
                self.agent_fn.web_memory = []
            if not hasattr(self.agent_fn, "reasoning_memory") or self.agent_fn.reasoning_memory is None:
                self.agent_fn.reasoning_memory = []
