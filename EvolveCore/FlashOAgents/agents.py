#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import importlib
import json
import re
from copy import deepcopy
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
import yaml
from jinja2 import StrictUndefined, Template
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .agent_types import AgentType, handle_agent_output_types
from .tools import FinalAnswerTool, VectorSimilarityRetrieve, Reasoning
from .memory import ActionStep, AgentMemory, PlanningStep, SummaryStep, SystemPromptStep, TaskStep, ToolCall
from .models import (
    ChatMessage,
    MessageRole,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
)
from .tools import Tool
import json_repair
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    parse_json_tool_call,
)


logger = getLogger(__name__)


def get_variable_names(self, template: str) -> Set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    # Use default Undefined instead of StrictUndefined to handle auto-generated templates
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        # Retry with lenient undefined handling for auto-generated templates
        compiled_template = Template(template)
        try:
            return compiled_template.render(**variables)
        except Exception as e2:
            raise Exception(f"Error during jinja template rendering: {type(e2).__name__}: {e2}")

def parse_model_content(content: Union[str, dict]) -> dict:

    if isinstance(content, dict):
        return content
    elif isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}
    else:
        return {"unknown_type": str(content)}

class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_plan (`str`): Initial plan prompt.
    """

    initial_plan: str

class SummaryPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        update_pre_messages (`str`): Progress execution prompt.
        update_post_messages (`str`): Progress execution prompt.
    """

    update_pre_messages: str
    update_post_messages: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        summary ([`~agents.SummaryPromptTemplate`]): Summary prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    summary: SummaryPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(initial_plan=""),
    summary=SummaryPromptTemplate(),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
    """

    def __init__(
            self,
            tools: List[Tool],
            model: Callable[[List[Dict[str, str]]], ChatMessage],
            prompt_templates: Optional[PromptTemplates] = None,
            max_steps: int = 6,
            verbosity_level: LogLevel = LogLevel.INFO,
            grammar: Optional[Dict[str, str]] = None,
            managed_agents: Optional[List] = None,
            summary_interval: Optional[int] = None,
            name: Optional[str] = None,
            description: Optional[str] = None,
            provide_run_summary: bool = False,
            debug: bool = False,
            prompts_type: Optional[str] = "default",
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number: int = 0
        self.grammar = grammar
        self.summary_interval = summary_interval
        self.state = {}
        self.name = name
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.debug = debug
        self.action_trajectory = []
        self.managed_agents = {}

        for tool in tools:
            assert isinstance(tool, Tool), f"This element is not of class Tool: {str(tool)}"
        self.tools = {tool.name: tool for tool in tools}
        self.tools["final_answer"] = FinalAnswerTool()
        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level)
        self.prompts_type = prompts_type
        self.planning = None  # To be initialized in subclasses

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    def initialize_system_prompt(self):
        """To be implemented in child classes"""
        pass

    def write_memory_to_messages(
            self,
            memory_steps: Optional[List[ActionStep]] = None,
            summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in memory_steps if memory_steps else self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def provide_final_answer(self, task: str) -> Tuple[str, str]:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            final_answer = chat_message.content
            final_cot_think = chat_message.reasoning_content
            final_answer_json = json_repair.loads(final_answer)
            final_answer_think, final_answer_res = final_answer_json.get("think", ""), final_answer_json.get("answer", "")
            return final_cot_think, final_answer_think, final_answer_res
        
        except Exception as e:
            return None, None, f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(arguments)
                else:
                    observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(**arguments)
                else:
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                error_msg = (
                    f"Error when executing tool {tool_name} with arguments {arguments}: {type(e).__name__}: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following: '{tool.description}'.\nIt takes inputs: {tool.inputs} and returns output type {tool.output_type}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def run(
            self,
            task: str,
            stream: bool = False,
            reset: bool = True,
            answer: Optional[str] = None,
            images: Optional[List[str]] = None,
            additional_args: Optional[Dict] = None,
    ):
        self.task = task
        self.answer = answer

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, images=images)
        # Outputs are returned only at the end as a string. We only look at the last step
        return deque(self._run(task=self.task, images=images), maxlen=1)[0]

    def _run(self, task: str, images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        pass
    def planning_step(self, task) -> PlanningStep:
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        if self.planning is not None:
            planning_step = self.planning.topology_initialize(task)
            if hasattr(self.model, "get_token_counts"):
                counts = self.model.get_token_counts()
                planning_step.input_tokens = counts.get("input_token_count", 0) or 0
                planning_step.output_tokens = counts.get("output_token_count", 0) or 0
            return planning_step

        # # Fallback to default implementation
        # input_messages = [
        #     {
        #         "role": MessageRole.SYSTEM,
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": populate_template(
        #                     self.prompt_templates["planning"]["initial_plan"],
        #                     variables={
        #                         "tools": self.tools,
        #                     },
        #                 ),
        #             }
        #         ],
        #     },
        # ]
        # task_messages = [{
        #     "role": MessageRole.USER,
        #     "content": [{"type": "text", "text": populate_template(self.prompt_templates["planning"]["task_input"], variables={"task": task})}],
        # }]
        # chat_message_plan: ChatMessage = self.model(input_messages + task_messages)
        # think_content = chat_message_plan.reasoning_content
        # plans = chat_message_plan.content
        # plans_think, plans_answer = "", plans

        # final_plan_redaction = textwrap.dedent(
        #     f"""Here is the plan of action that I will follow to solve the task:\n```\n{plans_answer}\n```\n"""
        # )

        # self.logger.log(
        #     Rule("[bold]Initial plan", style="orange"),
        #     Text(final_plan_redaction),
        #     level=LogLevel.INFO,
        # )

        # self.memory.steps.append(
        #     PlanningStep(
        #         model_input_messages=input_messages,
        #         plan=plans_answer,
        #         plan_think=plans_think,
        #         plan_reasoning=think_content,

        #     )
        # )

        # return PlanningStep(
        #     model_input_messages=input_messages,
        #     plan=plans_answer,
        #     plan_think=plans_think,
        #     plan_reasoning=think_content,
        # )


    def summary_step(self, task, step: int) -> SummaryStep:
        """
        Used periodically by the agent to summary the steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        if self.planning is not None:
            summary_step = self.planning.adaptation(task, step, self.write_memory_to_messages)
            if hasattr(self.model, "get_token_counts"):
                counts = self.model.get_token_counts()
                summary_step.input_tokens = counts.get("input_token_count", 0) or 0
                summary_step.output_tokens = counts.get("output_token_count", 0) or 0
            return summary_step

        # # Fallback to default implementation
        # memory_messages = self.write_memory_to_messages()[1:]

        # update_pre_messages = {
        #     "role": MessageRole.SYSTEM,
        #     "content": [{"type": "text", "text": self.prompt_templates["summary"]["update_pre_messages"]}],
        # }
        # update_post_messages = {
        #     "role": MessageRole.USER,
        #     "content": [{"type": "text", "text": self.prompt_templates["summary"]["update_post_messages"]}],
        # }
        # input_messages = [update_pre_messages] + memory_messages + [update_post_messages]
        # chat_message_summary: ChatMessage = self.model(input_messages)

        # summary_answer = chat_message_summary.content
        # summary_cot_content = chat_message_summary.reasoning_content


        # final_summary_redaction = textwrap.dedent(
        #     f"""
        #     Here is my summary of action to solve the task:
        #     ```
        #     {summary_answer}
        #     ```"""
        # )
        # self.memory.steps.append(
        #     SummaryStep(
        #         model_input_messages=input_messages,
        #         summary=summary_answer,
        #         summary_reasoning=summary_cot_content,
        #     )
        # )
        # self.logger.log(
        #     Rule("[bold]Summary", style="orange"),
        #     Text(final_summary_redaction),
        #     level=LogLevel.INFO,
        # )
        # return SummaryStep(
        #     model_input_messages=input_messages,
        #     summary=summary_answer,
        #     summary_reasoning=summary_cot_content,
        # )


    def to_dict(self) -> Dict[str, Any]:
        """Converts agent into a dictionary."""

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            BASE_BUILTIN_MODULES = [
                "collections",
                "datetime",
                "itertools",
                "math",
                "queue",
                "random",
                "re",
                "stat",
                "statistics",
                "time",
                "unicodedata",
            ]
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": {
                managed_agent.name: managed_agent.__class__.__name__ for managed_agent in self.managed_agents.values()
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "name": self.name,
            "description": self.description,
            "requirements": list(requirements),
        }
        return agent_dict



class ToolCallingAgent(MultiStepAgent):

    def __init__(
            self,
            tools: List[Tool],
            model: Callable[[List[Dict[str, str]]], ChatMessage],
            prompt_templates: Optional[PromptTemplates] = None,
            summary_interval: Optional[int] = None,
            prompts_type: Optional[str] = None,
            planning_system: Optional[str] = "flash_searcher",
            execute_model: Optional[Callable[[List[Dict[str, str]]], ChatMessage]] = None,
            **kwargs,
    ):
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            summary_interval=summary_interval,
            prompts_type=prompts_type,
            **kwargs,
        )
        # Use execute_model for action steps if provided, otherwise fallback to planning model
        self.execute_model = execute_model if execute_model is not None else model
        # If prompts_type is not specified, use planning_system name as default
        if prompts_type is None:
            prompts_type = planning_system

        try:
            prompt_path = importlib.resources.files(f"FlashOAgents.prompts.{prompts_type}").joinpath("toolcalling_agent.yaml")
            self.prompt_templates = prompt_templates or yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise AgentError(f"No prompt file：{prompts_type}/toolcalling_agent.yaml", self.logger)
        except yaml.YAMLError as e:
            raise AgentError(f"Yaml parse error：{e}", self.logger)
        self.summary_interval = summary_interval

        # Initialize planning module based on planning_system parameter
        planning_class_map = {
            "flash_searcher": "FlashSearcherPlanning",
            "oagent": "OAgentPlanning",
            "joy_agent": "JoyAgentPlanning",
            "owl": "OwlPlanning",
            "co-sight": "CosightPlanning",
            "flowsearch": "FlowSearcherPlanning",
            "agentorchestra": "AgentOrchestraPlanning",
            "planner": "PlannerPlanning",
        }

        # Handle dynamic planning systems generated by factory
        if planning_system not in planning_class_map:
            if planning_system.startswith("planner_"):
                planning_class_name = "PlannerPlanning"
            else:
                raise AgentError(f"Unknown planning system: {planning_system}. Available options: {list(planning_class_map.keys())}", self.logger)
        else:
            planning_class_name = planning_class_map[planning_system]

        planning_module = importlib.import_module(f"FlashOAgents.planning.{planning_system}")
        planning_class = getattr(planning_module, planning_class_name)

        self.planning = planning_class(
            model=self.model,
            tools=self.tools,
            prompt_templates=self.prompt_templates,
            memory=self.memory,
            logger=self.logger
        )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools},
        )
        return system_prompt

    def _run(self, task: str, images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        final_answer = None
        self.step_number = 0
        while final_answer is None and self.step_number <= self.max_steps:
            step_start_time = time.time()
            memory_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                if self.step_number == 0:
                    self.planning_step(task)
                    self.step_number += 1
                elif self.summary_interval is not None and self.step_number % self.summary_interval == 0:
                    self.summary_step(
                        task,
                        step=self.step_number,
                    )
                    self.step_number += 1
                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
                final_answer = self.step(memory_step)
            except AgentError as e:
                memory_step.error = e
                raise
            finally:
                memory_step.end_time = time.time()
                memory_step.duration = memory_step.end_time - step_start_time
                self.memory.steps.append(memory_step)
                self.step_number += 1
                yield memory_step

        if final_answer is None and self.step_number > self.max_steps:
            error_message = "Reached max steps."
            step_start_time = time.time()
            cot_think, final_think, final_answer = self.provide_final_answer(task)

            final_memory_step = ActionStep(
                step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
            )

            final_memory_step.action_reasoning = cot_think
            final_memory_step.action_think = final_think
            final_memory_step.action_output = final_answer
            final_memory_step.end_time = time.time()
            final_memory_step.duration = memory_step.end_time - step_start_time
            self.memory.steps.append(final_memory_step)

            yield final_memory_step

        yield handle_agent_output_types(final_answer)

    def reformulate_tool_fuctions(self, tool_list: List[Tool]) -> str:
        json_schema_list = []
        for tool in tool_list:
            required = []
            properties = deepcopy(tool.inputs)
            for key, value in properties.items():
                if value["type"] == "any":
                    value["type"] = "string"
                if not ("nullable" in value and value["nullable"]):
                    required.append(key)
            json_schema_list.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "properties": properties,
                    "required": required,
                }
            })
        return json.dumps(json_schema_list, indent=2, ensure_ascii=False)
    

    def step(self, memory_step: ActionStep, memory_messages=None) -> Union[None, Any]:
        memory_messages = self.write_memory_to_messages() if memory_messages is None else memory_messages
        self.input_messages = memory_messages

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        instruction_message = [{
            "role": MessageRole.USER,
            "content": [{
                "type": "text",
                "text": populate_template(
                    self.prompt_templates["step"]["pre_messages"],
                    variables={
                        "tool_functions_json": self.reformulate_tool_fuctions(list(self.tools.values())),
                        "task": self.task
                    }
                )
            }]
        }]
        
        try:
            # Use execute_model for action steps
            model_message: ChatMessage = self.execute_model(
                memory_messages + instruction_message,
            )
            if hasattr(self.execute_model, "get_token_counts"):
                counts = self.execute_model.get_token_counts()
                memory_step.input_tokens = counts.get("input_token_count", 0) or 0
                memory_step.output_tokens = counts.get("output_token_count", 0) or 0
            
            memory_step.model_output_messages = model_message
            try:
                content_dict = json_repair.loads(model_message.content)
            except Exception as e:
                content_dict = []
                raise Exception(f"Unsupported step output: {type(content_dict)}: {e}")
            
            if isinstance(content_dict, list):
                if "tools" in content_dict[0]:
                    answer_data = content_dict[0]['tools']
                    memory_step.action_think = content_dict[0].get("think", "No 'think' field in response")
                else:
                    answer_data = content_dict
                    memory_step.action_think = "No 'think' field in response"
            elif isinstance(content_dict, dict):
                answer_data = content_dict.get("tools", None)
                memory_step.action_think = content_dict.get("think", "No 'think' field in response")
            else:
                answer_data = "No fuction calling in response"
                memory_step.action_think = "No 'think' field in response"
            
            # Extract tool calls from response
            if isinstance(answer_data, list):
                tool_calls_list = answer_data
            elif isinstance(answer_data, dict):
                tool_calls_list = [answer_data]
            else:
                tool_calls_list = []

            memory_step.tool_calls = []
            final_answer_value = None
            observations = []
            
            # Process each tool call

            self.logger.log(
                Panel(Text(f"Function calling number: {len(tool_calls_list)} calls: {str(tool_calls_list)}")),
                level=LogLevel.INFO,
            )

            # Parallel tool execution. (Please ensure that the tool implement has sufficient concurrency!)
             # ================ Parallel Tool Execution ================== #
            with ThreadPoolExecutor() as executor:
                futures = []
                tool_info_list = []
                
                for idx, tool_call in enumerate(tool_calls_list):
                    tool_name = tool_call.get("name", "")
                    tool_arguments = tool_call.get("arguments", {})
                    tool_call_id = tool_call.get("id", "")
                    
                    tool_call_obj = ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
                    memory_step.tool_calls.append(tool_call_obj)
                    
                    self.logger.log(
                        Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                        level=LogLevel.INFO,
                    )
                    
                    if tool_name == "final_answer":
                        if isinstance(tool_arguments, dict):
                            answer = tool_arguments.get("answer", tool_arguments)
                        else:
                            answer = tool_arguments
                        
                        final_answer_value = answer
                        self.logger.log(
                            Text(f"Final answer: {final_answer_value}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )
                        observations.append(str(final_answer_value))
                        break

                    future = executor.submit(self.execute_tool_call, tool_name, tool_arguments)
                    futures.append((idx, future, tool_name, tool_arguments))
                    tool_info_list.append((idx, tool_name, tool_arguments))
                
                if final_answer_value is not None:
                    memory_step.observations = "\n\n".join(observations) if observations else "No observations"
                    return final_answer_value
                
                if futures:
                    futures.sort(key=lambda x: x[0])
                    
                    for idx, future, tool_name, tool_arguments in futures:
                        try:
                            observation = future.result()
                            updated_information = str(observation).strip()
                            
                            observations.append(
                                f"Results for tool call '{tool_name}' with arguments '{tool_arguments}':\n{updated_information}"
                            )
                            self.logger.log(
                                f"Observations: {updated_information.replace('[', '|')}",
                                level=LogLevel.INFO,
                            )
                        except Exception as e:
                            observation = str(e)
                            self.logger.error(f"Tool execution error: {observation}")
                            observations.append(
                                f"Error for tool call '{tool_name}' with arguments '{tool_arguments}':\n{observation}"
                            )
            memory_step.observations = "\n\n".join(observations) if observations else "No observations"
            # ================ Parallel Tool Execution ================== #

            # ================ Sequence Tool Execution ================== #
            # for tool_call in tool_calls_list:
            #     tool_name = tool_call.get("name", "")
            #     tool_arguments = tool_call.get("arguments", {})
            #     tool_call_id = tool_call.get("id", "")
                
            #     # Create tool call object
            #     tool_call_obj = ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
            #     memory_step.tool_calls.append(tool_call_obj)
                
            #     self.logger.log(
            #         Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            #         level=LogLevel.INFO,
            #     )
            #     if tool_name == "final_answer":
            #         if isinstance(tool_arguments, dict):
            #             answer = tool_arguments.get("answer", tool_arguments)
            #         else:
            #             answer = tool_arguments
                    
            #         final_answer_value = answer
            #         self.logger.log(
            #             Text(f"Final answer: {final_answer_value}", style=f"bold {YELLOW_HEX}"),
            #             level=LogLevel.INFO,
            #         )
                
            #         observations.append(str(final_answer_value))
            #         break

            #     try:
            #         observation = self.execute_tool_call(tool_name, tool_arguments)
            #     except Exception as e:
            #         observation = str(e)
            #         self.logger.error(f"Tool execution error: {str(e)}")

            #     updated_information = str(observation).strip()
                
            #     observations.append(f"Results for tool call '{tool_name}' with arguments '{tool_arguments}':\n{updated_information}")
            #     self.logger.log(
            #         f"Observations: {updated_information.replace('[', '|')}",
            #         level=LogLevel.INFO,
            #     )

            # # Set step observations
            # memory_step.observations = "\n\n".join(observations) if observations else "No observations"
            # ============== Sequence Tool Execution ================ #
            
            # Handle final answer if present
            if final_answer_value is not None:
                return final_answer_value
            
            return None

        except Exception as e:
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e