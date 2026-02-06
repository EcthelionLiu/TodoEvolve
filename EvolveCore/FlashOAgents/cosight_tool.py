from typing import Any, Dict, List, Optional, Tuple
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Template, StrictUndefined

from .tools import Tool
from .search_tools import WebSearchTool, CrawlPageTool
from .memory import ActionStep


JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class CoSightBaseTool(Tool):
    """Base class sharing common utilities for Co-Sight tools."""
    _shared_buffer: List[Dict[str, Any]] = []

    def __init__(self, model, verbose: bool = True, prompt_templates: Optional[Dict] = None):
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.prompt_templates = prompt_templates
        self._logger = logging.getLogger(f"FlashOAgents.{self.name}")

    def set_prompt_templates(self, prompt_templates: Dict[str, Any]):
        self.prompt_templates = prompt_templates

    def _populate_template(self, template: str, variables: Dict[str, Any]) -> str:
        if not template or not isinstance(template, str):
            # Return a safe default if template is None or invalid
            return ""
        compiled_template = Template(template, undefined=StrictUndefined)
        return compiled_template.render(**variables)

    def _log(self, msg: str):
        if not self.verbose:
            return
        self._logger.info(msg)

    def _short(self, s: str, n: int = 220) -> str:
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else s[:n] + "..."

    def _llm(self, prompt: str) -> str:
        self._log(f"[{self.name}.llm] prompt={self._short(prompt, 160)}")
        t0 = time.time()
        msg = [{"role": "user", "content": prompt}]
        resp = self.model(msg)
        dt = time.time() - t0
        out = getattr(resp, "content", str(resp))
        self._log(f"[{self.name}.llm] done in {dt:.2f}s, out={self._short(out, 200)}")
        return out


class ExpertParallelTool(CoSightBaseTool):
    """
    Step 1: Runs multiple ToolCallingAgents (experts) in parallel to gather diverse findings.
    """
    name = "expert_parallel"
    description = "Spawn multiple experts to solve a task in parallel and gather their findings."
    inputs = {
        "task": {"type": "string", "description": "User question/task"},
        "num_expert": {"type": "integer", "description": "Number of experts, e.g. 1-4"},
        "facts_snapshot": {"type": "string", "description": "Already verified global facts", "nullable": True},
        "failure_context": {"type": "string", "description": "Context from previous failed rounds", "nullable": True},
    }
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, model, agents: List[Any], verbose: bool = True, prompt_templates: Optional[Dict] = None):
        super().__init__(model, verbose, prompt_templates)
        self.agents = agents

    def _extract_from_agent(self, agent: Any, task: str, facts_snapshot: Optional[str]) -> Dict[str, Any]:
        """Extract TRSF notes and facts from a finished ToolCallingAgent."""
        # 1. Collect tool records from memory
        tool_records = []
        for step in agent.memory.steps:
            if isinstance(step, ActionStep) and step.tool_calls:
                for tc in step.tool_calls:
                    tool_records.append({
                        "step": step.step_number,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "observation": step.observations
                    })

        # 2. Extract Notes
        notes_template = self.prompt_templates.get("cosight_internal", {}).get("extract_notes", {}).get("prompt")
        if not notes_template:
            # Fallback: use simple extraction without template
            notes = [f"Executed {len(tool_records)} tool calls"]
        else:
            notes_prompt = self._populate_template(notes_template, {
                "task": task,
                "tool_records": json.dumps(tool_records, ensure_ascii=False)
            })
            notes_raw = self._llm(notes_prompt)
            try:
                notes_match = JSON_PATTERN.search(notes_raw)
                notes = json.loads(notes_match.group(0)).get("notes", [notes_raw]) if notes_match else [notes_raw]
            except Exception:
                notes = [notes_raw]

        # 3. Extract Answer & Facts
        expert_template = self.prompt_templates.get("cosight_internal", {}).get("expert", {}).get("prompt")
        if not expert_template:
            # Fallback: return simple answer
            return {
                "expert_id": agent.name,
                "answer": f"Expert {agent.name} completed task analysis",
                "facts_local": [{"key": "completion", "value": "task analyzed", "confidence": 0.7}],
                "notes": notes,
                "tool_records": tool_records
            }

        expert_prompt = self._populate_template(expert_template, {
            "expert_id": agent.name,
            "task": task,
            "facts_snapshot": facts_snapshot or "(none)",
            "notes": json.dumps(notes, ensure_ascii=False)
        })
        final_out = self._llm(expert_prompt)
        try:
            final_match = JSON_PATTERN.search(final_out)
            if final_match:
                final_data = json.loads(final_match.group(0))
                return {
                    "expert_id": agent.name,
                    "answer": final_data.get("answer", ""),
                    "facts_local": final_data.get("facts_local", []),
                    "notes": notes,
                    "tool_records": tool_records
                }
        except Exception:
            pass
        
        return {
            "expert_id": agent.name,
            "answer": final_out.strip(),
            "facts_local": [{"key": "answer", "value": final_out.strip(), "confidence": 0.5}],
            "notes": notes,
            "tool_records": tool_records
        }

    def forward(self, task: str, num_expert: int = 3, facts_snapshot: Optional[str] = None, failure_context: Optional[str] = None, **kwargs) -> str:
        self._log(f"[ExpertParallel] Running {num_expert} experts for task: {self._short(task)}")
        packages: List[Dict[str, Any]] = []
        
        CoSightBaseTool._shared_buffer = []
        
        # Select agents to use
        active_agents = self.agents[:num_expert]
        
        def run_one_agent(agent):
            # Reset agent memory for the new task
            if hasattr(agent, "memory"):
                agent.memory.reset()
            
            # Pass failure context via task or initial state if needed
            augmented_task = task
            if failure_context:
                augmented_task = f"{task}\n\nContext from previous attempts: {failure_context}"
            
            # Run the agent
            agent.run(augmented_task)
            # Extract results
            return self._extract_from_agent(agent, task, facts_snapshot)

        with ThreadPoolExecutor(max_workers=max(1, len(active_agents))) as ex:
            futs = [ex.submit(run_one_agent, agent) for agent in active_agents]
            for f in as_completed(futs):
                try:
                    pkg = f.result()
                    packages.append(pkg)
                    self._log(f"[PROPOSE] Expert {pkg.get('expert_id')} finished.")
                except Exception as e:
                    self._log(f"[PROPOSE] Expert failed: {str(e)}")

        packages.sort(key=lambda x: x.get("expert_id", ""))
        CoSightBaseTool._shared_buffer = packages
        
        simplified_packages = [{"expert_id": p.get("expert_id"), "answer": p.get("answer")} for p in packages]
        return json.dumps(simplified_packages, ensure_ascii=False)


class CAMVTool(CoSightBaseTool):
    """
    Step 2: Conflict-Aware Multi-agent Verification.
    Processes expert outputs: Pruning -> Anchoring -> Conflict Auditing -> Synthesis.
    """
    name = "camv"
    description = "Verify and synthesize findings from multiple experts using CAMV pipeline."
    inputs = {
        "task": {"type": "string", "description": "User question/task"},
        "expert_packages": {"type": "string", "description": "JSON string of findings from expert_parallel"},
        "facts_snapshot": {"type": "string", "description": "Already verified global facts to build upon", "nullable": True},
    }
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, model, theta_default: int = 2, B_max_default: int = 3, verbose: bool = True, prompt_templates: Optional[Dict] = None):
        super().__init__(model, verbose, prompt_templates)
        self.theta_default = theta_default
        self.B_max_default = B_max_default
        self._web = WebSearchTool()
        self._crawl = CrawlPageTool(model=model)
        self.fact_store: List[Dict[str, Any]] = []

    def _fact_snapshot(self, max_items: int = 20) -> str:
        supported = [f for f in self.fact_store if f.get("status") == "supported"]
        supported = supported[-max_items:]
        lines = []
        for x in supported:
            key = x.get('key'); val = x.get('value'); ev = x.get('evidence', {})
            context = f"Evidence: {ev.get('evidence_snippet', '')} | Context: {ev.get('reason', '')}"
            lines.append(f"- {key} = {val} ({context} | Source: {ev.get('url', '')})")
        return "\n".join(lines) or ""

    def _normalize_claims(self, task: str, facts_local: List[Any]) -> List[Dict[str, Any]]:
        template = self.prompt_templates.get("cosight_internal", {}).get("normalization", {}).get("prompt")
        if not template: return []
        prompt = self._populate_template(template, {"task": task, "claims": json.dumps(facts_local, ensure_ascii=False)})
        out = self._llm(prompt)
        try:
            match = re.search(r"\[.*\]", out, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(out)
            return [{k: v for k, v in x.items() if k in ["key", "value", "confidence", "source_url", "source_snippet"]} for x in data if isinstance(x, dict)]
        except (json.JSONDecodeError, AttributeError, KeyError):
            return []

    def _prune_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pruned = []
        neg_keywords = ["unknown", "not found", "no evidence", "not stated", "none", "null"]
        for c in claims:
            k, v = str(c.get("key", "")), str(c.get("value", ""))
            if k and v and not any(x in v.lower() for x in neg_keywords) and float(c.get("confidence", 1.0)) >= 0.3:
                pruned.append(c)
        return pruned

    def _vote(self, norm_pkgs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        by_key = {}
        for pkg in norm_pkgs:
            e_id = pkg.get("expert_id", "Unknown")
            for c in pkg.get("normalized_claims", []):
                k, v = c["key"], c["value"]
                by_key.setdefault(k, {}).setdefault(v, []).append({**c, "expert_id": e_id})
        
        anchors, conflicts = [], []
        for k, val_map in by_key.items():
            if len(val_map) == 1:
                v, infos = next(iter(val_map.items()))
                best = max(infos, key=lambda x: x["confidence"])
                anchors.append({"key": k, "value": v, "support": len(infos), "confidence": sum(i["confidence"] for i in infos)/len(infos), "hint_url": best.get("source_url"), "hint_snippet": best.get("source_snippet")})
            else:
                cands = [{"value": v, "support": len(i), "confidence": sum(x["confidence"] for x in i)/len(i), "hint_url": max(i, key=lambda x: x["confidence"]).get("source_url"), "hint_snippet": max(i, key=lambda x: x["confidence"]).get("source_snippet")} for v, i in val_map.items()]
                conflicts.append({"key": k, "candidates": cands})
        return anchors, conflicts

    def _verify_claim(self, key: str, value: str, task: str, h_url: Optional[str], h_snippet: Optional[str]) -> Dict[str, Any]:
        url, web_obs, page_obs = h_url, "", ""
        if not url:
            template = self.prompt_templates.get("cosight_internal", {}).get("verification_query", {}).get("prompt")
            q = self._llm(self._populate_template(template, {"task": task, "key": key, "value": value})).strip().strip('"')
            web_obs = self._web.forward(query=q[:150])
            m = re.search(r"\]\((https?://[^)]+)\)", web_obs)
            url = m.group(1) if m else None
        
        if url:
            try:
                page_obs = self._crawl.forward(url=url, query=f"Verify: {key}={value}")
            except Exception:
                pass
        
        template = self.prompt_templates.get("cosight_internal", {}).get("judge", {}).get("prompt")
        out = self._llm(self._populate_template(template, {"task": task, "key": key, "value": value, "hint_snippet": h_snippet or "(none)", "web_obs": web_obs, "page_obs": page_obs[:15000]}))
        try:
            match = JSON_PATTERN.search(out)
            j = json.loads(match.group(0)) if match else json.loads(out)
            return {"supported": bool(j.get("supported")), "is_contradicted": bool(j.get("is_contradicted")), "evidence": {"url": j.get("url", url or ""), "snippet": j.get("evidence_snippet", ""), "reason": j.get("reason", "")}}
        except (json.JSONDecodeError, AttributeError):
            return {"supported": False, "is_contradicted": False, "evidence": {"url": url or "", "reason": "judge_failed"}}

    def forward(self, task: str, expert_packages: Any, **kwargs) -> str:
        packages = []
        if CoSightBaseTool._shared_buffer:
            packages = CoSightBaseTool._shared_buffer
        else:
            if isinstance(expert_packages, str):
                try:
                    packages = json.loads(expert_packages)
                except json.JSONDecodeError:
                    return "Error: expert_packages is not valid JSON."
            elif isinstance(expert_packages, (list, dict)):
                packages = expert_packages
            else:
                return f"Error: Unsupported type for expert_packages: {type(expert_packages)}"
        
        if isinstance(packages, dict):
            packages = [packages]
            
        self.fact_store = [] # Initialize for this CAMV call
        norm_pkgs = []
        for pkg in packages:
            # Stage 1: Constraint-Based Pruning
            # Ensure facts_local exists (it won't if using simplified expert_packages string)
            facts = pkg.get("facts_local", [])
            normalized = self._normalize_claims(task, facts)
            pruned = self._prune_claims(normalized)
            pkg_copy = dict(pkg)
            pkg_copy["normalized_claims"] = pruned
            norm_pkgs.append(pkg_copy)

        # Stage 2: Voting & Anchoring
        anchors, conflicts = self._vote(norm_pkgs)
        
        # Stage 3: Iterative Conflict Auditing
        # Prioritize conflicts, then anchors for verification
        checks = []
        # Sort conflicts by importance (sum of support)
        for c in sorted(conflicts, key=lambda x: sum(cand["support"] for cand in x["candidates"]), reverse=True):
            for tv in sorted(c["candidates"], key=lambda x: x["support"], reverse=True):
                checks.append((c["key"], tv["value"], tv.get("hint_url"), tv.get("hint_snippet")))
        
        # Add anchors to verification queue
        for a in sorted(anchors, key=lambda x: x["support"], reverse=True):
            checks.append((a["key"], a["value"], a.get("hint_url"), a.get("hint_snippet")))

        # Budget-limited iterative verification
        to_verify = checks[:self.B_max_default]
        updated_facts = []
        if to_verify:
            self._log(f"[CAMV Stage 3] Auditing {len(to_verify)} points...")
            with ThreadPoolExecutor(max_workers=max(1, len(to_verify))) as executor:
                fut_map = {executor.submit(self._verify_claim, k, v, task, u, s): (k, v) for (k, v, u, s) in to_verify}
                for fut in as_completed(fut_map):
                    k, v = fut_map[fut]
                    try:
                        vr = fut.result()
                        if vr["supported"]:
                            updated_facts.append({"key": k, "value": v, "status": "supported", "evidence": vr["evidence"]})
                        elif vr.get("is_contradicted"):
                            self._log(f"[CAMV Stage 3] Conflict Audit: Claim {k}={v} REJECTED.")
                    except Exception:
                        pass

        self.fact_store = updated_facts
        snap = self._fact_snapshot()
        
        # Stage 4: Integrative Synthesis
        template = self.prompt_templates.get("cosight_internal", {}).get("decision", {}).get("prompt")
        out = self._llm(self._populate_template(template, {"task": task, "supported_snapshot": snap}))
        
        try:
            match = JSON_PATTERN.search(out)
            d = json.loads(match.group(0)) if match else json.loads(out)
            ready = bool(d.get("ready"))
            final_ans = d.get("final_answer", "")
            
            if ready and final_ans:
                self._log("[CAMV Stage 4] Synthesis successful. Returning final answer.")
                return f"FINAL_ANSWER_FOUND: {final_ans}\n\nVerified Evidence:\n{snap}"

            self._log("[CAMV Stage 4] Synthesis incomplete. Triggering Fallback best-effort answer.")
            fallback_template = self.prompt_templates.get("cosight_internal", {}).get("fallback", {}).get("prompt")
            if not fallback_template:
                return f"FINAL_ANSWER_FOUND: I could not verify all details, but here is what I found: {snap}"
                
            fallback_out = self._llm(self._populate_template(fallback_template, {"task": task, "supported_snapshot": snap}))
            return f"FINAL_ANSWER_FOUND: [Best Effort] {fallback_out}\n\nVerified Evidence:\n{snap}"
            
        except (json.JSONDecodeError, AttributeError):
            return f"FINAL_ANSWER_FOUND: Error parsing synthesis result. Partial evidence: {snap}"

__all__ = [
    "ExpertParallelTool",
    "CAMVTool",
]
