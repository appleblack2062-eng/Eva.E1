"""Orchestrates task execution across LLM, Workflows, and Tools."""

from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, Optional, Type
from ..core.memory_types import ExecutionMode, WorkflowSpec
from ..tools.registry import ToolRegistry
from ..tools.sandbox import SafeExecutionSandbox

class ExecutionOrchestrator:
    """Routes execution to the appropriate engine based on mode."""
    
    def __init__(self, config, llm_client, sandbox: SafeExecutionSandbox, tool_registry: ToolRegistry):
        self.config = config
        self.llm = llm_client
        self.sandbox = sandbox
        self.tools = tool_registry
    
    async def execute_llm_guided(
        self,
        task_id: str,
        description: str,
        input_data: Any,
        output_type: Type,
        context: Optional[Dict[str, Any]],
        tool_registry: ToolRegistry,
    ):
        """Execute task using LLM planning + Tool execution."""
        start = time.time()
        
        # 1. LLM plans steps
        plan_prompt = self._build_planning_prompt(description, input_data, tool_registry.list_tools())
        plan_response = await self.llm.generate(plan_prompt, max_tokens=500)
        
        # 2. Parse plan into tool calls
        tool_calls = self._parse_tool_plan(plan_response.content)
        
        # 3. Execute tools sequentially
        results = []
        for call in tool_calls:
            try:
                tool_result = await self.tools.execute(call["tool_name"], call["arguments"])
                results.append(tool_result)
            except Exception as e:
                return {"success": False, "error": f"Tool execution failed: {str(e)}"}
        
        # 4. LLM synthesizes final answer from tool results
        synthesis_prompt = self._build_synthesis_prompt(description, results)
        final_response = await self.llm.generate(synthesis_prompt, max_tokens=1000)
        
        latency = (time.time() - start) * 1000
        
        return {
            "success": True,
            "output": final_response.content,
            "tokens_used": plan_response.token_count + final_response.token_count,
            "latency_ms": latency
        }
    
    async def run_workflow(
        self,
        workflow: WorkflowSpec,
        input_data: Any,
        output_type: Type,
        context: Optional[Dict[str, Any]],
        mode: ExecutionMode,
    ) -> Any:
        """Execute a compiled or interpreted workflow."""
        
        if mode == ExecutionMode.WORKFLOW_COMPILED and workflow.compiled_code:
            # Execute optimized Python code in sandbox
            return await self.sandbox.execute_code(
                code=workflow.compiled_code,
                input_data=input_data,
                timeout=self.config.max_workflow_execution_time_seconds
            )
        
        else:
            # Interpret pseudo-code steps (slower, for drafts)
            return await self._interpret_workflow_steps(workflow, input_data)
    
    async def _interpret_workflow_steps(self, workflow: WorkflowSpec, input_data: Any) -> Any:
        """Step-by-step interpreter for draft workflows."""
        context = {"input": input_data, "variables": {}}
        
        for step in workflow.steps:
            if step.is_terminal:
                # Resolve return variable
                var_name = step.return_value
                if isinstance(var_name, dict) and "variable_ref" in var_name:
                    return context["variables"].get(var_name["variable_ref"])
                return var_name
            
            # Execute operation
            tool_name = self.config.operation_to_tool_map.get(step.operation)
            if tool_name:
                args = self._resolve_args(step.parameters, context)
                result = await self.tools.execute(tool_name, args)
                
                # Store result if step has assignment (simplified)
                if step.step_number:
                    context["variables"][f"step_{step.step_number}"] = result
        
        return None
    
    def _resolve_args(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        resolved = {}
        for k, v in params.items():
            if isinstance(v, dict) and "variable_ref" in v:
                resolved[k] = context["variables"].get(v["variable_ref"])
            else:
                resolved[k] = v
        return resolved
    
    # Prompt builders (simplified)
    def _build_planning_prompt(self, desc, inp, tools):
        return f"Plan steps for: {desc} using tools: {tools}"
    
    def _parse_tool_plan(self, content):
        # Would parse JSON or structured text from LLM
        return [] 
    
    def _build_synthesis_prompt(self, desc, results):
        return f"Synthesize answer for {desc} from results: {results}"
