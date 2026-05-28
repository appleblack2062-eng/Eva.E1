"""Polyglot DAG Execution Engine with async wave-based runner."""

from __future__ import annotations
import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from pathlib import Path

try:
    from .models import DAGNode, DAGEdge, WorkflowDAG, NodeTelemetry
    from ...tools.sandbox import SafeExecutionSandbox
except ImportError:
    from orchestration.polyglot.models import DAGNode, DAGEdge, WorkflowDAG, NodeTelemetry
    from tools.sandbox import SafeExecutionSandbox


class SchemaValidationError(Exception):
    """Raised when node output doesn't match expected schema."""
    pass


class LLMRouter:
    """Routes LLM requests with adaptive prompt templates and retry logic."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def generate(
        self, 
        template: str, 
        inputs: Dict[str, Any], 
        retry_policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response from LLM with retry logic."""
        max_retries = retry_policy.get('max_retries', 0)
        backoff_ms = retry_policy.get('backoff_ms', 500)
        
        # Format template with inputs
        prompt = template.format(**inputs) if inputs else template
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.llm.generate(prompt)
                
                # Try to parse as JSON if possible
                try:
                    return json.loads(response.content)
                except (json.JSONDecodeError, AttributeError):
                    return {"content": response.content, "tokens_used": getattr(response, 'token_count', 0)}
                    
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(backoff_ms / 1000.0 * (2 ** attempt))
                else:
                    raise RuntimeError(f"LLM generation failed after {max_retries + 1} attempts: {e}")
        
        return {}


class StateStore:
    """Persistent storage for node outputs and execution state."""
    
    def __init__(self, storage_path: str = "./nexus_data/dag_state"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_state: Dict[str, Any] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from current state."""
        return self.current_state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in current state."""
        self.current_state[key] = value
    
    def suspend(self, node_id: str, state: Dict[str, Any]):
        """Suspend execution and persist state for HITL."""
        resume_key = f"suspend_{node_id}_{int(time.time())}"
        state_file = self.storage_path / f"{resume_key}.json"
        state_file.write_text(json.dumps({
            "node_id": node_id,
            "state": state,
            "suspended_at": time.time()
        }, indent=2))
        return resume_key
    
    def resume(self, resume_key: str) -> Optional[Dict[str, Any]]:
        """Resume execution from suspended state."""
        state_file = self.storage_path / f"{resume_key}.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            state_file.unlink()  # Clean up
            return data.get("state")
        return None
    
    def clear(self):
        """Clear current state."""
        self.current_state = {}


class PolyglotDAGRunner:
    """
    Async topological DAG runner with wave-based parallel execution.
    
    Executes nodes in parallel waves based on dependency depth,
    supports multiple runtimes (BASH, PYTHON, LLM, CLI_TOOL, HITL),
    and includes schema validation and error handling.
    """
    
    def __init__(
        self, 
        dag: WorkflowDAG, 
        sandbox: SafeExecutionSandbox, 
        llm_router: Optional[LLMRouter] = None,
        state_store: Optional[StateStore] = None
    ):
        self.dag = dag
        self.sandbox = sandbox
        self.llm_router = llm_router
        self.state_store = state_store or StateStore()
        self.state: Dict[str, Any] = {}
    
    async def run(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the entire DAG starting from initial input.
        
        Args:
            initial_input: Input data for the workflow
            
        Returns:
            Execution result with status and output
        """
        start_time = time.time()
        self.state = {
            "_input": initial_input, 
            "_meta": {
                "start": start_time,
                "dag_id": self.dag.dag_id,
                "version": self.dag.version
            }
        }
        
        # Compute execution waves
        waves = self._topological_waves()
        
        for wave_idx, wave_nodes in enumerate(waves):
            print(f"[DAG Runner] Executing wave {wave_idx} with {len(wave_nodes)} nodes")
            
            # Execute all nodes in this wave in parallel
            tasks = [self._execute_node(node) for node in wave_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for HITL suspension
            for node, result in zip(wave_nodes, results):
                if isinstance(result, Exception):
                    self.state[f"{node.node_id}_error"] = str(result)
                elif isinstance(result, dict) and result.get("_suspend"):
                    return {
                        "status": "SUSPENDED",
                        "suspend_node": node.node_id,
                        "resume_key": self.state_store.suspend(node.node_id, self.state),
                        "reason": result.get("_reason", "HITL_REQUIRED")
                    }
        
        end_time = time.time()
        self.state["_meta"]["end"] = end_time
        self.state["_meta"]["duration_ms"] = (end_time - start_time) * 1000
        
        return {
            "status": "COMPLETED",
            "output": self.state.get("final", {}),
            "meta": self.state.get("_meta", {})
        }
    
    async def _execute_node(self, node: DAGNode) -> Dict[str, Any]:
        """Execute a single node based on its runtime type."""
        start_time = time.time()
        
        # Resolve inputs from parent outputs
        inputs = self._resolve_inputs(node)
        
        try:
            # Execute based on node runtime
            if node.runtime == "BASH":
                out = await self._run_bash(node.payload, inputs, node.timeout_ms)
            elif node.runtime == "PYTHON":
                out = await self._run_python(node.payload, inputs, node.timeout_ms)
            elif node.runtime == "LLM":
                if not self.llm_router:
                    raise RuntimeError("LLM router not configured")
                out = await self.llm_router.generate(
                    node.payload, 
                    inputs, 
                    node.retry_policy
                )
            elif node.runtime == "CLI_TOOL":
                out = await self._run_cli_tool(node.payload, inputs, node.timeout_ms)
            elif node.runtime == "HITL":
                self.state_store.suspend(node.node_id, self.state)
                return {"_suspend": True, "_reason": "HITL_REQUIRED"}
            elif node.runtime == "SUBGRAPH":
                out = await self._run_subgraph(node.payload, inputs)
            else:
                raise ValueError(f"Unknown runtime type: {node.runtime}")
            
            # Schema validation
            if node.output_schema and not self._validate_schema(out, node.output_schema):
                raise SchemaValidationError(f"Output mismatch for node {node.node_id}")
            
            # Store result
            self.state[node.node_id] = out
            
            # Add metadata
            execution_time = (time.time() - start_time) * 1000
            out["_execution_time_ms"] = execution_time
            out["_output_hash"] = hashlib.sha256(
                json.dumps(out, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            return out
            
        except Exception as e:
            # Handle retries
            max_retries = node.retry_policy.get('max_retries', 0)
            if max_retries > 0:
                return await self._retry_node(node, inputs, e, max_retries)
            else:
                self.state[f"{node.node_id}_error"] = str(e)
                raise
    
    def _resolve_inputs(self, node: DAGNode) -> Dict[str, Any]:
        """Resolve node inputs from parent node outputs."""
        inputs = {}
        
        for parent_id, local_var in node.input_mapping.items():
            parent_output = self.state.get(parent_id, {})
            # Extract the specific value if it's a nested key
            if '.' in local_var:
                keys = local_var.split('.')
                value = parent_output
                for key in keys:
                    if isinstance(value, dict):
                        value = value.get(key, {})
                    else:
                        value = {}
                        break
                inputs[local_var.replace('.', '_')] = value
            else:
                inputs[local_var] = parent_output
        
        # Also include global input
        inputs["_input"] = self.state.get("_input", {})
        
        return inputs
    
    async def _run_bash(self, script: str, inputs: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        """Execute bash script with provided inputs."""
        # Prepare environment variables from inputs
        env_vars = {k.upper(): str(v) for k, v in inputs.items() if isinstance(v, (str, int, float))}
        
        # Create temp script file
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\n")
            # Export input variables
            for k, v in env_vars.items():
                f.write(f"export {k}={v}\n")
            f.write(script)
            script_path = f.name
        
        try:
            loop = asyncio.get_event_loop()
            proc = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ['bash', script_path],
                        capture_output=True,
                        text=True,
                        timeout=timeout_ms / 1000.0,
                        env={**env_vars}
                    )
                ),
                timeout=timeout_ms / 1000.0
            )
            
            if proc.returncode != 0:
                raise RuntimeError(f"Bash script failed: {proc.stderr}")
            
            # Try to parse stdout as JSON
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return {"stdout": proc.stdout, "stderr": proc.stderr}
                
        finally:
            import os
            os.unlink(script_path)
    
    async def _run_python(self, code: str, inputs: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        """Execute Python code in sandbox."""
        # Wrap code to receive inputs and return outputs
        wrapped_code = f"""
input_data = {json.dumps(inputs)}
{code}
# Assume result is stored in 'result' variable
import json
print(json.dumps(result if 'result' in locals() else None))
"""
        
        try:
            exec_result = await asyncio.wait_for(
                self.sandbox.execute_code(
                    code=wrapped_code,
                    input_data=inputs,
                    timeout_seconds=timeout_ms / 1000.0
                ),
                timeout=timeout_ms / 1000.0
            )
            return exec_result if isinstance(exec_result, dict) else {"result": exec_result}
        except asyncio.TimeoutError:
            raise RuntimeError(f"Python execution timed out after {timeout_ms}ms")
    
    async def _run_cli_tool(self, tool_cmd: str, inputs: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        """Execute CLI tool with argument mapping."""
        import subprocess
        
        # Simple command parsing and argument substitution
        cmd_parts = tool_cmd.split()
        resolved_cmd = []
        for part in cmd_parts:
            for key, value in inputs.items():
                part = part.replace(f"${{{key}}}", str(value))
            resolved_cmd.append(part)
        
        try:
            loop = asyncio.get_event_loop()
            proc = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        resolved_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_ms / 1000.0
                    )
                ),
                timeout=timeout_ms / 1000.0
            )
            
            if proc.returncode != 0:
                raise RuntimeError(f"CLI tool failed: {proc.stderr}")
            
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return {"stdout": proc.stdout}
                
        except asyncio.TimeoutError:
            raise RuntimeError(f"CLI tool timed out after {timeout_ms}ms")
    
    async def _run_subgraph(self, subgraph_payload: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a subgraph (nested DAG)."""
        # For now, treat as JSON-encoded subgraph
        try:
            subgraph_data = json.loads(subgraph_payload)
            sub_dag = WorkflowDAG(**subgraph_data)
            sub_runner = PolyglotDAGRunner(
                dag=sub_dag,
                sandbox=self.sandbox,
                llm_router=self.llm_router,
                state_store=self.state_store
            )
            return await sub_runner.run(inputs)
        except Exception as e:
            raise RuntimeError(f"Subgraph execution failed: {e}")
    
    def _validate_schema(self, output: Any, schema: Dict[str, Any]) -> bool:
        """Validate output against expected schema."""
        if not schema:
            return True
        
        # Simple schema validation (type checking)
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(output, dict):
            return False
        elif expected_type == "array" and not isinstance(output, list):
            return False
        elif expected_type == "string" and not isinstance(output, str):
            return False
        elif expected_type == "number" and not isinstance(output, (int, float)):
            return False
        elif expected_type == "boolean" and not isinstance(output, bool):
            return False
        
        # Check required properties
        required = schema.get("required", [])
        if isinstance(output, dict):
            for prop in required:
                if prop not in output:
                    return False
        
        return True
    
    async def _retry_node(
        self, 
        node: DAGNode, 
        inputs: Dict[str, Any], 
        error: Exception,
        max_retries: int
    ) -> Dict[str, Any]:
        """Retry node execution with exponential backoff."""
        backoff_ms = node.retry_policy.get('backoff_ms', 500)
        
        for attempt in range(max_retries):
            print(f"[DAG Runner] Retrying node {node.node_id}, attempt {attempt + 1}/{max_retries}")
            await asyncio.sleep(backoff_ms / 1000.0 * (2 ** attempt))
            
            try:
                return await self._execute_node(node)
            except Exception as retry_error:
                if attempt == max_retries - 1:
                    self.state[f"{node.node_id}_error"] = f"Failed after {max_retries} retries: {retry_error}"
                    raise
                continue
        
        raise error
    
    def _topological_waves(self) -> List[List[DAGNode]]:
        """
        Group nodes into parallel execution waves using Kahn's algorithm.
        
        Returns:
            List of waves, where each wave contains nodes that can execute in parallel
        """
        # Build adjacency list and in-degree count
        in_degree: Dict[str, int] = {node.node_id: 0 for node in self.dag.nodes}
        adj: Dict[str, List[str]] = defaultdict(list)
        
        for edge in self.dag.edges:
            in_degree[edge.target_id] += 1
            adj[edge.source_id].append(edge.target_id)
        
        # Find all nodes with no dependencies (first wave)
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        waves: List[List[DAGNode]] = []
        
        node_map = {node.node_id: node for node in self.dag.nodes}
        
        while queue:
            # Current wave
            wave_ids = list(queue)
            wave_nodes = [node_map[nid] for nid in wave_ids if nid in node_map]
            waves.append(wave_nodes)
            
            # Find next wave
            next_queue = []
            for node_id in queue:
                for neighbor in adj[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue
        
        return waves
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about the executed DAG."""
        meta = self.state.get("_meta", {})
        return {
            "dag_id": self.dag.dag_id,
            "total_nodes": len(self.dag.nodes),
            "total_waves": len(self._topological_waves()),
            "duration_ms": meta.get("duration_ms", 0),
            "nodes_executed": len([k for k in self.state.keys() if not k.startswith("_")]),
            "errors": len([k for k in self.state.keys() if k.endswith("_error")])
        }
