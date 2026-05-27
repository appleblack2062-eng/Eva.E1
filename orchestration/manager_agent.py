"""The Manager Agent: Plans, Spawns, and Supervises."""

from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional

try:
    from ..workspace.spatial_context import SpatialContextGenerator
    from ..workspace.filesystem_graph import FileSystemGraph
    from .task_decomposer import TaskDecomposer, TaskStep
    from .worker_factory import WorkerFactory
except ImportError:
    from workspace.spatial_context import SpatialContextGenerator
    from workspace.filesystem_graph import FileSystemGraph
    from orchestration.task_decomposer import TaskDecomposer, TaskStep
    from orchestration.worker_factory import WorkerFactory


class ManagerAgent:
    """
    High-level cognitive agent that orchestrates complex tasks.
    
    The Manager:
    - Decomposes high-level goals into atomic steps
    - Manages execution state and dependencies
    - Spawns specialized workers for each step
    - Coordinates parallel execution where possible
    - Handles errors and recovery strategies
    - Maintains the "Master Plan" throughout execution
    """
    
    def __init__(
        self, 
        agent_id: str, 
        llm_client, 
        fs_graph: FileSystemGraph, 
        config
    ):
        self.agent_id = agent_id
        self.llm = llm_client
        self.fs_graph = fs_graph
        self.config = config
        
        # Initialize components
        self.context_gen = SpatialContextGenerator(fs_graph)
        self.decomposer = TaskDecomposer(llm_client)
        self.worker_factory = WorkerFactory(llm_client, config)
        
        # Runtime state
        self.active_workers: Dict[str, Any] = {}
        self.step_results: Dict[str, Any] = {}
        self.current_goal: Optional[str] = None
    
    async def execute_goal(self, goal: str) -> Dict[str, Any]:
        """
        Main entry point for executing a complex goal.
        
        Args:
            goal: High-level goal description
            
        Returns:
            Dictionary with execution status and results
        """
        print(f"[MANAGER {self.agent_id}] Received Goal: {goal}")
        self.current_goal = goal
        
        try:
            # 1. Get Clean Workspace Map
            workspace_map = self.context_gen.generate_clean_map(goal)
            print(f"[MANAGER] Generated workspace context")
            
            # 2. Decompose Task
            steps = await self.decomposer.decompose(goal, workspace_map)
            print(f"[MANAGER] Decomposed into {len(steps)} steps.")
            
            if not steps:
                return {
                    "status": "failed",
                    "error": "Failed to decompose task into steps",
                    "results": {}
                }
            
            # 3. Sort steps topologically
            sorted_steps = TaskDecomposer.topological_sort(steps)
            
            # 4. Execute Steps (respecting dependencies)
            completed_steps = set()
            failed_steps = set()
            
            while len(completed_steps) + len(failed_steps) < len(sorted_steps):
                # Find ready steps (dependencies met)
                ready_steps = [
                    s for s in sorted_steps 
                    if s.id not in completed_steps 
                    and s.id not in failed_steps
                    and all(
                        dep in completed_steps 
                        for dep in s.dependencies
                    )
                ]
                
                if not ready_steps:
                    if len(completed_steps) + len(failed_steps) < len(sorted_steps):
                        # Deadlock detected
                        remaining = [
                            s for s in sorted_steps 
                            if s.id not in completed_steps and s.id not in failed_steps
                        ]
                        print(f"[MANAGER] Deadlock detected. Remaining steps: {[s.id for s in remaining]}")
                        break
                
                # Spawn Workers for ready steps in parallel
                tasks = []
                step_workers = []
                
                for step in ready_steps:
                    worker = await self._spawn_worker(step, workspace_map)
                    step_workers.append((step, worker))
                    tasks.append(worker.run())
                
                # Wait for all parallel workers
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect Results
                for (step, worker), result in zip(step_workers, results):
                    if isinstance(result, Exception):
                        print(f"[MANAGER] Step {step.id} failed: {result}")
                        self.step_results[step.id] = {
                            "error": str(result),
                            "success": False
                        }
                        failed_steps.add(step.id)
                    else:
                        print(f"[MANAGER] Step {step.id} completed.")
                        self.step_results[step.id] = {
                            "output": result,
                            "success": True
                        }
                        completed_steps.add(step.id)
                        
                        # Update Workspace Graph if files changed
                        if step.expected_output_type == "code":
                            # Assume worker wrote to file, trigger update
                            for f in step.required_files:
                                self.fs_graph.update_node(f)
            
            # Determine overall status
            if failed_steps:
                status = "completed_with_errors"
            elif len(completed_steps) == len(sorted_steps):
                status = "completed"
            else:
                status = "partial"
            
            return {
                "status": status,
                "goal": goal,
                "total_steps": len(sorted_steps),
                "completed_steps": len(completed_steps),
                "failed_steps": len(failed_steps),
                "results": self.step_results
            }
            
        except Exception as e:
            print(f"[MANAGER] Critical error: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": self.step_results
            }
        finally:
            self.current_goal = None
            self.active_workers.clear()

    async def _spawn_worker(
        self, 
        step: TaskStep, 
        global_context: str
    ) -> Any:
        """
        Create a specialized worker with minimal context.
        
        Args:
            step: TaskStep to execute
            global_context: Full workspace context
            
        Returns:
            Configured WorkerInstance
        """
        # Extract ONLY the specific context this worker needs
        specific_context = self.fs_graph.get_subgraph_context(
            step.required_files, 
            depth=1
        )
        
        # Infer role from step description
        role = self._infer_role(step)
        
        # Create Worker
        worker_id = f"worker_{step.id}"
        worker = self.worker_factory.create_worker(
            worker_id=worker_id,
            role=role,
            task_description=step.description,
            context=specific_context,
            output_type=step.expected_output_type
        )
        
        self.active_workers[worker_id] = worker
        return worker

    def _infer_role(self, step: TaskStep) -> str:
        """
        Infer the appropriate specialist role for a step.
        
        Args:
            step: TaskStep to analyze
            
        Returns:
            Role name string
        """
        desc_lower = step.description.lower()
        output_lower = step.expected_output_type.lower()
        
        if "test" in desc_lower or "spec" in desc_lower:
            return "Tester"
        elif "document" in desc_lower or "readme" in desc_lower:
            return "Documenter"
        elif "analyze" in desc_lower or "review" in desc_lower:
            return "Analyst"
        elif "architect" in desc_lower or "design" in desc_lower:
            return "Architect"
        elif "debug" in desc_lower or "fix" in desc_lower:
            return "Debugger"
        elif output_lower == "code" or "write" in desc_lower or "create" in desc_lower:
            return "Coder"
        else:
            return "Generalist"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "agent_id": self.agent_id,
            "current_goal": self.current_goal,
            "active_workers": len(self.active_workers),
            "completed_steps": len([
                r for r in self.step_results.values() 
                if r.get('success', False)
            ]),
            "total_results": len(self.step_results)
        }
    
    def reset(self):
        """Reset manager state for new goal."""
        self.active_workers.clear()
        self.step_results.clear()
        self.current_goal = None
