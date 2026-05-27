"""Decides how to execute a task: LLM vs Workflow."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from ...core.memory_types import ExecutionMode
from ...config.thresholds import OptimizationThresholds

class RoutingDecision:
    def __init__(self, mode: ExecutionMode, workflow_id: Optional[str], task_pattern: Dict):
        self.mode = mode
        self.workflow_id = workflow_id
        self.task_pattern = task_pattern

class HybridExecutionRouter:
    """Intelligent router using learned strategies."""
    
    def __init__(self, config, strategy_memory, workflow_memory):
        self.config = config
        self.strategy_memory = strategy_memory
        self.workflow_memory = workflow_memory
    
    async def decide_execution_mode(
        self,
        task_description: str,
        task_input: Any,
        similar_tasks: List[Dict],
        context: Optional[Dict],
        thresholds: OptimizationThresholds
    ) -> RoutingDecision:
        
        # 1. Extract Pattern
        pattern = {
            "intent": self._classify_intent(task_description),
            "input_type": type(task_input).__name__,
            "complexity": "low" if len(str(task_input)) < 100 else "high"
        }
        
        # 2. Check Strategy Learner for preferred mode
        available_modes = [ExecutionMode.LLM_ONLY, ExecutionMode.LLM_GUIDED]
        
        # If we have similar tasks, check for existing workflows
        workflow_candidates = []
        if similar_tasks:
            # In a real system, we'd map pattern to workflow IDs via Graph Store
            # For now, we simulate checking if a workflow exists for this pattern
            pass 
            
        # Ask Strategy Learner
        preferred_mode = await self.strategy_memory.decide_mode(
            task_pattern=pattern,
            available_modes=available_modes
        )
        
        # 3. Override if confidence is low or task is new
        if len(similar_tasks) < thresholds.min_task_repetitions_for_synthesis:
            # Not enough data to use workflow, stick to LLM
            return RoutingDecision(ExecutionMode.LLM_ONLY, None, pattern)
        
        # 4. Return Decision
        # If strategy says LLM_GUIDED but we have a workflow, upgrade to WORKFLOW_DRAFT
        if preferred_mode == ExecutionMode.LLM_GUIDED and workflow_candidates:
             return RoutingDecision(ExecutionMode.WORKFLOW_DRAFT, workflow_candidates[0]['id'], pattern)
             
        return RoutingDecision(preferred_mode, None, pattern)

    async def register_workflow_pattern(self, task_pattern: Dict, workflow_id: str, confidence: float, expected_speedup: float):
        """Update router knowledge after successful optimization."""
        # Update strategy learner with new positive example
        await self.strategy_memory.update_policy(
            task_pattern=task_pattern,
            outcome=True,
            latency_ms=100.0 / expected_speedup, # Simulated fast latency
            tokens_used=0,
            mode_used=ExecutionMode.WORKFLOW_COMPILED
        )

    def _classify_intent(self, desc: str) -> str:
        if "filter" in desc.lower(): return "data_filter"
        if "search" in desc.lower(): return "search"
        return "general"
