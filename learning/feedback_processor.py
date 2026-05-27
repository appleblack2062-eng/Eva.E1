"""Extracts learning signals from task execution results."""

from __future__ import annotations
from typing import Dict, Any, List
from ..core.memory_types import TaskResult, ExecutionMode

class FeedbackProcessor:
    """Analyzes execution outcomes to drive optimization."""
    
    def __init__(self, config):
        self.config = config
    
    def extract_signals(
        self,
        task_meta: Dict,
        execution_result: TaskResult,
        profile: Dict,
        routing_decision: Dict,
    ) -> Dict[str, Any]:
        """Create a feedback signal for the learning system."""
        
        signal = {
            "task_id": task_meta["task_id"],
            "timestamp": task_meta["timestamp"],
            "success": execution_result.success,
            "latency_ms": execution_result.latency_ms,
            "tokens_used": execution_result.tokens_used,
            "mode_used": execution_result.execution_mode.name,
            "fallback_triggered": execution_result.fallback_triggered,
        }
        
        # Calculate efficiency score (0-1)
        if execution_result.success:
            # Lower latency and tokens = higher score
            latency_score = max(0, 1.0 - (execution_result.latency_ms / 5000))
            token_score = max(0, 1.0 - (execution_result.tokens_used / 4000))
            signal["efficiency_score"] = (latency_score + token_score) / 2
        else:
            signal["efficiency_score"] = 0.0
        
        return signal
