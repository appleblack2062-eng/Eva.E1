"""Tracks performance metrics for optimization decisions."""

from __future__ import annotations
import time
from collections import defaultdict
from typing import Dict, List

class ExecutionProfiler:
    """Aggregates execution statistics."""
    
    def __init__(self, config):
        self.config = config
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
    
    def record_execution(self, task_id: str, mode, latency_ms: float, tokens_used: int, success: bool, workflow_id: str):
        key = mode.name
        self.metrics[f"{key}_latency"].append(latency_ms)
        self.metrics[f"{key}_tokens"].append(tokens_used)
        self.counters[f"{key}_count"] += 1
        self.counters["total_tasks"] += 1
        
        if success:
            self.counters[f"{key}_success"] += 1
            
    async def get_average_latency(self) -> float:
        if not self.metrics.get("total_latency"):
            return 0.0
        # Weighted average of all modes
        total_lat = sum(sum(v) for k, v in self.metrics.items() if "latency" in k)
        total_count = sum(self.counters.get(k, 0) for k in self.counters if "count" in k and "total" not in k)
        return total_lat / total_count if total_count > 0 else 0.0

    async def get_average_tokens(self) -> float:
        # Similar logic for tokens
        return 0.0
