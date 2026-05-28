"""Multi-Objective Optimizer for balancing speed, cost, accuracy, and energy."""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class WorkflowVariant:
    workflow_id: str
    metrics: Dict[str, float]
    utility: float = 0.0
    constraints_satisfied: bool = True
    created_at: float = field(default_factory=lambda: __import__('time').time())
    usage_count: int = 0


class ParetoOptimizer:
    def __init__(self, objectives: List[str] = None, user_weights: Dict[str, float] = None):
        self.objectives = objectives or ["latency", "cost", "accuracy", "energy"]
        self.weights = user_weights or {obj: 1.0/len(self.objectives) for obj in self.objectives}
        self.frontier: List[WorkflowVariant] = []
        self.metric_ranges = {
            "latency": (0, 5000),
            "cost": (0, 10),
            "accuracy": (0, 1),
            "energy": (0, 100),
        }
    
    def evaluate_variant(self, variant: WorkflowVariant, metrics: Dict[str, float]) -> bool:
        variant.metrics = metrics
        scores = {obj: self._normalize(metrics.get(obj, 0), obj) for obj in self.objectives}
        utility = sum(self.weights[obj] * scores[obj] for obj in self.objectives)
        variant.utility = utility
        
        dominated = False
        for existing in self.frontier:
            if self._dominates(existing, variant):
                dominated = True
                break
        
        if not dominated:
            self.frontier = [v for v in self.frontier if not self._dominates(variant, v)]
            self.frontier.append(variant)
            return True
        return False
    
    def select_best(self, constraints: Dict[str, float]) -> Optional[WorkflowVariant]:
        feasible = [v for v in self.frontier if self._satisfies_constraints(v, constraints)]
        if not feasible:
            return None
        return max(feasible, key=lambda v: v.utility)
    
    def get_frontier_size(self) -> int:
        return len(self.frontier)
    
    def clear_frontier(self):
        self.frontier.clear()
    
    def _normalize(self, value: float, objective: str) -> float:
        min_val, max_val = self.metric_ranges.get(objective, (0, 1))
        value = max(min_val, min(max_val, value))
        if objective in ["latency", "cost", "energy"]:
            return 1.0 - (value - min_val) / (max_val - min_val + 1e-9)
        return (value - min_val) / (max_val - min_val + 1e-9)
    
    def _dominates(self, v1: WorkflowVariant, v2: WorkflowVariant) -> bool:
        better_in_one = False
        for obj in self.objectives:
            val1, val2 = v1.metrics.get(obj, 0), v2.metrics.get(obj, 0)
            if obj in ["latency", "cost", "energy"]:
                if val1 > val2: return False
                if val1 < val2: better_in_one = True
            else:
                if val1 < val2: return False
                if val1 > val2: better_in_one = True
        return better_in_one
    
    def _satisfies_constraints(self, variant: WorkflowVariant, constraints: Dict[str, float]) -> bool:
        for metric, max_value in constraints.items():
            actual = variant.metrics.get(metric, 0)
            if metric in ["latency", "cost", "energy"] and actual > max_value:
                return False
            elif metric == "accuracy" and actual < max_value:
                return False
        return True
