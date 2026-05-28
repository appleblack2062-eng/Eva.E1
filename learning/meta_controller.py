"""Meta-Strategy Controller for high-level optimization decisions.

This module implements a hybrid heuristic + contextual bandit system
for making meta-decisions about when to optimize, wait, deprecate, or retrain.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import random


@dataclass
class PendingUpdate:
    """Pending bandit update with context and action."""
    context: np.ndarray
    action: int
    timestamp: float


class ContextualBandit:
    """Simple contextual bandit for action selection."""
    
    def __init__(self, context_dim: int, action_dim: int):
        self.context_dim = context_dim
        self.action_dim = action_dim
        
        # Linear model: weights[action] = context_dim vector
        self.weights = np.zeros((action_dim, context_dim))
        self.covariance = [np.eye(context_dim) for _ in range(action_dim)]
        self.rewards_sum = np.zeros(action_dim)
        self.action_counts = np.zeros(action_dim)
        
        # Pending updates for delayed reward
        self.pending_updates: List[PendingUpdate] = []
    
    def select_action(self, context: np.ndarray) -> int:
        """Select action using Thompson Sampling or UCB."""
        if np.all(self.action_counts == 0):
            # Random exploration initially
            return random.randint(0, self.action_dim - 1)
        
        # Compute UCB scores
        ucb_scores = np.zeros(self.action_dim)
        
        for action in range(self.action_dim):
            if self.action_counts[action] == 0:
                ucb_scores[action] = float('inf')
            else:
                mean_reward = self.rewards_sum[action] / self.action_counts[action]
                exploration_bonus = np.sqrt(2 * np.log(sum(self.action_counts)) / self.action_counts[action])
                ucb_scores[action] = mean_reward + exploration_bonus
        
        return int(np.argmax(ucb_scores))
    
    def register_pending(self, context: np.ndarray, action: int):
        """Register pending update for delayed reward."""
        import time
        self.pending_updates.append(PendingUpdate(
            context=context.copy(),
            action=action,
            timestamp=time.time()
        ))
    
    def update(self, context: np.ndarray, action: int, reward: float):
        """Update bandit model with observed reward."""
        # Update statistics
        self.rewards_sum[action] += reward
        self.action_counts[action] += 1
        
        # Update linear model (simplified Ridge regression)
        self.covariance[action] += np.outer(context, context)
        self.weights[action] += context * reward
    
    def get_pending_updates(self) -> List[PendingUpdate]:
        """Get all pending updates."""
        return self.pending_updates
    
    def clear_pending(self):
        """Clear pending updates after processing."""
        self.pending_updates.clear()


class MetaStrategyController:
    """Heuristic + Bandit hybrid for meta-decisions."""
    
    ACTIONS = ["wait", "optimize", "deprecate", "retrain"]
    
    def __init__(self, config):
        self.config = config
        self.bandit = ContextualBandit(context_dim=16, action_dim=4)
        
        # Heuristic thresholds
        self.heuristics = {
            "min_samples_for_opt": getattr(config, 'min_task_repetitions_for_synthesis', 3),
            "max_cost_before_opt": 10.0,  # USD
            "staleness_threshold_hours": 72,
            "error_rate_threshold": 0.15,
        }
        
        # Override from config if available
        if hasattr(config, 'meta_controller_config'):
            self.heuristics.update(config.meta_controller_config)
    
    def decide_optimization_action(
        self, 
        pattern_stats: Dict[str, Any], 
        system_state: Dict[str, Any]
    ) -> str:
        """Return action: 'optimize', 'wait', 'deprecate', 'retrain'."""
        
        # Heuristic filters first (fast path)
        if pattern_stats.get("sample_count", 0) < self.heuristics["min_samples_for_opt"]:
            return "wait"
        
        if pattern_stats.get("last_used_hours_ago", 0) > self.heuristics["staleness_threshold_hours"]:
            return "deprecate"
        
        if system_state.get("error_rate_recent", 0) > self.heuristics["error_rate_threshold"]:
            return "retrain"
        
        # Bandit for nuanced decisions
        context = self._encode_context(pattern_stats, system_state)
        action_idx = self.bandit.select_action(context)
        
        # Register pending update for delayed reward
        self.bandit.register_pending(context, action_idx)
        
        return self.ACTIONS[action_idx]
    
    def record_outcome(self, context: np.ndarray, action: int, reward: float):
        """Update bandit with actual outcome."""
        self.bandit.update(context, action, reward)
    
    def get_pending_updates(self) -> List[PendingUpdate]:
        """Get pending bandit updates."""
        return self.bandit.get_pending_updates()
    
    def clear_pending_updates(self):
        """Clear pending updates after processing."""
        self.bandit.clear_pending()
    
    def _encode_context(
        self, 
        pattern_stats: Dict[str, Any], 
        system_state: Dict[str, Any]
    ) -> np.ndarray:
        """Encode decision context into fixed-length vector."""
        # Normalize features to [0, 1] range
        context = np.array([
            min(pattern_stats.get("sample_count", 0) / 100.0, 1.0),
            min(pattern_stats.get("avg_cost", 0) / 10.0, 1.0),
            min(pattern_stats.get("error_rate", 0), 1.0),
            min(pattern_stats.get("success_rate", 0), 1.0),
            min(pattern_stats.get("last_used_hours_ago", 0) / 168.0, 1.0),  # 1 week
            min(system_state.get("cpu_load", 0) / 100.0, 1.0),
            min(system_state.get("queue_length", 0) / 100.0, 1.0),
            min(system_state.get("memory_usage", 0) / 100.0, 1.0),
            min(system_state.get("error_rate_recent", 0), 1.0),
            min(system_state.get("avg_latency_ms", 0) / 5000.0, 1.0),
            1.0 if system_state.get("is_peak_hours", False) else 0.0,
            1.0 if system_state.get("resource_constrained", False) else 0.0,
            min(system_state.get("optimization_budget_remaining", 1.0), 1.0),
            min(system_state.get("llm_quota_usage", 0) / 100.0, 1.0),
            1.0 if system_state.get("drift_detected", False) else 0.0,
            min(system_state.get("component_availability", 1.0), 1.0),
        ])
        
        return context
    
    def _compute_reward(
        self, 
        action: str, 
        outcome: Dict[str, Any]
    ) -> float:
        """Compute reward signal from action outcome."""
        base_reward = 1.0 if outcome.get("success", False) else 0.0
        
        # Adjust reward based on action type
        if action == "optimize":
            # Reward if optimization improved performance
            improvement = outcome.get("performance_improvement", 0)
            return base_reward + min(improvement, 1.0)
        
        elif action == "wait":
            # Reward if waiting was appropriate (no degradation)
            return base_reward * (1.0 - outcome.get("opportunity_cost", 0))
        
        elif action == "deprecate":
            # Reward if deprecation saved resources
            return base_reward + outcome.get("resources_saved", 0.5)
        
        elif action == "retrain":
            # Reward if retraining fixed issues
            return base_reward + outcome.get("accuracy_gain", 0)
        
        return base_reward
