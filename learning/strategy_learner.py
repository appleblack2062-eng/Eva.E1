"""Learn optimal task routing policies from execution feedback."""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from ..config.settings import AgentConfig
from ..core.memory_types import ExecutionMode, TaskPattern

@dataclass
class RoutingPolicy:
    """Policy for routing tasks to execution modes."""
    task_pattern_hash: str
    preferred_mode: ExecutionMode
    confidence: float
    avg_latency_ms: float
    avg_tokens: int
    success_rate: float
    last_updated: float
    sample_count: int = 0

class StrategyLearner:
    """
    Multi-armed bandit + contextual learning for execution routing.
    
    Learns which execution mode (LLM vs workflow) works best for 
    which task patterns, optimizing for:
    - Latency
    - Token usage  
    - Success rate
    - Cost
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._policies: Dict[str, RoutingPolicy] = {}
        self._exploration_rate = 0.1  # Epsilon-greedy exploration
        self._decay_factor = 0.99  # For moving averages
        
    async def update_policy(
        self,
        task_pattern: Dict[str, Any],
        outcome: bool,
        latency_ms: float,
        tokens_used: int,
        mode_used: ExecutionMode,
    ):
        """Update routing policy based on execution outcome."""
        
        pattern_hash = self._hash_task_pattern(task_pattern)
        
        # Get or create policy for this pattern
        policy = self._policies.get(pattern_hash)
        if not policy:
            policy = RoutingPolicy(
                task_pattern_hash=pattern_hash,
                preferred_mode=mode_used,
                confidence=0.5,
                avg_latency_ms=latency_ms,
                avg_tokens=tokens_used,
                success_rate=1.0 if outcome else 0.0,
                last_updated=time.time(),
                sample_count=1,
            )
            self._policies[pattern_hash] = policy
            return
        
        # Update moving averages with exponential decay
        policy.sample_count += 1
        policy.avg_latency_ms = (
            policy.avg_latency_ms * self._decay_factor + 
            latency_ms * (1 - self._decay_factor)
        )
        policy.avg_tokens = int(
            policy.avg_tokens * self._decay_factor + 
            tokens_used * (1 - self._decay_factor)
        )
        policy.success_rate = (
            policy.success_rate * self._decay_factor + 
            (1.0 if outcome else 0.0) * (1 - self._decay_factor)
        )
        
        # Update confidence based on sample count and consistency
        policy.confidence = min(
            1.0,
            0.5 + 0.5 * (1 - np.exp(-policy.sample_count / 20))
        )
        
        # Update preferred mode if new mode is significantly better
        if self._should_switch_mode(policy, mode_used, outcome, latency_ms, tokens_used):
            policy.preferred_mode = mode_used
            policy.last_updated = time.time()
        
        # Decay exploration rate over time
        self._exploration_rate *= 0.999
    
    def _should_switch_mode(
        self,
        current_policy: RoutingPolicy,
        new_mode: ExecutionMode,
        new_outcome: bool,
        new_latency: float,
        new_tokens: int,
    ) -> bool:
        """Determine if we should switch to a new execution mode."""
        
        # Must have enough samples to be confident
        if current_policy.sample_count < 10:
            return False
        
        # New mode must have better success rate
        if not new_outcome and current_policy.success_rate > 0.9:
            return False
        
        # New mode should be faster or use fewer tokens
        latency_improvement = current_policy.avg_latency_ms / (new_latency + 1)
        token_improvement = current_policy.avg_tokens / (new_tokens + 1)
        
        # Switch if improvement > threshold AND success rate acceptable
        improvement = max(latency_improvement, token_improvement)
        return (
            improvement > 1.3 and  # 30% improvement
            new_outcome and
            current_policy.success_rate < 0.95  # Room for improvement
        )
    
    async def decide_mode(
        self,
        task_pattern: Dict[str, Any],
        available_modes: List[ExecutionMode],
    ) -> ExecutionMode:
        """Decide which execution mode to use for a task."""
        
        pattern_hash = self._hash_task_pattern(task_pattern)
        policy = self._policies.get(pattern_hash)
        
        # Epsilon-greedy: explore with probability _exploration_rate
        if np.random.random() < self._exploration_rate:
            return np.random.choice(available_modes)
        
        # Exploit: use best known mode for this pattern
        if policy and policy.preferred_mode in available_modes:
            return policy.preferred_mode
        
        # Fallback: prefer most efficient mode generally
        return self._default_mode_selection(available_modes)
    
    def _default_mode_selection(self, available_modes: List[ExecutionMode]) -> ExecutionMode:
        """Default mode selection when no policy exists."""
        # Prefer optimized modes in order of efficiency
        preference_order = [
            ExecutionMode.WORKFLOW_JIT,
            ExecutionMode.WORKFLOW_COMPILED,
            ExecutionMode.WORKFLOW_DRAFT,
            ExecutionMode.LLM_GUIDED,
            ExecutionMode.LLM_ONLY,
        ]
        
        for mode in preference_order:
            if mode in available_modes:
                return mode
        return available_modes[0]
    
    def _hash_task_pattern(self, pattern: Dict[str, Any]) -> str:
        """Create hash from task pattern for policy lookup."""
        import hashlib
        import json
        # Normalize and hash
        normalized = json.dumps(pattern, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    async def get_policy_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics about learned policies."""
        
        if not self._policies:
            return {"policy_count": 0}
        
        policies = list(self._policies.values())
        
        return {
            "policy_count": len(policies),
            "avg_confidence": np.mean([p.confidence for p in policies]),
            "avg_success_rate": np.mean([p.success_rate for p in policies]),
            "mode_distribution": self._count_mode_preferences(policies),
            "top_performers": sorted(
                policies,
                key=lambda p: p.success_rate * (1 / (p.avg_latency_ms + 1)),
                reverse=True
            )[:5],
        }
    
    def _count_mode_preferences(self, policies: List[RoutingPolicy]) -> Dict[str, int]:
        """Count how many policies prefer each execution mode."""
        counts = defaultdict(int)
        for policy in policies:
            counts[policy.preferred_mode.name] += 1
        return dict(counts)
