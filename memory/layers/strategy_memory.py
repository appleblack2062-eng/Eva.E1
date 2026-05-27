"""Stores learned routing policies."""

from __future__ import annotations
from typing import Dict, Any
from ...learning.strategy_learner import StrategyLearner

class StrategyMemoryLayer:
    """Interface for the strategy learner."""
    
    def __init__(self, agent_id: str, config):
        self.learner = StrategyLearner(config)
    
    async def update_policy(self, **kwargs):
        await self.learner.update_policy(**kwargs)
    
    async def decide_mode(self, **kwargs):
        return await self.learner.decide_mode(**kwargs)
    
    async def compute_llm_offload_ratio(self) -> float:
        stats = await self.learner.get_policy_stats()
        # Calculate based on mode distribution
        modes = stats.get("mode_distribution", {})
        total = sum(modes.values())
        if total == 0: return 0.0
        non_llm = sum(v for k, v in modes.items() if "LLM" not in k)
        return non_llm / total
