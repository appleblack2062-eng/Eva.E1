"""Hardware & Cost Aware Contextual Routing for multi-tiered execution.

This module implements a tri-tiered routing engine that dynamically routes tasks
based on hardware utilization, cost considerations, and task complexity:

Tier 1: Cached/Compiled Workflows - Zero-token local code execution (fastest, cheapest)
Tier 2: Local Edge SLM - Lightweight local models for reasoning tasks
Tier 3: Cloud Frontier LLM - Complex, novel tasks requiring deep knowledge

The router creates a closed-loop utility cost function balancing compute power,
latency goals, and financial cost seamlessly behind a single endpoint.
"""

from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics


class ExecutionTier(Enum):
    """Execution tiers for the routing matrix."""
    TIER1_COMPILED = "tier1_compiled"  # Cached/compiled workflows
    TIER2_LOCAL_SLM = "tier2_local_slm"  # Local small language models
    TIER3_CLOUD_LLM = "tier3_cloud_llm"  # Cloud frontier models


class HardwareState(Enum):
    """Hardware utilization states."""
    IDLE = "idle"
    NORMAL = "normal"
    LOADED = "loaded"
    THROTTLED = "throttled"
    CRITICAL = "critical"


@dataclass
class TierMetrics:
    """Performance metrics for an execution tier."""
    tier: ExecutionTier
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    success_rate: float = 1.0
    throughput_per_minute: float = 0.0
    error_count: int = 0
    total_executions: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    selected_tier: ExecutionTier
    confidence: float
    reason: str
    estimated_latency_ms: float
    estimated_cost_usd: float
    fallback_tiers: List[ExecutionTier] = field(default_factory=list)


@dataclass
class TaskContext:
    """Context information for routing decisions."""
    task_id: str
    task_type: str
    complexity_score: float  # 0.0 to 1.0
    input_size_bytes: int
    expected_output_size_bytes: int
    latency_budget_ms: Optional[float] = None
    cost_budget_usd: Optional[float] = None
    requires_reasoning: bool = False
    requires_knowledge: bool = False
    is_recurring: bool = False
    cached_workflow_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareMonitor:
    """
    Monitors local hardware utilization for routing decisions.
    
    Tracks CPU, GPU, memory, and thermal state to determine
    when to shift load from local to cloud execution.
    """
    
    def __init__(self, config):
        self.config = config
        self._cpu_samples: List[float] = []
        self._memory_samples: List[float] = []
        self._gpu_samples: List[float] = []
        self._temperature_samples: List[float] = []
        self._sample_window_seconds = 60
        self._last_sample_time = 0.0
    
    async def sample(self) -> Dict[str, float]:
        """Collect hardware utilization samples."""
        current_time = time.time()
        
        # Sample CPU usage
        cpu_usage = await self._get_cpu_usage()
        self._cpu_samples.append((current_time, cpu_usage))
        
        # Sample memory usage
        memory_usage = await self._get_memory_usage()
        self._memory_samples.append((current_time, memory_usage))
        
        # Sample GPU usage if available
        gpu_usage = await self._get_gpu_usage()
        if gpu_usage is not None:
            self._gpu_samples.append((current_time, gpu_usage))
        
        # Sample temperature if available
        temp = await self._get_temperature()
        if temp is not None:
            self._temperature_samples.append((current_time, temp))
        
        # Clean old samples
        self._cleanup_old_samples(current_time)
        
        return {
            'cpu': cpu_usage,
            'memory': memory_usage,
            'gpu': gpu_usage,
            'temperature': temp
        }
    
    def get_hardware_state(self) -> HardwareState:
        """Determine current hardware state from samples."""
        if not self._cpu_samples:
            return HardwareState.IDLE
        
        # Get recent averages
        recent_cpu = self._get_recent_average(self._cpu_samples)
        recent_memory = self._get_recent_average(self._memory_samples)
        recent_temp = self._get_recent_average(self._temperature_samples) if self._temperature_samples else 0
        
        # Determine state based on thresholds
        if recent_cpu > 90 or recent_memory > 90 or (recent_temp and recent_temp > 85):
            return HardwareState.CRITICAL
        elif recent_cpu > 70 or recent_memory > 70 or (recent_temp and recent_temp > 75):
            return HardwareState.THROTTLED
        elif recent_cpu > 50 or recent_memory > 50:
            return HardwareState.LOADED
        elif recent_cpu > 20 or recent_memory > 20:
            return HardwareState.NORMAL
        else:
            return HardwareState.IDLE
    
    def should_offload_to_cloud(self) -> bool:
        """Determine if load should be offloaded to cloud."""
        state = self.get_hardware_state()
        return state in (HardwareState.THROTTLED, HardwareState.CRITICAL)
    
    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback: return estimated value
            return 30.0
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 40.0
    
    async def _get_gpu_usage(self) -> Optional[float]:
        """Get current GPU usage percentage if available."""
        try:
            # Try pynvml for NVIDIA GPUs
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            pynvml.nvmlShutdown()
            return usage
        except Exception:
            return None
    
    async def _get_temperature(self) -> Optional[float]:
        """Get system temperature if available."""
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
        except Exception:
            pass
        return None
    
    def _cleanup_old_samples(self, current_time: float) -> None:
        """Remove samples older than sample window."""
        cutoff = current_time - self._sample_window_seconds
        
        self._cpu_samples = [(t, v) for t, v in self._cpu_samples if t > cutoff]
        self._memory_samples = [(t, v) for t, v in self._memory_samples if t > cutoff]
        self._gpu_samples = [(t, v) for t, v in self._gpu_samples if t > cutoff]
        self._temperature_samples = [(t, v) for t, v in self._temperature_samples if t > cutoff]
    
    def _get_recent_average(self, samples: List[Tuple[float, float]]) -> float:
        """Calculate average of recent samples."""
        if not samples:
            return 0.0
        values = [v for _, v in samples]
        return statistics.mean(values)


class CostTracker:
    """
    Tracks execution costs across tiers for optimization.
    
    Maintains running averages of latency, token usage, and
    monetary cost to inform routing decisions.
    """
    
    def __init__(self, config):
        self.config = config
        self._metrics: Dict[ExecutionTier, TierMetrics] = {
            tier: TierMetrics(tier=tier) for tier in ExecutionTier
        }
        self._latency_history: Dict[ExecutionTier, List[float]] = defaultdict(list)
        self._cost_history: Dict[ExecutionTier, List[float]] = defaultdict(list)
    
    def record_execution(
        self,
        tier: ExecutionTier,
        latency_ms: float,
        cost_usd: float,
        success: bool,
        tokens_used: int = 0
    ) -> None:
        """Record execution metrics for a tier."""
        metrics = self._metrics[tier]
        
        # Update latency
        self._latency_history[tier].append(latency_ms)
        if len(self._latency_history[tier]) > 1000:
            self._latency_history[tier].pop(0)
        metrics.avg_latency_ms = statistics.mean(self._latency_history[tier])
        metrics.p95_latency_ms = sorted(self._latency_history[tier])[int(len(self._latency_history[tier]) * 0.95)] if self._latency_history[tier] else 0
        
        # Update cost
        self._cost_history[tier].append(cost_usd)
        if len(self._cost_history[tier]) > 1000:
            self._cost_history[tier].pop(0)
        metrics.avg_cost_usd = statistics.mean(self._cost_history[tier])
        
        # Update success rate
        metrics.total_executions += 1
        if success:
            metrics.error_count += 0  # Would track successes separately
        else:
            metrics.error_count += 1
        
        metrics.success_rate = 1.0 - (metrics.error_count / max(1, metrics.total_executions))
        metrics.last_updated = time.time()
    
    def get_tier_metrics(self, tier: ExecutionTier) -> TierMetrics:
        """Get current metrics for a tier."""
        return self._metrics[tier]
    
    def predict_cost(
        self,
        tier: ExecutionTier,
        task_context: TaskContext
    ) -> float:
        """Predict cost for executing a task on a tier."""
        base_cost = self._metrics[tier].avg_cost_usd
        
        # Adjust based on task characteristics
        if tier == ExecutionTier.TIER3_CLOUD_LLM:
            # Cloud LLM cost scales with tokens
            estimated_tokens = task_context.input_size_bytes // 4 + task_context.expected_output_size_bytes // 4
            cost_per_token = getattr(self.config, 'cloud_cost_per_token', 0.00002)
            return estimated_tokens * cost_per_token
        
        elif tier == ExecutionTier.TIER2_LOCAL_SLM:
            # Local SLM has fixed electricity cost
            return getattr(self.config, 'local_electricity_cost_per_task', 0.0001)
        
        else:
            # Compiled workflow has near-zero marginal cost
            return 0.00001
    
    def predict_latency(
        self,
        tier: ExecutionTier,
        task_context: TaskContext
    ) -> float:
        """Predict latency for executing a task on a tier."""
        base_latency = self._metrics[tier].avg_latency_ms
        
        # Adjust based on task complexity
        complexity_factor = 1.0 + (task_context.complexity_score * 0.5)
        
        if tier == ExecutionTier.TIER1_COMPILED:
            # Compiled workflows are fast and predictable
            return base_latency * complexity_factor
        
        elif tier == ExecutionTier.TIER2_LOCAL_SLM:
            # Local SLM latency depends on model size and input
            input_factor = 1.0 + (task_context.input_size_bytes / 10000)
            return base_latency * complexity_factor * input_factor
        
        else:
            # Cloud LLM has network latency plus processing
            network_latency = getattr(self.config, 'cloud_network_latency_ms', 100)
            return network_latency + (base_latency * complexity_factor)


class HybridExecutionRouter:
    """
    Multi-tiered routing engine for optimal execution selection.
    
    Implements a closed-loop utility cost function that balances:
    - Latency requirements
    - Cost constraints
    - Hardware utilization
    - Task complexity
    - Historical performance
    """
    
    def __init__(self, config, llm_client=None, workflow_cache=None):
        self.config = config
        self.llm = llm_client
        self.workflow_cache = workflow_cache
        self.hardware_monitor = HardwareMonitor(config)
        self.cost_tracker = CostTracker(config)
        
        # Routing policy weights
        self.latency_weight = getattr(config, 'routing_latency_weight', 0.4)
        self.cost_weight = getattr(config, 'routing_cost_weight', 0.3)
        self.reliability_weight = getattr(config, 'routing_reliability_weight', 0.3)
        
        # Tier-specific configurations
        self.tier_configs = {
            ExecutionTier.TIER1_COMPILED: {
                'max_complexity': 0.3,
                'requires_cache': True,
                'min_confidence': 0.9
            },
            ExecutionTier.TIER2_LOCAL_SLM: {
                'max_complexity': 0.7,
                'requires_cache': False,
                'min_confidence': 0.7
            },
            ExecutionTier.TIER3_CLOUD_LLM: {
                'max_complexity': 1.0,
                'requires_cache': False,
                'min_confidence': 0.0
            }
        }
    
    async def route_task(self, context: TaskContext) -> RoutingDecision:
        """
        Determine optimal execution tier for a task.
        
        Args:
            context: Task context with complexity, budgets, and requirements
            
        Returns:
            RoutingDecision with selected tier and rationale
        """
        # Refresh hardware state
        await self.hardware_monitor.sample()
        hw_state = self.hardware_monitor.get_hardware_state()
        
        # Evaluate each tier
        tier_scores: Dict[ExecutionTier, float] = {}
        tier_reasons: Dict[ExecutionTier, str] = {}
        
        for tier in ExecutionTier:
            score, reason = await self._evaluate_tier(tier, context, hw_state)
            tier_scores[tier] = score
            tier_reasons[tier] = reason
        
        # Select best tier
        sorted_tiers = sorted(tier_scores.items(), key=lambda x: x[1], reverse=True)
        best_tier = sorted_tiers[0][0]
        best_score = sorted_tiers[0][1]
        
        # Build fallback list
        fallbacks = [t for t, _ in sorted_tiers[1:] if tier_scores[t] > 0.3]
        
        # Calculate estimates
        estimated_cost = self.cost_tracker.predict_cost(best_tier, context)
        estimated_latency = self.cost_tracker.predict_latency(best_tier, context)
        
        return RoutingDecision(
            selected_tier=best_tier,
            confidence=best_score,
            reason=tier_reasons[best_tier],
            estimated_latency_ms=estimated_latency,
            estimated_cost_usd=estimated_cost,
            fallback_tiers=fallbacks
        )
    
    async def _evaluate_tier(
        self,
        tier: ExecutionTier,
        context: TaskContext,
        hw_state: HardwareState
    ) -> Tuple[float, str]:
        """Evaluate suitability of a tier for a task."""
        score = 0.0
        reasons = []
        
        tier_config = self.tier_configs[tier]
        metrics = self.cost_tracker.get_tier_metrics(tier)
        
        # Check basic eligibility
        if context.complexity_score > tier_config['max_complexity']:
            return 0.0, f"Task complexity {context.complexity_score:.2f} exceeds tier max {tier_config['max_complexity']:.2f}"
        
        if tier_config['requires_cache'] and not context.cached_workflow_available:
            return 0.0, "No cached workflow available for Tier 1"
        
        # Hardware-based adjustments
        if tier in (ExecutionTier.TIER1_COMPILED, ExecutionTier.TIER2_LOCAL_SLM):
            if hw_state == HardwareState.CRITICAL:
                score -= 0.5
                reasons.append("Local hardware critical")
            elif hw_state == HardwareState.THROTTLED:
                score -= 0.3
                reasons.append("Local hardware throttled")
        
        # Latency score
        predicted_latency = self.cost_tracker.predict_latency(tier, context)
        if context.latency_budget_ms:
            if predicted_latency <= context.latency_budget_ms:
                latency_score = 1.0
            else:
                latency_score = max(0, 1.0 - (predicted_latency - context.latency_budget_ms) / context.latency_budget_ms)
        else:
            latency_score = 1.0 / (1.0 + predicted_latency / 1000)
        
        # Cost score
        predicted_cost = self.cost_tracker.predict_cost(tier, context)
        if context.cost_budget_usd:
            if predicted_cost <= context.cost_budget_usd:
                cost_score = 1.0
            else:
                cost_score = max(0, 1.0 - (predicted_cost - context.cost_budget_usd) / context.cost_budget_usd)
        else:
            cost_score = 1.0 / (1.0 + predicted_cost * 100)
        
        # Reliability score
        reliability_score = metrics.success_rate
        
        # Calculate weighted score
        score = (
            latency_score * self.latency_weight +
            cost_score * self.cost_weight +
            reliability_score * self.reliability_weight
        )
        
        # Bonus for recurring tasks with cached workflows
        if context.is_recurring and tier == ExecutionTier.TIER1_COMPILED:
            score += 0.2
            reasons.append("Recurring task with cache")
        
        # Penalty for cloud when local is viable
        if tier == ExecutionTier.TIER3_CLOUD_LLM and hw_state == HardwareState.IDLE:
            if context.complexity_score < 0.5:
                score -= 0.15
                reasons.append("Could use local execution")
        
        reason = "; ".join(reasons) if reasons else "Standard routing"
        return score, reason
    
    async def execute_with_routing(
        self,
        context: TaskContext,
        tier1_executor: Optional[Callable] = None,
        tier2_executor: Optional[Callable] = None,
        tier3_executor: Optional[Callable] = None
    ) -> Any:
        """
        Route and execute a task with automatic fallback.
        
        Args:
            context: Task context
            tier1_executor: Executor for compiled workflows
            tier2_executor: Executor for local SLM
            tier3_executor: Executor for cloud LLM
            
        Returns:
            Execution result
        """
        # Get routing decision
        decision = await self.route_task(context)
        
        executors = {
            ExecutionTier.TIER1_COMPILED: tier1_executor,
            ExecutionTier.TIER2_LOCAL_SLM: tier2_executor,
            ExecutionTier.TIER3_CLOUD_LLM: tier3_executor
        }
        
        # Try selected tier and fallbacks
        for tier in [decision.selected_tier] + decision.fallback_tiers:
            executor = executors.get(tier)
            if not executor:
                continue
            
            start_time = time.time()
            try:
                result = await executor(context)
                
                # Record success
                latency = (time.time() - start_time) * 1000
                cost = self.cost_tracker.predict_cost(tier, context)
                self.cost_tracker.record_execution(tier, latency, cost, True)
                
                return result
                
            except Exception as e:
                # Record failure
                latency = (time.time() - start_time) * 1000
                self.cost_tracker.record_execution(tier, latency, 0, False)
                
                # Try next fallback
                continue
        
        raise RuntimeError(f"All execution tiers failed for task {context.task_id}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics."""
        return {
            'hardware_state': self.hardware_monitor.get_hardware_state().value,
            'tier_metrics': {
                tier.value: {
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'avg_cost_usd': metrics.avg_cost_usd,
                    'success_rate': metrics.success_rate,
                    'total_executions': metrics.total_executions
                }
                for tier, metrics in self.cost_tracker._metrics.items()
            },
            'policy_weights': {
                'latency': self.latency_weight,
                'cost': self.cost_weight,
                'reliability': self.reliability_weight
            }
        }


def create_hybrid_router(config, llm_client=None, workflow_cache=None) -> HybridExecutionRouter:
    """Factory function to create hybrid execution router."""
    return HybridExecutionRouter(config, llm_client, workflow_cache)
