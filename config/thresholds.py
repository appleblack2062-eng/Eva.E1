"""Confidence and optimization thresholds for agent decision-making."""

from dataclasses import dataclass

@dataclass(frozen=True)
class OptimizationThresholds:
    """When to transition between execution modes."""
    
    # Workflow synthesis triggers
    min_task_repetitions_for_synthesis: int = 3
    min_synthesis_confidence: float = 0.85
    
    # Validation requirements
    min_test_pass_rate_for_deployment: float = 0.95
    min_validation_coverage: float = 0.8
    
    # Optimization triggers
    min_executions_before_optimization: int = 10
    min_performance_gain_for_optimization: float = 0.3  # 30% faster
    
    # Deployment criteria
    min_workflow_stability_score: float = 0.9
    max_acceptable_error_rate: float = 0.02
    
    # LLM fallback triggers
    max_workflow_retry_attempts: int = 2
    confidence_threshold_for_llm_fallback: float = 0.7
    
    # Resource limits
    max_workflow_execution_time_seconds: float = 30.0
    max_memory_usage_mb: int = 512
    max_generated_code_lines: int = 500
    
    # Learning rate controls
    workflow_refinement_rate: float = 0.1  # How aggressively to update workflows
    strategy_update_frequency: int = 100   # Update routing policy every N tasks

@dataclass(frozen=True)
class EfficiencyTargets:
    """Performance goals for the agent."""
    
    # Latency targets (milliseconds)
    target_p95_latency_simple_task: float = 200.0
    target_p95_latency_complex_task: float = 2000.0
    
    # Token efficiency
    target_llm_token_reduction: float = 0.7  # Reduce LLM tokens by 70%
    max_tokens_per_task_initial: int = 4000
    max_tokens_per_task_optimized: int = 500
    
    # Success metrics
    target_task_success_rate: float = 0.98
    target_workflow_reuse_rate: float = 0.85
    
    # Cost optimization
    target_cost_reduction_vs_llm_only: float = 0.6
