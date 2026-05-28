"""Pessimistic Weakness Analyzer (PWA) for DAG nodes."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import asyncio

try:
    from ..orchestration.polyglot.models import NodeTelemetry, WeaknessReport, WorkflowDAG
    from ..telemetry.recorder import TelemetryRecorder
except ImportError:
    from orchestration.polyglot.models import NodeTelemetry, WeaknessReport, WorkflowDAG
    from telemetry.recorder import TelemetryRecorder


class PessimisticWeaknessAnalyzer:
    """
    Analyzes DAG nodes to identify the weakest links.
    
    Calculates fragility scores based on multiple factors:
    - Bottleneck impact (p95 latency relative to total DAG time)
    - Instability (variance in execution time)
    - Error rate
    - Cost burn
    
    Triggers auto-remediation when thresholds are exceeded.
    """
    
    # Thresholds for auto-remediation
    FRAGILITY_THRESHOLD = 0.7
    ERROR_RATE_THRESHOLD = 0.15
    DETERMINISM_THRESHOLD = 0.8
    COST_BURN_THRESHOLD = 0.4
    VARIANCE_MULTIPLIER_THRESHOLD = 2.0
    
    def __init__(self, telemetry_recorder: TelemetryRecorder):
        self.recorder = telemetry_recorder
    
    def calculate_fragility(
        self, 
        telemetry: NodeTelemetry, 
        dag_stats: Dict[str, Any]
    ) -> float:
        """
        Calculate fragility score for a node.
        
        Args:
            telemetry: Node telemetry data
            dag_stats: Overall DAG statistics
            
        Returns:
            Fragility score between 0.0 and 1.0
        """
        # Bottleneck weight: How much does this node contribute to total latency?
        bottleneck_weight = 0.3 * (
            telemetry.p95_ms / max(dag_stats.get("total_p95", 1), 1)
        )
        
        # Instability weight: How variable is execution time?
        instability_weight = 0.3 * min(telemetry.variance_ms / 500, 1.0)
        
        # Error weight: How often does it fail?
        error_weight = 0.2 * telemetry.error_rate
        
        # Cost weight: How expensive is it relative to total cost?
        cost_weight = 0.2 * (
            telemetry.cost_usd / max(dag_stats.get("total_cost", 0.01), 0.01)
        )
        
        return min(bottleneck_weight + instability_weight + error_weight + cost_weight, 1.0)
    
    def analyze_node(
        self, 
        node_id: str, 
        dag_id: str
    ) -> Optional[WeaknessReport]:
        """
        Analyze a single node and generate weakness report if needed.
        
        Args:
            node_id: Node identifier
            dag_id: DAG identifier
            
        Returns:
            WeaknessReport if node exceeds thresholds, None otherwise
        """
        # Get telemetry and stats
        telemetry = self.recorder.get_recent_telemetry(node_id, window=100)
        dag_stats = self.recorder.get_dag_stats(dag_id, window=100)
        
        # Calculate fragility
        fragility_score = self.calculate_fragility(telemetry, dag_stats)
        
        if fragility_score < self.FRAGILITY_THRESHOLD:
            return None
        
        # Determine primary cause
        primary_cause, remediation = self._diagnose_issue(telemetry, dag_stats)
        
        # Calculate confidence based on sample size
        sample_confidence = min(1.0, len(self.recorder.get_recent_telemetry(node_id, 1000).variance_ms) / 100)
        confidence = fragility_score * sample_confidence
        
        return WeaknessReport(
            dag_id=dag_id,
            node_id=node_id,
            fragility_score=fragility_score,
            primary_cause=primary_cause,
            remediation=remediation,
            confidence=confidence
        )
    
    def _diagnose_issue(
        self, 
        telemetry: NodeTelemetry, 
        dag_stats: Dict[str, Any]
    ) -> tuple:
        """Diagnose the primary issue and suggest remediation."""
        
        # Check for bottleneck
        if telemetry.p95_ms > dag_stats.get("total_p95", 1) * 0.5:
            return "BOTTLENECK", "Consider parallelizing or optimizing this node"
        
        # Check for high error rate with low determinism
        if telemetry.error_rate > self.ERROR_RATE_THRESHOLD and \
           telemetry.determinism_score < self.DETERMINISM_THRESHOLD:
            return "INSTABILITY", "Inject retry/backoff + schema validation"
        
        # Check for cost burn on LLM nodes
        if telemetry.cost_usd > dag_stats.get("total_cost", 0.01) * self.COST_BURN_THRESHOLD:
            return "COST_BURN", "Downgrade prompt to regex/bash template"
        
        # Check for high variance
        if telemetry.variance_ms > telemetry.p50_ms * self.VARIANCE_MULTIPLIER_THRESHOLD:
            return "INSTABILITY", "Add network timeout + exponential backoff"
        
        # Check for low determinism alone
        if telemetry.determinism_score < self.DETERMINISM_THRESHOLD:
            return "LOW_DETERMINISM", "Add input validation and output normalization"
        
        # Default case
        return "COMPOSITE", "Review node implementation and consider refactoring"
    
    async def analyze_dag(self, dag: WorkflowDAG) -> List[WeaknessReport]:
        """
        Analyze all nodes in a DAG and return weakness reports.
        
        Args:
            dag: WorkflowDAG to analyze
            
        Returns:
            List of WeaknessReport for nodes exceeding thresholds
        """
        reports = []
        
        for node in dag.nodes:
            report = self.analyze_node(node.node_id, dag.dag_id)
            if report:
                reports.append(report)
        
        # Sort by fragility score descending
        reports.sort(key=lambda r: r.fragility_score, reverse=True)
        
        return reports
    
    def get_remediation_strategy(self, report: WeaknessReport) -> Dict[str, Any]:
        """Get detailed remediation strategy for a weakness report."""
        strategies = {
            "BOTTLENECK": {
                "actions": [
                    "Profile node execution to identify slow operations",
                    "Consider fusing with adjacent nodes",
                    "Evaluate if operation can be done in a faster runtime (BASH vs PYTHON)",
                    "Add caching for repeated inputs"
                ],
                "priority": "HIGH",
                "estimated_impact": "30-60% latency reduction"
            },
            "INSTABILITY": {
                "actions": [
                    "Add retry policy with exponential backoff",
                    "Implement circuit breaker pattern",
                    "Add input validation before execution",
                    "Consider switching to more deterministic runtime"
                ],
                "priority": "HIGH",
                "estimated_impact": "50-80% error reduction"
            },
            "COST_BURN": {
                "actions": [
                    "Analyze if LLM call is necessary",
                    "Replace with BASH/jq for structured data extraction",
                    "Cache responses for similar inputs",
                    "Use smaller/faster model if available"
                ],
                "priority": "MEDIUM",
                "estimated_impact": "40-70% cost reduction"
            },
            "LOW_DETERMINISM": {
                "actions": [
                    "Add strict input schema validation",
                    "Normalize output format",
                    "Add post-processing to ensure consistent output",
                    "Consider replacing non-deterministic operations"
                ],
                "priority": "MEDIUM",
                "estimated_impact": "60-90% determinism improvement"
            },
            "COMPOSITE": {
                "actions": [
                    "Conduct full node audit",
                    "Consider complete rewrite",
                    "Evaluate splitting into multiple specialized nodes",
                    "Add comprehensive monitoring"
                ],
                "priority": "LOW",
                "estimated_impact": "Variable"
            }
        }
        
        return strategies.get(report.primary_cause, strategies["COMPOSITE"])


class MutationEngine:
    """
    Generates and tests node mutations for fragile nodes.
    
    Creates candidate variants (BASH/PYTHON/LLM) and selects
    the best performer based on speed, accuracy, and cost.
    """
    
    def __init__(self, pwa: PessimisticWeaknessAnalyzer, sandbox, llm_router=None):
        self.pwa = pwa
        self.sandbox = sandbox
        self.llm_router = llm_router
    
    async def spawn_targeted_loop(
        self, 
        dag: WorkflowDAG, 
        report: WeaknessReport
    ) -> Optional[Dict[str, Any]]:
        """
        Spawn mutation loop for a fragile node.
        
        Args:
            dag: Original DAG
            report: Weakness report identifying the issue
            
        Returns:
            Best mutation result if improvement found, None otherwise
        """
        node = dag.get_node_by_id(report.node_id)
        if not node:
            return None
        
        print(f"[Mutation Engine] Spawning targeted loop for node {node.node_id}")
        
        # Generate candidates
        candidates = await self._generate_candidates(node, report)
        
        # Get historical inputs for testing
        test_inputs = await self._get_test_inputs(node.node_id)
        
        # Test each candidate
        results = []
        for candidate in candidates:
            score = await self._test_candidate(candidate, test_inputs)
            results.append((candidate, score))
        
        # Select best if significant improvement
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_score = results[0]
            
            # Check if improvement > 15%
            current_baseline = 1.0 - report.fragility_score
            if best_score > current_baseline * 1.15:
                print(f"[Mutation Engine] Found {best_score:.2f} score ({(best_score/current_baseline - 1)*100:.1f}% improvement)")
                return {
                    "original_node": node,
                    "mutated_node": best_candidate,
                    "improvement_score": best_score,
                    "version_bump": dag.version + 1
                }
        
        return None
    
    async def _generate_candidates(
        self, 
        node: Any, 
        report: WeaknessReport
    ) -> List[Any]:
        """Generate mutation candidates based on weakness type."""
        candidates = []
        
        # Try different runtimes based on the issue
        if report.primary_cause == "COST_BURN" and node.runtime == "LLM":
            # Downgrade to BASH/jq
            bash_candidate = await self._downgrade_to_bash(node)
            if bash_candidate:
                candidates.append(bash_candidate)
        
        if report.primary_cause == "INSTABILITY" and node.runtime == "BASH":
            # Upgrade to PYTHON for better error handling
            python_candidate = await self._upgrade_to_python(node)
            if python_candidate:
                candidates.append(python_candidate)
        
        # Always try optimized version of same runtime
        optimized = await self._optimize_same_runtime(node, report)
        if optimized:
            candidates.append(optimized)
        
        return candidates
    
    async def _downgrade_to_bash(self, node: Any) -> Optional[Any]:
        """Try to replace LLM node with BASH equivalent."""
        # This would use LLM to generate bash equivalent
        # For now, return a placeholder
        if self.llm_router:
            prompt = f"""Convert this LLM task to a bash script using curl/jq/awk:
            Task: {node.name}
            Current payload: {node.payload[:200]}
            
            Output ONLY the bash script, no explanations."""
            
            try:
                result = await self.llm_router.generate(prompt, {}, {"max_retries": 1})
                if 'content' in result:
                    # Create new node with bash runtime
                    from ..orchestration.polyglot.models import DAGNode
                    return DAGNode(
                        node_id=f"{node.node_id}_bash_mut",
                        name=f"{node.name}_bash",
                        runtime="BASH",
                        payload=result['content'],
                        input_mapping=node.input_mapping,
                        output_schema=node.output_schema,
                        timeout_ms=node.timeout_ms // 2,  # Bash should be faster
                        retry_policy=node.retry_policy
                    )
            except Exception:
                pass
        
        return None
    
    async def _upgrade_to_python(self, node: Any) -> Optional[Any]:
        """Upgrade BASH node to PYTHON for better error handling."""
        # Placeholder - would generate Python code
        return None
    
    async def _optimize_same_runtime(self, node: Any, report: WeaknessReport) -> Optional[Any]:
        """Optimize node within same runtime."""
        # Add retries if instability issue
        if report.primary_cause == "INSTABILITY":
            from ..orchestration.polyglot.models import DAGNode
            return DAGNode(
                node_id=f"{node.node_id}_opt",
                name=f"{node.name}_optimized",
                runtime=node.runtime,
                payload=node.payload,
                input_mapping=node.input_mapping,
                output_schema=node.output_schema,
                timeout_ms=node.timeout_ms,
                retry_policy={"max_retries": 3, "backoff_ms": 500}
            )
        
        return None
    
    async def _get_test_inputs(self, node_id: str) -> List[Dict[str, Any]]:
        """Get historical inputs for testing."""
        # Query telemetry for past inputs
        # For now, return empty list
        return []
    
    async def _test_candidate(
        self, 
        candidate: Any, 
        test_inputs: List[Dict[str, Any]]
    ) -> float:
        """Test a candidate mutation and return score."""
        if not test_inputs:
            # No test inputs, return baseline score
            return 0.5
        
        # Run candidate with test inputs and measure performance
        # Placeholder implementation
        return 0.6
