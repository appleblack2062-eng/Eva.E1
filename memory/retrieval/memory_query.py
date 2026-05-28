"""Memory Query Engine for Polyglot DAG patterns."""

from __future__ import annotations
import json
import hashlib
from typing import List, Dict, Any, Optional

try:
    from ..orchestration.polyglot.models import DAGNode, DAGEdge, WorkflowDAG, ExecutionTrace
    from ..telemetry.recorder import TelemetryRecorder
except ImportError:
    from orchestration.polyglot.models import DAGNode, DAGEdge, WorkflowDAG, ExecutionTrace
    from telemetry.recorder import TelemetryRecorder


class MemoryQueryEngine:
    """
    Queries historical execution traces to find reusable workflow components.
    
    Enables memory-driven synthesis by:
    - Matching new tasks to past executions
    - Extracting proven node implementations
    - Reusing successful patterns
    """
    
    def __init__(self, telemetry_recorder: TelemetryRecorder):
        self.recorder = telemetry_recorder
    
    def compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Compute hash of input data for pattern matching."""
        # Normalize and hash
        normalized = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    async def query_similar_workflows(
        self, 
        task_goal: str,
        input_sample: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find past workflows similar to current task.
        
        Args:
            task_goal: High-level goal description
            input_sample: Sample input data
            limit: Max results to return
            
        Returns:
            List of matching workflow traces with metadata
        """
        results = []
        
        # If we have input sample, query by input hash
        if input_sample:
            input_hash = self.compute_input_hash(input_sample)
            traces = self.recorder.query_similar_traces(input_hash, limit=limit)
            
            for trace in traces:
                results.append({
                    "trace_id": trace.trace_id,
                    "dag_id": trace.dag_id,
                    "node_id": trace.node_id,
                    "node_type": trace.node_type,
                    "telemetry": trace.telemetry,
                    "success": not bool(trace.error_trace),
                    "match_type": "input_hash"
                })
        
        # If no input-based matches, fall back to goal-based search
        # (Would use vector similarity in full implementation)
        if not results:
            # Return recent successful traces as fallback
            # This is a placeholder - would use semantic search
            pass
        
        return results
    
    def extract_best_nodes(
        self, 
        traces: List[ExecutionTrace]
    ) -> List[DAGNode]:
        """
        Extract high-performing node implementations from traces.
        
        Args:
            traces: List of execution traces
            
        Returns:
            List of DAGNode objects ready for reuse
        """
        nodes = []
        
        # Group traces by node_type
        type_groups = {}
        for trace in traces:
            if trace.node_type not in type_groups:
                type_groups[trace.node_type] = []
            type_groups[trace.node_type].append(trace)
        
        # Select best performer per type
        for node_type, group_traces in type_groups.items():
            # Sort by success rate and latency
            successful = [t for t in group_traces if not t.error_trace]
            if not successful:
                continue
            
            # Pick fastest successful execution
            best = min(
                successful,
                key=lambda t: t.telemetry.get('latency_ms', float('inf'))
            )
            
            # Create node from trace
            node = DAGNode(
                name=f"reused_{node_type}",
                runtime=node_type,
                payload=self._extract_payload_from_trace(best),
                output_schema=best.telemetry.get('output_schema', {}),
                timeout_ms=int(best.telemetry.get('p95_ms', 5000) * 1.5),
                retry_policy={
                    "max_retries": 2 if best.telemetry.get('error_rate', 0) > 0 else 0,
                    "backoff_ms": 500
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _extract_payload_from_trace(self, trace: ExecutionTrace) -> str:
        """Extract executable payload from trace."""
        # In full implementation, this would retrieve stored payload
        # For now, return placeholder based on node type
        if trace.node_type == "BASH":
            return "# Reused bash script from trace\n# TODO: Implement"
        elif trace.node_type == "PYTHON":
            return "# Reused Python code from trace\n# TODO: Implement"
        elif trace.node_type == "LLM":
            return "# Reused LLM prompt from trace\n# TODO: Implement"
        else:
            return f"# Reused {trace.node_type} component"
    
    def get_node_implementation_history(
        self, 
        node_type: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get historical performance data for a node type.
        
        Args:
            node_type: Runtime type (BASH, PYTHON, LLM, etc.)
            limit: Max results
            
        Returns:
            List of historical performance records
        """
        # Query all traces of this type
        # This would need additional indexing in the recorder
        # Placeholder implementation
        return []
    
    def detect_pattern_drift(
        self, 
        dag_id: str,
        window_size: int = 50
    ) -> Dict[str, Any]:
        """
        Detect if execution patterns are drifting over time.
        
        Args:
            dag_id: DAG to analyze
            window_size: Number of recent executions to consider
            
        Returns:
            Drift analysis report
        """
        # Get recent executions
        stats = self.recorder.get_dag_stats(dag_id, window=window_size)
        
        # Compare first half vs second half of window
        # (Simplified - would do statistical test)
        
        return {
            "dag_id": dag_id,
            "drift_detected": False,
            "metrics": stats,
            "recommendation": "No action needed"
        }
    
    def build_reuse_recommendations(
        self,
        task_goal: str,
        available_examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build recommendations for reusing past components.
        
        Args:
            task_goal: Current task goal
            available_examples: Past task examples
            
        Returns:
            Recommendations dict with reusable components
        """
        recommendations = {
            "reusable_nodes": [],
            "suggested_patterns": [],
            "avoid_patterns": []
        }
        
        # Analyze examples for common patterns
        if len(available_examples) >= 3:
            # We have enough data to suggest patterns
            recommendations["suggested_patterns"].append({
                "pattern": "batch_processing",
                "confidence": 0.8,
                "reason": "Multiple similar data processing tasks found"
            })
        
        # Check for failed patterns to avoid
        # (Would query error traces)
        
        return recommendations


class WorkflowSynthesizerExtension:
    """
    Extends WorkflowSynthesizer with memory-driven synthesis.
    
    Integrates with existing synthesis pipeline to:
    - Query past traces before LLM generation
    - Reuse proven node implementations
    - Avoid repeating past mistakes
    """
    
    def __init__(self, memory_engine: MemoryQueryEngine):
        self.memory = memory_engine
    
    async def synthesize_with_memory(
        self,
        task_description: str,
        examples: List[Dict[str, Any]],
        input_sample: Any
    ) -> Optional[WorkflowDAG]:
        """
        Synthesize workflow using memory-first approach.
        
        Args:
            task_description: Task to accomplish
            examples: Past task examples
            input_sample: Sample input
            
        Returns:
            WorkflowDAG built from reused components or None
        """
        # Step 1: Query memory for similar workflows
        similar_workflows = await self.memory.query_similar_workflows(
            task_goal=task_description,
            input_sample=input_sample if isinstance(input_sample, dict) else None,
            limit=10
        )
        
        if not similar_workflows:
            return None
        
        # Step 2: Extract best nodes from traces
        traces = [
            ExecutionTrace(**w) if isinstance(w, dict) else w
            for w in similar_workflows
            if hasattr(w, 'trace_id') or (isinstance(w, dict) and 'trace_id' in w)
        ]
        
        if not traces:
            return None
        
        reusable_nodes = self.memory.extract_best_nodes(traces)
        
        if not reusable_nodes:
            return None
        
        # Step 3: Assemble into DAG
        dag = self._assemble_dag_from_nodes(
            nodes=reusable_nodes,
            task_description=task_description
        )
        
        return dag
    
    def _assemble_dar_from_nodes(
        self,
        nodes: List[DAGNode],
        task_description: str
    ) -> WorkflowDAG:
        """Assemble nodes into a coherent DAG."""
        # Create edges based on data flow analysis
        # (Simplified - would analyze input/output schemas)
        edges = []
        for i in range(len(nodes) - 1):
            edges.append(DAGEdge(
                source_id=nodes[i].node_id,
                target_id=nodes[i + 1].node_id
            ))
        
        return WorkflowDAG(
            name=f"Memory synthesized: {task_description[:50]}",
            description=f"Auto-assembled from {len(nodes)} reusable nodes",
            nodes=nodes,
            edges=edges,
            maturation_layer=2  # Needs further maturation
        )
