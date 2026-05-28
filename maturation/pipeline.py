"""Maturation Pipeline for Polyglot DAG Workflows."""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import json

try:
    from ..orchestration.polyglot.models import DAGNode, DAGEdge, WorkflowDAG
except ImportError:
    from orchestration.polyglot.models import DAGNode, DAGEdge, WorkflowDAG


class MaturationPipeline:
    """
    Grows workflows through layered maturation.
    
    Each layer adds optimization and reliability:
    - L1: Logical DAG (abstract graph)
    - L2: Runtime Mapping (polyglot assignments)
    - L3: Concurrency & Fusion (parallel waves + I/O fusion)
    - L4: Hardening (retries, timeouts, schema guards)
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def mature_workflow(
        self,
        goal: str,
        initial_dag: Optional[WorkflowDAG] = None
    ) -> WorkflowDAG:
        """
        Run a DAG through all maturation layers.
        
        Args:
            goal: High-level workflow goal
            initial_dag: Optional starting DAG
            
        Returns:
            Production-ready WorkflowDAG at Layer 4
        """
        # L1: Generate or use provided logical DAG
        if initial_dag:
            dag = initial_dag
        else:
            dag = await self._generate_logical_dag(goal)
        
        print(f"[Maturation] L1 Complete: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
        
        # L2: Assign runtimes to nodes
        dag = self._assign_runtimes(dag)
        print(f"[Maturation] L2 Complete: Runtimes assigned")
        
        # L3: Optimize concurrency and fuse nodes
        dag = self._optimize_concurrency(dag)
        dag = self._fuse_sequential_nodes(dag)
        print(f"[Maturation] L3 Complete: Optimized to {len(dag.parallel_groups)} waves")
        
        # L4: Add hardening
        dag = self._harden_dag(dag)
        print(f"[Maturation] L4 Complete: Hardening applied")
        
        dag.maturation_layer = 4
        return dag
    
    async def _generate_logical_dag(self, goal: str) -> WorkflowDAG:
        """L1: Generate abstract DAG from goal (LLM-driven)."""
        if not self.llm:
            # Return minimal placeholder DAG
            return WorkflowDAG(
                name="Generated Workflow",
                description=f"Auto-generated for: {goal}",
                nodes=[],
                edges=[]
            )
        
        # Use LLM to generate DAG structure
        prompt = f"""Generate a workflow DAG for this goal: {goal}

Output JSON with:
- nodes: list of {{name, description, input_requirements, output_format}}
- edges: list of {{from_node, to_node, data_flow}}

Focus on logical steps, not implementation details."""
        
        try:
            response = await self.llm.generate(prompt)
            # Parse response into DAG
            # Placeholder - would parse JSON response
            return WorkflowDAG(
                name="Generated Workflow",
                description=f"Auto-generated for: {goal}",
                nodes=[],
                edges=[]
            )
        except Exception:
            return WorkflowDAG(
                name="Generated Workflow",
                description=f"Auto-generated for: {goal}",
                nodes=[],
                edges=[]
            )
    
    def _assign_runtimes(self, dag: WorkflowDAG) -> WorkflowDAG:
        """L2: Assign optimal runtime to each node."""
        for node in dag.nodes:
            # Heuristic-based runtime assignment
            runtime = self._classify_node_runtime(node)
            node.runtime = runtime
        
        return dag
    
    def _classify_node_runtime(self, node: DAGNode) -> str:
        """Classify node into optimal runtime category."""
        name_lower = node.name.lower()
        desc_lower = getattr(node, 'description', '').lower() if hasattr(node, 'description') else ''
        payload_lower = node.payload.lower()
        
        # BASH: I/O operations, CLI tools, piping
        bash_keywords = ['curl', 'wget', 'jq', 'awk', 'grep', 'sed', 'sort', 
                        'pipe', 'http', 'fetch', 'download', 'file', 'bash']
        if any(kw in name_lower or kw in desc_lower or kw in payload_lower 
               for kw in bash_keywords):
            return "BASH"
        
        # PYTHON: Math, complex algorithms, data processing
        python_keywords = ['math', 'calculate', 'transform', 'pandas', 'numpy',
                          'algorithm', 'process', 'dataframe', 'array']
        if any(kw in name_lower or kw in desc_lower or kw in payload_lower 
               for kw in python_keywords):
            return "PYTHON"
        
        # LLM: Reasoning, extraction, generation, ambiguity
        llm_keywords = ['extract', 'summarize', 'generate', 'reason', 'analyze',
                       'interpret', 'understand', 'classify', 'sentiment']
        if any(kw in name_lower or kw in desc_lower or kw in payload_lower 
               for kw in llm_keywords):
            return "LLM"
        
        # CLI_TOOL: Specific binaries
        cli_tools = ['ffmpeg', 'git', 'pandoc', 'docker', 'kubectl']
        if any(tool in payload_lower for tool in cli_tools):
            return "CLI_TOOL"
        
        # Default to PYTHON for general computation
        return "PYTHON"
    
    def _optimize_concurrency(self, dag: WorkflowDAG) -> WorkflowDAG:
        """L3: Identify parallel execution groups."""
        # Compute topological levels
        in_degree = {node.node_id: 0 for node in dag.nodes}
        adj = {}
        
        for node in dag.nodes:
            adj[node.node_id] = []
        
        for edge in dag.edges:
            in_degree[edge.target_id] += 1
            if edge.source_id in adj:
                adj[edge.source_id].append(edge.target_id)
        
        # Group by level
        levels = []
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        
        while queue:
            levels.append(list(queue))
            next_queue = []
            for node_id in queue:
                for neighbor in adj.get(node_id, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue
        
        dag.parallel_groups = levels
        return dag
    
    def _fuse_sequential_nodes(self, dag: WorkflowDAG) -> WorkflowDAG:
        """L3: Fuse sequential BASH nodes into single scripts."""
        # Find chains of sequential BASH nodes
        fused_count = 0
        
        # Build adjacency info
        successors = {}
        predecessors = {}
        for node in dag.nodes:
            successors[node.node_id] = []
            predecessors[node.node_id] = []
        
        for edge in dag.edges:
            if edge.source_id in successors:
                successors[edge.source_id].append(edge.target_id)
            if edge.target_id in predecessors:
                predecessors[edge.target_id].append(edge.source_id)
        
        # Find fusible chains (sequential BASH nodes with single predecessor/successor)
        node_map = {node.node_id: node for node in dag.nodes}
        
        for node in dag.nodes:
            if node.runtime != "BASH":
                continue
            
            # Check if this is start of a BASH chain
            if len(predecessors.get(node.node_id, [])) != 1:
                continue
            
            pred_id = predecessors[node.node_id][0]
            pred_node = node_map.get(pred_id)
            
            if not pred_node or pred_node.runtime != "BASH":
                continue
            
            # Check if predecessor has only this successor
            if len(successors.get(pred_id, [])) != 1:
                continue
            
            # Fuse these nodes
            fused_payload = f"{pred_node.payload}\n\n# --- Fused from {node.name} ---\n{node.payload}"
            
            # Update predecessor
            pred_node.payload = fused_payload
            pred_node.name = f"{pred_node.name}_fused_{node.name}"
            pred_node.fusion_candidates.append(node.node_id)
            
            # Remove current node and update edges
            dag.nodes.remove(node)
            
            # Update edges: redirect edges pointing to current node to point to predecessor
            for edge in dag.edges:
                if edge.target_id == node.node_id:
                    edge.target_id = pred_id
                if edge.source_id == node.node_id:
                    edge.source_id = pred_id
            
            # Remove edges between fused nodes
            dag.edges = [e for e in dag.edges 
                        if not (e.source_id == pred_id and e.target_id == node.node_id)]
            
            fused_count += 1
        
        if fused_count > 0:
            print(f"[Maturation] Fused {fused_count} node pairs")
        
        return dag
    
    def _harden_dag(self, dag: WorkflowDAG) -> WorkflowDAG:
        """L4: Inject retries, timeouts, schema guards."""
        for node in dag.nodes:
            # Add retry policy based on runtime
            if node.runtime == "LLM":
                node.retry_policy = {"max_retries": 3, "backoff_ms": 1000}
                node.timeout_ms = max(node.timeout_ms, 10000)
            elif node.runtime == "BASH":
                node.retry_policy = {"max_retries": 2, "backoff_ms": 500}
                node.timeout_ms = max(node.timeout_ms, 5000)
            elif node.runtime == "PYTHON":
                node.retry_policy = {"max_retries": 1, "backoff_ms": 200}
                node.timeout_ms = max(node.timeout_ms, 3000)
            
            # Add schema validation if missing
            if not node.output_schema:
                node.output_schema = self._infer_output_schema(node)
        
        return dag
    
    def _infer_output_schema(self, node: DAGNode) -> Dict[str, Any]:
        """Infer output schema from node payload."""
        # Simple heuristic inference
        payload_lower = node.payload.lower()
        
        if 'json' in payload_lower or 'dict' in payload_lower:
            return {"type": "object"}
        elif 'list' in payload_lower or 'array' in payload_lower:
            return {"type": "array"}
        elif 'return' in payload_lower and 'print' not in payload_lower:
            return {"type": "object"}
        else:
            return {"type": "object"}  # Default
    
    def get_maturation_report(self, dag: WorkflowDAG) -> Dict[str, Any]:
        """Generate report on maturation status."""
        runtime_counts = {}
        for node in dag.nodes:
            runtime_counts[node.runtime] = runtime_counts.get(node.runtime, 0) + 1
        
        return {
            "dag_id": dag.dag_id,
            "maturation_layer": dag.maturation_layer,
            "total_nodes": len(dag.nodes),
            "total_edges": len(dag.edges),
            "parallel_waves": len(dag.parallel_groups),
            "runtime_distribution": runtime_counts,
            "fused_nodes": sum(len(n.fusion_candidates) for n in dag.nodes),
            "nodes_with_retries": sum(
                1 for n in dag.nodes 
                if n.retry_policy.get('max_retries', 0) > 0
            ),
            "nodes_with_schema": sum(
                1 for n in dag.nodes 
                if n.output_schema
            )
        }


class NodeFusionOptimizer:
    """Specialized optimizer for node fusion patterns."""
    
    @staticmethod
    def detect_fusion_opportunities(dag: WorkflowDAG) -> List[Tuple[str, str]]:
        """Detect opportunities for node fusion."""
        opportunities = []
        
        # Build graph representation
        successors = {}
        for node in dag.nodes:
            successors[node.node_id] = []
        
        for edge in dag.edges:
            if edge.source_id in successors:
                successors[edge.source_id].append(edge.target_id)
        
        node_map = {node.node_id: node for node in dag.nodes}
        
        # Find sequential same-runtime nodes
        for node in dag.nodes:
            succs = successors.get(node.node_id, [])
            if len(succs) == 1:
                succ_node = node_map.get(succs[0])
                if succ_node and succ_node.runtime == node.runtime:
                    # Check if succ has only this predecessor
                    # (simplified check)
                    opportunities.append((node.node_id, succs[0]))
        
        return opportunities
    
    @staticmethod
    def estimate_fusion_benefit(node1: DAGNode, node2: DAGNode) -> Dict[str, float]:
        """Estimate performance benefit from fusing two nodes."""
        # Estimate subprocess overhead saved
        subprocess_overhead_ms = 2.0  # ~2ms per subprocess spawn
        
        # Estimate disk I/O saved (if any)
        io_savings_ms = 0.5  # ~0.5ms per eliminated intermediate write
        
        total_latency_reduction = subprocess_overhead_ms + io_savings_ms
        percentage_reduction = (total_latency_reduction / max(
            node1.timeout_ms + node2.timeout_ms, 1
        )) * 100
        
        return {
            "latency_reduction_ms": total_latency_reduction,
            "percentage_improvement": percentage_reduction,
            "subprocess_savings": 1,  # One less subprocess
            "io_savings": 1  # One less intermediate write
        }
