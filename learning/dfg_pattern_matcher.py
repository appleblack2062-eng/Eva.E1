"""Isomorphic Data-Flow Graph Matching for structural pattern recognition.

This module moves beyond purely text-based embedding matching to recognize
structural isomorphism between workflow data-flow graphs, enabling the system
to repurpose compiled workflow templates across semantically different but
structurally identical tasks.
"""

from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from .pattern_engine import WorkflowRef


class NodeOperation(Enum):
    """Standardized data-flow operations for structural matching."""
    LOAD = "LOAD"
    FILTER = "FILTER"
    MAP = "MAP"
    REDUCE = "REDUCE"
    AGGREGATE = "AGGREGATE"
    TRANSFORM = "TRANSFORM"
    JOIN = "JOIN"
    SPLIT = "SPLIT"
    VALIDATE = "VALIDATE"
    SERIALIZE = "SERIALIZE"
    RETURN = "RETURN"
    CONDITIONAL = "CONDITIONAL"
    LOOP = "LOOP"


@dataclass
class DFGNode:
    """Represents a node in a Data-Flow Graph."""
    node_id: str
    operation: NodeOperation
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_signature(self) -> str:
        """Get structural signature of this node (ignoring specific values)."""
        return f"{self.operation.value}|{json.dumps(sorted(self.parameters.keys()))}"


@dataclass
class DFGEdge:
    """Represents an edge in a Data-Flow Graph."""
    source_node: str
    target_node: str
    data_type: str = "any"
    transformation: Optional[str] = None


@dataclass
class DataFlowGraph:
    """
    Represents a workflow as a Data-Flow Graph for structural matching.
    
    Unlike semantic embeddings that capture textual similarity, DFGs capture
    the structural flow of data through operations, enabling recognition that:
    - "Filter CSV rows where age < 21" and "Remove sensor readings below 5V" 
      are structurally isomorphic (both are LOAD -> FILTER -> RETURN)
    - Different domains can share the same computational patterns
    """
    graph_id: str
    nodes: Dict[str, DFGNode] = field(default_factory=dict)
    edges: List[DFGEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _networkx_graph: Optional[Any] = None
    
    def add_node(self, node: DFGNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        self._invalidate_cache()
    
    def add_edge(self, edge: DFGEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self._invalidate_cache()
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached NetworkX graph."""
        self._networkx_graph = None
    
    def to_networkx(self) -> Optional[Any]:
        """Convert to NetworkX DiGraph for algorithm support."""
        if not NETWORKX_AVAILABLE:
            return None
        
        if self._networkx_graph is None:
            G = nx.DiGraph()
            
            for node_id, node in self.nodes.items():
                G.add_node(
                    node_id,
                    operation=node.operation.value,
                    signature=node.get_signature(),
                    **node.metadata
                )
            
            for edge in self.edges:
                G.add_edge(
                    edge.source_node,
                    edge.target_node,
                    data_type=edge.data_type,
                    transformation=edge.transformation or ""
                )
            
            self._networkx_graph = G
        
        return self._networkx_graph
    
    def get_canonical_form(self) -> str:
        """
        Get canonical string representation of graph structure.
        
        This enables fast hashing and comparison of graph structures.
        """
        if not NETWORKX_AVAILABLE or self.to_networkx() is None:
            # Fallback: simple serialization
            node_sigs = sorted([n.get_signature() for n in self.nodes.values()])
            edge_sigs = sorted([(e.source_node, e.target_node) for e in self.edges])
            return json.dumps({"nodes": node_sigs, "edges": edge_sigs}, sort_keys=True)
        
        # Use NetworkX graph hashing
        G = self.to_networkx()
        
        # Create structural hash based on node signatures and connectivity
        structure = []
        for node_id in sorted(G.nodes()):
            node_data = G.nodes[node_id]
            sig = node_data.get('signature', '')
            predecessors = sorted(G.predecessors(node_id))
            successors = sorted(G.successors(node_id))
            structure.append({
                'node': node_id,
                'sig': sig,
                'pred': predecessors,
                'succ': successors
            })
        
        return json.dumps(structure, sort_keys=True)
    
    def compute_hash(self) -> str:
        """Compute hash of graph structure for fast lookup."""
        canonical = self.get_canonical_form()
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    @classmethod
    def from_workflow_spec(cls, workflow_spec: Any, graph_id: Optional[str] = None) -> 'DataFlowGraph':
        """
        Convert a workflow specification to a Data-Flow Graph.
        
        Args:
            workflow_spec: Workflow specification object or dict
            graph_id: Optional graph identifier
            
        Returns:
            DataFlowGraph representation
        """
        if graph_id is None:
            graph_id = f"dfg_{int(time.time())}"
        
        dfg = DataFlowGraph(graph_id=graph_id)
        
        # Extract nodes from workflow steps
        if hasattr(workflow_spec, 'steps'):
            steps = workflow_spec.steps
        elif isinstance(workflow_spec, dict) and 'steps' in workflow_spec:
            steps = workflow_spec['steps']
        else:
            steps = []
        
        prev_node_id = None
        for i, step in enumerate(steps):
            node = cls._step_to_dfg_node(step, i)
            dfg.add_node(node)
            
            # Add edge from previous node
            if prev_node_id:
                dfg.add_edge(DFGEdge(
                    source_node=prev_node_id,
                    target_node=node.node_id,
                    data_type="stream"
                ))
            
            prev_node_id = node.node_id
        
        return dfg
    
    @classmethod
    def _step_to_dfg_node(cls, step: Any, index: int) -> DFGNode:
        """Convert a workflow step to a DFG node."""
        # Extract operation type
        if hasattr(step, 'operation'):
            op_str = step.operation
        elif isinstance(step, dict):
            op_str = step.get('operation', 'TRANSFORM')
        else:
            op_str = 'TRANSFORM'
        
        # Map to standard operation
        operation = cls._map_operation(op_str)
        
        # Extract parameters
        if hasattr(step, 'parameters'):
            params = step.parameters
        elif isinstance(step, dict):
            params = step.get('parameters', {})
        else:
            params = {}
        
        return DFGNode(
            node_id=f"step_{index}",
            operation=operation,
            parameters=params
        )
    
    @staticmethod
    def _map_operation(op_str: str) -> NodeOperation:
        """Map operation string to standard NodeOperation."""
        op_map = {
            'LOAD': NodeOperation.LOAD,
            'READ': NodeOperation.LOAD,
            'FETCH': NodeOperation.LOAD,
            'FILTER': NodeOperation.FILTER,
            'WHERE': NodeOperation.FILTER,
            'MAP': NodeOperation.MAP,
            'TRANSFORM': NodeOperation.TRANSFORM,
            'CONVERT': NodeOperation.TRANSFORM,
            'REDUCE': NodeOperation.REDUCE,
            'AGGREGATE': NodeOperation.AGGREGATE,
            'SUM': NodeOperation.AGGREGATE,
            'COUNT': NodeOperation.AGGREGATE,
            'GROUP': NodeOperation.AGGREGATE,
            'JOIN': NodeOperation.JOIN,
            'MERGE': NodeOperation.JOIN,
            'SPLIT': NodeOperation.SPLIT,
            'BRANCH': NodeOperation.SPLIT,
            'VALIDATE': NodeOperation.VALIDATE,
            'CHECK': NodeOperation.VALIDATE,
            'SERIALIZE': NodeOperation.SERIALIZE,
            'OUTPUT': NodeOperation.SERIALIZE,
            'RETURN': NodeOperation.RETURN,
            'IF': NodeOperation.CONDITIONAL,
            'SWITCH': NodeOperation.CONDITIONAL,
            'LOOP': NodeOperation.LOOP,
            'FOREACH': NodeOperation.LOOP,
        }
        
        return op_map.get(op_str.upper(), NodeOperation.TRANSFORM)


class IsomorphicMatcher:
    """
    Performs isomorphic matching between Data-Flow Graphs.
    
    Uses graph isomorphism algorithms to determine if two workflows
    have the same structural pattern, regardless of their domain-specific
    details.
    """
    
    def __init__(self):
        if not NETWORKX_AVAILABLE:
            print("Warning: networkx not available. Isomorphic matching will use fallback mode.")
    
    def are_isomorphic(
        self,
        graph1: DataFlowGraph,
        graph2: DataFlowGraph,
        strict: bool = False
    ) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Check if two DFGs are isomorphic (structurally identical).
        
        Args:
            graph1: First data-flow graph
            graph2: Second data-flow graph
            strict: If True, require exact operation match; if False, allow semantic grouping
            
        Returns:
            Tuple of (is_isomorphic, node_mapping)
        """
        if not NETWORKX_AVAILABLE:
            return self._fallback_isomorphic_check(graph1, graph2)
        
        G1 = graph1.to_networkx()
        G2 = graph2.to_networkx()
        
        if G1 is None or G2 is None:
            return self._fallback_isomorphic_check(graph1, graph2)
        
        # Define node matching function
        def node_match(n1, n2):
            if strict:
                # Exact operation match required
                return n1.get('operation') == n2.get('operation')
            else:
                # Allow semantic grouping (e.g., FILTER ~ WHERE)
                op1 = n1.get('operation')
                op2 = n2.get('operation')
                return self._operations_compatible(op1, op2)
        
        # Check for isomorphism
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(G1, G2, node_match=node_match)
        
        if matcher.is_isomorphic():
            return True, matcher.mapping
        else:
            return False, None
    
    def _operations_compatible(self, op1: str, op2: str) -> bool:
        """Check if two operations are semantically compatible."""
        # Group similar operations
        filter_ops = {NodeOperation.FILTER.value, NodeOperation.FILTER.value}
        transform_ops = {NodeOperation.MAP.value, NodeOperation.TRANSFORM.value}
        aggregate_ops = {NodeOperation.REDUCE.value, NodeOperation.AGGREGATE.value}
        
        if op1 in filter_ops and op2 in filter_ops:
            return True
        if op1 in transform_ops and op2 in transform_ops:
            return True
        if op1 in aggregate_ops and op2 in aggregate_ops:
            return True
        
        return op1 == op2
    
    def _fallback_isomorphic_check(
        self,
        graph1: DataFlowGraph,
        graph2: DataFlowGraph
    ) -> Tuple[bool, Optional[Dict[str, str]]]:
        """Fallback isomorphism check without networkx."""
        # Compare canonical forms
        if graph1.get_canonical_form() == graph2.get_canonical_form():
            # Simple 1:1 mapping
            mapping = {n1: n2 for n1, n2 in zip(
                sorted(graph1.nodes.keys()),
                sorted(graph2.nodes.keys())
            )}
            return True, mapping
        
        return False, None
    
    def find_subgraph_isomorphism(
        self,
        pattern: DataFlowGraph,
        target: DataFlowGraph
    ) -> List[Dict[str, str]]:
        """
        Find all occurrences of a pattern graph within a target graph.
        
        Args:
            pattern: Pattern to search for
            target: Target graph to search in
            
        Returns:
            List of node mappings for each match
        """
        if not NETWORKX_AVAILABLE:
            return []
        
        G_pattern = pattern.to_networkx()
        G_target = target.to_networkx()
        
        if G_pattern is None or G_target is None:
            return []
        
        # Find all subgraph isomorphisms
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            G_target, G_pattern,
            node_match=lambda n1, n2: n1.get('operation') == n2.get('operation')
        )
        
        matches = []
        for mapping in matcher.subgraph_monomorphisms_iter():
            # Invert mapping to get pattern->target
            inverted = {v: k for k, v in mapping.items()}
            matches.append(inverted)
        
        return matches
    
    def compute_graph_similarity(
        self,
        graph1: DataFlowGraph,
        graph2: DataFlowGraph
    ) -> float:
        """
        Compute similarity score between two graphs (0.0 to 1.0).
        
        Useful when graphs are not perfectly isomorphic but share structure.
        """
        if not NETWORKX_AVAILABLE:
            return self._fallback_similarity(graph1, graph2)
        
        G1 = graph1.to_networkx()
        G2 = graph2.to_networkx()
        
        if G1 is None or G2 is None:
            return self._fallback_similarity(graph1, graph2)
        
        # Calculate graph edit distance (expensive but accurate)
        # For large graphs, use approximation
        if len(G1.nodes()) > 20 or len(G2.nodes()) > 20:
            return self._approximate_similarity(G1, G2)
        
        try:
            edit_distance = nx.graph_edit_distance(G1, G2)
            max_size = max(len(G1.nodes()) + len(G1.edges()), len(G2.nodes()) + len(G2.edges()))
            if max_size == 0:
                return 1.0
            similarity = 1.0 - (edit_distance / max_size)
            return max(0.0, min(1.0, similarity))
        except Exception:
            return self._fallback_similarity(graph1, graph2)
    
    def _approximate_similarity(self, G1: Any, G2: Any) -> float:
        """Approximate graph similarity for large graphs."""
        # Compare node operation distributions
        ops1 = defaultdict(int)
        ops2 = defaultdict(int)
        
        for node_data in G1.nodes(data=True):
            ops1[node_data[1].get('operation', '')] += 1
        
        for node_data in G2.nodes(data=True):
            ops2[node_data[1].get('operation', '')] += 1
        
        # Calculate overlap
        all_ops = set(ops1.keys()) | set(ops2.keys())
        if not all_ops:
            return 1.0
        
        total_diff = sum(abs(ops1.get(op, 0) - ops2.get(op, 0)) for op in all_ops)
        total_count = sum(ops1.values()) + sum(ops2.values())
        
        if total_count == 0:
            return 1.0
        
        return 1.0 - (total_diff / total_count)
    
    def _fallback_similarity(
        self,
        graph1: DataFlowGraph,
        graph2: DataFlowGraph
    ) -> float:
        """Fallback similarity without networkx."""
        # Compare node counts and operation types
        if len(graph1.nodes) != len(graph2.nodes):
            size_diff = abs(len(graph1.nodes) - len(graph2.nodes)) / max(len(graph1.nodes), len(graph2.nodes))
        else:
            size_diff = 0
        
        # Compare operation distributions
        ops1 = defaultdict(int)
        ops2 = defaultdict(int)
        
        for node in graph1.nodes.values():
            ops1[node.operation.value] += 1
        
        for node in graph2.nodes.values():
            ops2[node.operation.value] += 1
        
        all_ops = set(ops1.keys()) | set(ops2.keys())
        if not all_ops:
            return 1.0
        
        op_diff = sum(abs(ops1.get(op, 0) - ops2.get(op, 0)) for op in all_ops)
        op_total = sum(ops1.values()) + sum(ops2.values())
        
        if op_total == 0:
            return 1.0
        
        return 1.0 - (size_diff * 0.5 + (op_diff / op_total) * 0.5)


class StructuralPatternEngine:
    """
    Pattern engine that uses structural DFG matching instead of just text embeddings.
    
    This complements the existing PatternEngine by adding structural recognition
    capabilities, enabling the system to:
    - Recognize when different tasks have the same computational structure
    - Reuse optimized workflows across domains
    - Accelerate learning by transferring structural patterns
    """
    
    def __init__(self, config, embedding_provider=None):
        self.config = config
        self.isomorphic_matcher = IsomorphicMatcher()
        self._dfg_cache: Dict[str, DataFlowGraph] = {}  # workflow_id -> DFG
        self._structure_index: Dict[str, List[str]] = defaultdict(list)  # structure_hash -> [workflow_ids]
        
        # Optionally integrate with existing pattern engine
        if embedding_provider:
            from .pattern_engine import PatternGeneralizationEngine
            self.semantic_engine = PatternGeneralizationEngine(config, embedding_provider)
        else:
            self.semantic_engine = None
    
    async def index_workflow(
        self,
        workflow_id: str,
        workflow_spec: Any,
        task_description: str
    ) -> None:
        """Index a workflow by its structural pattern."""
        # Convert workflow to DFG
        dfg = DataFlowGraph.from_workflow_spec(workflow_spec, workflow_id)
        self._dfg_cache[workflow_id] = dfg
        
        # Index by structure hash
        structure_hash = dfg.compute_hash()
        self._structure_index[structure_hash].append(workflow_id)
        
        # Also index with semantic engine if available
        if self.semantic_engine:
            # Would need input schema for full indexing
            pass
    
    async def find_structural_matches(
        self,
        query_workflow: Any,
        min_similarity: float = 0.8,
        limit: int = 10
    ) -> List[Tuple[str, float, DataFlowGraph]]:
        """
        Find workflows with similar structure to the query.
        
        Args:
            query_workflow: Query workflow specification
            min_similarity: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            List of (workflow_id, similarity_score, DFG) tuples
        """
        # Convert query to DFG
        query_dfg = DataFlowGraph.from_workflow_spec(query_workflow, "query")
        
        results = []
        
        # First check for exact structural matches (fast)
        query_hash = query_dfg.compute_hash()
        if query_hash in self._structure_index:
            for wf_id in self._structure_index[query_hash]:
                if wf_id in self._dfg_cache:
                    results.append((wf_id, 1.0, self._dfg_cache[wf_id]))
        
        # Then check for approximate matches
        for wf_id, stored_dfg in self._dfg_cache.items():
            if wf_id in [r[0] for r in results]:
                continue  # Already found as exact match
            
            similarity = self.isomorphic_matcher.compute_graph_similarity(query_dfg, stored_dfg)
            
            if similarity >= min_similarity:
                results.append((wf_id, similarity, stored_dfg))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    async def adapt_workflow_template(
        self,
        template_workflow: Any,
        target_task_description: str,
        target_input_sample: Any
    ) -> Any:
        """
        Adapt a structural workflow template to a new task.
        
        This is the key capability: taking a workflow that solves one problem
        and adapting it to solve a structurally similar problem in a different
        domain.
        
        Args:
            template_workflow: Source workflow to adapt
            target_task_description: Description of target task
            target_input_sample: Sample input for target task
            
        Returns:
            Adapted workflow specification
        """
        # Analyze structural pattern
        template_dfg = DataFlowGraph.from_workflow_spec(template_workflow)
        
        # Generate adaptation plan using LLM
        if hasattr(self.config, 'llm_client') and self.config.llm_client:
            adapted_spec = await self._llm_adapt_template(
                template_workflow,
                template_dfg,
                target_task_description,
                target_input_sample
            )
            return adapted_spec
        
        # Fallback: return template as-is
        return template_workflow
    
    async def _llm_adapt_template(
        self,
        template: Any,
        template_dfg: DataFlowGraph,
        target_desc: str,
        target_input: Any
    ) -> Any:
        """Use LLM to adapt template to target domain."""
        # Serialize structural pattern
        structure_summary = []
        for node_id, node in template_dfg.nodes.items():
            structure_summary.append(f"- {node.operation.value}: {json.dumps(node.parameters)}")
        
        prompt = f"""
You are adapting a workflow template to a new domain.

ORIGINAL WORKFLOW STRUCTURE:
{chr(10).join(structure_summary)}

TARGET TASK: {target_desc}
TARGET INPUT SAMPLE: {json.dumps(target_input, default=str)[:500]}

Your task:
1. Keep the same structural pattern (same sequence of operations)
2. Adapt the parameters, field names, and thresholds to the target domain
3. Return the adapted workflow specification as JSON

Example adaptation:
- Original: FILTER condition="age < 21"
- Adapted: FILTER condition="voltage < 5.0"

Return ONLY the adapted workflow spec as valid JSON.
"""
        
        llm_client = getattr(self.config, 'llm_client', None)
        if llm_client:
            response = await llm_client.generate(prompt, max_tokens=1500, response_format="json")
            # Parse and return adapted spec
            if hasattr(response, 'content'):
                import json as json_module
                try:
                    return json_module.loads(response.content)
                except:
                    pass
        
        return template


def create_structural_pattern_engine(config, embedding_provider=None) -> StructuralPatternEngine:
    """Factory function to create structural pattern engine."""
    return StructuralPatternEngine(config, embedding_provider)
