"""Component Library & Multi-Task Synthesis for atomic workflow reuse.

This module provides a registry of reusable atomic operations and composes
workflows from existing components before falling back to LLM synthesis.
"""

from __future__ import annotations
import networkx as nx
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
import hashlib


@dataclass
class ComponentSpec:
    """Specification for a reusable atomic component."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    code: str  # Python function code
    dependencies: List[str] = field(default_factory=list)
    success_rate: float = 0.9
    avg_latency_ms: float = 50.0
    avg_cost_usd: float = 0.001
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"


@dataclass
class WorkflowSpec:
    """Complete workflow specification composed from components."""
    id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    steps: List[Dict[str, Any]]
    components_used: List[str]
    estimated_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0


class ComponentLibrary:
    """Registry of reusable atomic operations."""
    
    def __init__(self):
        self.components: Dict[str, ComponentSpec] = {}
        self.dependency_graph = nx.DiGraph()
        self._usage_stats: Dict[str, int] = {}
    
    def register_component(self, spec: ComponentSpec):
        """Add new atomic component to library."""
        self.components[spec.name] = spec
        self._usage_stats[spec.name] = 0
        
        # Parse dependencies from spec.code imports
        deps = self._extract_dependencies(spec.code)
        for dep in deps:
            self.dependency_graph.add_edge(spec.name, dep)
    
    def find_components_for_task(self, task_spec: Dict[str, Any]) -> List[ComponentSpec]:
        """Match task requirements to available components."""
        candidates = []
        
        for comp in self.components.values():
            if self._matches_requirement(comp, task_spec):
                candidates.append(comp)
        
        # Rank by specificity and past success rate
        return sorted(candidates, key=lambda c: c.success_rate, reverse=True)
    
    def compose_workflow(
        self, 
        task_spec: Dict[str, Any], 
        available_components: List[ComponentSpec]
    ) -> Optional[WorkflowSpec]:
        """Try to build workflow from existing components."""
        if not available_components:
            return None
        
        # Graph search: find sequence of components that satisfies task_spec
        workflow = self._graph_search_composition(task_spec, available_components)
        
        if workflow:
            # Update usage stats
            for comp_name in workflow.components_used:
                self._usage_stats[comp_name] = self._usage_stats.get(comp_name, 0) + 1
        
        return workflow
    
    def get_component(self, name: str) -> Optional[ComponentSpec]:
        """Get component by name."""
        return self.components.get(name)
    
    def get_high_usage_components(self, min_usage: int = 5) -> List[ComponentSpec]:
        """Get components with high usage for optimization."""
        high_usage = [
            name for name, count in self._usage_stats.items() 
            if count >= min_usage
        ]
        return [self.components[name] for name in high_usage if name in self.components]
    
    def _matches_requirement(
        self, 
        comp: ComponentSpec, 
        task_spec: Dict[str, Any]
    ) -> bool:
        """Check if component satisfies a task requirement."""
        # Simple: input/output schema compatibility
        input_compat = self._schema_compatible(
            comp.input_schema, 
            task_spec.get("input_schema", {})
        )
        output_compat = self._schema_compatible(
            task_spec.get("output_schema", {}),
            comp.output_schema
        )
        
        return input_compat and output_compat
    
    def _schema_compatible(
        self, 
        required: Dict[str, Any], 
        provided: Dict[str, Any]
    ) -> bool:
        """Check if provided schema satisfies required schema."""
        if not required:
            return True
        if not provided:
            return False
        
        # Simple type checking
        req_type = required.get("type")
        prov_type = provided.get("type")
        
        if req_type and prov_type:
            if req_type == prov_type:
                return True
            # Handle subtype relationships
            if req_type == "number" and prov_type in ["integer", "float"]:
                return True
        
        # Check properties if both are objects
        if req_type == "object" and prov_type == "object":
            req_props = required.get("properties", {})
            prov_props = provided.get("properties", {})
            
            for prop_name, prop_schema in req_props.items():
                if prop_name not in prov_props:
                    return False
                if not self._schema_compatible(prop_schema, prov_props[prop_name]):
                    return False
        
        return True
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract import dependencies from code."""
        deps = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import '):
                # import module
                parts = line.replace('import ', '').split(',')
                for part in parts:
                    deps.append(part.strip().split('.')[0])
            elif line.startswith('from '):
                # from module import ...
                parts = line.replace('from ', '').split(' import ')
                if len(parts) == 2:
                    deps.append(parts[0].strip().split('.')[0])
        
        return deps
    
    def _graph_search_composition(
        self, 
        task_spec: Dict[str, Any], 
        components: List[ComponentSpec]
    ) -> Optional[WorkflowSpec]:
        """Use A* search to find optimal component composition."""
        # Build component graph
        comp_graph = self._build_component_graph(components)
        
        # Find start components (match input schema)
        input_schema = task_spec.get("input_schema", {})
        start_components = [
            c for c in components 
            if self._schema_compatible(input_schema, c.input_schema)
        ]
        
        # Find end components (match output schema)
        output_schema = task_spec.get("output_schema", {})
        end_components = [
            c for c in components 
            if self._schema_compatible(c.output_schema, output_schema)
        ]
        
        if not start_components or not end_components:
            return None
        
        # BFS/DFS to find path from start to end
        for start in start_components:
            for end in end_components:
                path = self._find_path(comp_graph, start.name, end.name)
                if path:
                    return self._build_workflow_from_path(task_spec, path)
        
        return None
    
    def _build_component_graph(
        self, 
        components: List[ComponentSpec]
    ) -> nx.DiGraph:
        """Build directed graph of component connections."""
        graph = nx.DiGraph()
        
        for comp in components:
            graph.add_node(comp.name, component=comp)
        
        # Add edges based on schema compatibility
        for comp1 in components:
            for comp2 in components:
                if comp1.name != comp2.name:
                    if self._schema_compatible(comp1.output_schema, comp2.input_schema):
                        graph.add_edge(comp1.name, comp2.name)
        
        return graph
    
    def _find_path(
        self, 
        graph: nx.DiGraph, 
        start: str, 
        end: str
    ) -> Optional[List[str]]:
        """Find shortest path between two components."""
        try:
            path = nx.shortest_path(graph, source=start, target=end)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def _build_workflow_from_path(
        self, 
        task_spec: Dict[str, Any], 
        path: List[str]
    ) -> WorkflowSpec:
        """Build workflow specification from component path."""
        steps = []
        total_latency = 0.0
        total_cost = 0.0
        
        for i, comp_name in enumerate(path):
            comp = self.components[comp_name]
            steps.append({
                "step_number": i + 1,
                "component": comp_name,
                "operation": comp.description,
                "input_schema": comp.input_schema,
                "output_schema": comp.output_schema,
            })
            total_latency += comp.avg_latency_ms
            total_cost += comp.avg_cost_usd
        
        return WorkflowSpec(
            id=f"wf_{'_'.join(path)}",
            name=f"Composed workflow: {' -> '.join(path)}",
            description=task_spec.get("description", ""),
            input_schema=task_spec.get("input_schema", {}),
            output_schema=task_spec.get("output_schema", {}),
            steps=steps,
            components_used=path,
            estimated_latency_ms=total_latency,
            estimated_cost_usd=total_cost,
        )
