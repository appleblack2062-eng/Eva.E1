"""Graph storage for workflow dependencies and task patterns."""

from __future__ import annotations
import networkx as nx
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class WorkflowGraphStore:
    """Maintains relationships between tasks, workflows, and tools."""
    
    def __init__(self, agent_id: str, config):
        self.agent_id = agent_id
        self.config = config
        self.graph = nx.DiGraph()
        self._storage_path = Path(config.base_storage_path) / agent_id / "workflow_graph.json"
        self._load_graph()
    
    def _load_graph(self):
        if self._storage_path.exists():
            try:
                data = json.loads(self._storage_path.read_text())
                self.graph = nx.node_link_graph(data)
            except Exception:
                self.graph = nx.DiGraph()
    
    def save_graph(self):
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        self._storage_path.write_text(json.dumps(data, indent=2))
    
    def add_workflow_node(self, workflow_id: str, metadata: Dict[str, Any]):
        self.graph.add_node(workflow_id, type="workflow", **metadata)
        self.save_graph()
    
    def add_task_pattern_node(self, pattern_hash: str, metadata: Dict[str, Any]):
        self.graph.add_node(pattern_hash, type="pattern", **metadata)
        self.save_graph()
    
    def link_pattern_to_workflow(self, pattern_hash: str, workflow_id: str, confidence: float):
        self.graph.add_edge(pattern_hash, workflow_id, 
                           relation="implements", 
                           confidence=confidence,
                           weight=confidence)
        self.save_graph()
    
    def get_workflows_for_pattern(self, pattern_hash: str) -> List[Dict[str, Any]]:
        """Find workflows associated with a task pattern."""
        neighbors = list(self.graph.successors(pattern_hash))
        results = []
        for wf_id in neighbors:
            if self.graph.has_node(wf_id):
                node_data = self.graph.nodes[wf_id]
                edge_data = self.graph.get_edge_data(pattern_hash, wf_id)
                results.append({
                    "workflow_id": wf_id,
                    "confidence": edge_data.get("confidence", 0),
                    "metadata": node_data
                })
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    def get_dependency_chain(self, workflow_id: str) -> List[str]:
        """Get all tools/sub-workflows required by a workflow."""
        try:
            return list(nx.descendants(self.graph, workflow_id))
        except nx.NetworkXError:
            return []
