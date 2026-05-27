"""Stores synthesized and optimized workflows."""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..stores.graph_store import WorkflowGraphStore
from ...core.memory_types import WorkflowSpec

class WorkflowMemoryLayer:
    """Manages lifecycle of workflows."""
    
    def __init__(self, agent_id: str, config):
        self.agent_id = agent_id
        self.config = config
        self._graph_store: Optional[WorkflowGraphStore] = None
        self._workflows: Dict[str, WorkflowSpec] = {}
        self._storage_path = Path(config.base_storage_path) / agent_id / "workflows"
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def set_dependencies(self, graph_store: WorkflowGraphStore):
        self._graph_store = graph_store

    async def deploy_workflow(self, workflow: WorkflowSpec, task_pattern: Dict, performance_baseline: Dict) -> str:
        """Save and register a new workflow."""
        wf_id = workflow.id if hasattr(workflow, 'id') else f"wf_{int(time.time())}"
        
        # Save to disk
        file_path = self._storage_path / f"{wf_id}.json"
        with open(file_path, 'w') as f:
            # Convert Pydantic model to dict for JSON serialization
            f.write(json.dumps(workflow.model_dump(), default=str))
        
        # Register in graph
        if self._graph_store:
            self._graph_store.add_workflow_node(wf_id, {
                "name": workflow.name,
                "version": workflow.version,
                "created_at": time.time(),
                "baseline_latency": performance_baseline.get("latency_baseline", 0)
            })
            
            pattern_hash = hash(json.dumps(task_pattern, sort_keys=True)) % (10**8)
            pattern_str = f"pattern_{pattern_hash}"
            self._graph_store.add_task_pattern_node(pattern_str, task_pattern)
            self._graph_store.link_pattern_to_workflow(pattern_str, wf_id, 0.9)

        self._workflows[wf_id] = workflow
        return wf_id

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowSpec]:
        """Retrieve a workflow by ID."""
        if workflow_id in self._workflows:
            return self._workflows[workflow_id]
        
        # Load from disk if not in memory
        file_path = self._storage_path / f"{workflow_id}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # Reconstruct WorkflowSpec (simplified)
                wf = WorkflowSpec(**data)
                self._workflows[workflow_id] = wf
                return wf
            except Exception:
                return None
        return None

    async def has_recent_optimization(self, description: str, hours: int) -> bool:
        """Check if we recently tried to optimize this type of task."""
        # Simplified: Check if any workflow was created in last N hours
        cutoff = time.time() - (hours * 3600)
        for wf in self._workflows.values():
            if getattr(wf, 'metadata', {}).get('generation_timestamp', 0) > cutoff:
                return True
        return False

    async def count_active_workflows(self) -> int:
        return len(self._workflows)

    async def compute_reuse_rate(self) -> float:
        # Placeholder for metric calculation
        return 0.0

    async def list_workflows(self, pattern_filter: Optional[str] = None) -> List[Dict]:
        return [{"id": k, "name": v.name} for k, v in self._workflows.items()]
    
    async def flush(self):
        """Persist all in-memory workflows."""
        pass # Already persisted on deploy
