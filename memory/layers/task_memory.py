"""Stores history of executed tasks for pattern matching."""

from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional
from ..stores.vector_store import TaskVectorStore

class TaskMemoryLayer:
    """High-level interface for task history."""
    
    def __init__(self, agent_id: str, config):
        self.agent_id = agent_id
        self.config = config
        # Lazy init to avoid circular imports if needed
        self._store: Optional[TaskVectorStore] = None
        self._embedder = None # Injected from brain
    
    def set_dependencies(self, store: TaskVectorStore, embedder):
        self._store = store
        self._embedder = embedder

    async def store_result(self, task_id: str, description: str, input_data: Any, 
                           output: Any, execution_mode, success: bool, 
                           latency_ms: float, tokens_used: int, workflow_id: Optional[str]):
        """Save a completed task to memory."""
        if not self._store: return
        
        metadata = {
            "success": success,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "mode": execution_mode.name if hasattr(execution_mode, 'name') else str(execution_mode),
            "workflow_id": workflow_id or "",
            "output_preview": str(output)[:100] if output else ""
        }
        
        await self._store.add_task(task_id, description, input_data, metadata)

    async def find_similar(self, query: str, input_context: Any, limit: int, min_similarity: float) -> List[Dict]:
        """Retrieve similar past tasks."""
        if not self._store: return []
        return await self._store.search_similar(query, limit, min_similarity)
