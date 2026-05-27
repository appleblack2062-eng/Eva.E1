"""Immutable ledger of execution feedback."""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class FeedbackMemoryLayer:
    """Append-only log for training data."""
    
    def __init__(self, agent_id: str, config):
        self.path = Path(config.base_storage_path) / agent_id / "feedback_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    async def store(self, signal: Dict[str, Any]):
        with open(self.path, 'a') as f:
            f.write(json.dumps(signal, default=str) + "\n")
    
    async def store_critical_error(self, task_id: str, error: str, traceback: bool=False):
        await self.store({
            "type": "CRITICAL_ERROR",
            "task_id": task_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    async def store_validation_failure(self, task_description: str, workflow_id: str, failures: List):
        await self.store({
            "type": "VALIDATION_FAIL",
            "description": task_description,
            "workflow_id": workflow_id,
            "failures": failures
        })
        
    async def store_optimization_success(self, **kwargs):
        await self.store({"type": "OPTIMIZATION_SUCCESS", **kwargs})
        
    async def store_optimization_error(self, error: str):
        await self.store({"type": "OPTIMIZATION_ERROR", "error": error})
        
    async def flush(self):
        pass # File is append-only
