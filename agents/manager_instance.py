"""Concrete Manager agent wrapper."""

from typing import Any, Dict
from .base_agent import BaseAgent

try:
    from ..orchestration.manager_agent import ManagerAgent as CoreManagerAgent
    from ..workspace.filesystem_graph import FileSystemGraph
except ImportError:
    from orchestration.manager_agent import ManagerAgent as CoreManagerAgent
    from workspace.filesystem_graph import FileSystemGraph


class ManagerInstance(BaseAgent):
    """
    Concrete Manager agent implementation.
    
    Wraps the core ManagerAgent logic and provides the standard
    BaseAgent interface for integration with the broader system.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        llm_client, 
        fs_graph: FileSystemGraph, 
        config
    ):
        """
        Initialize manager instance.
        
        Args:
            agent_id: Unique identifier for this manager
            llm_client: LLM client for generating responses
            fs_graph: File system graph for workspace awareness
            config: Agent configuration object
        """
        super().__init__(agent_id, llm_client, config)
        
        # Initialize core manager
        self.core = CoreManagerAgent(
            agent_id=agent_id,
            llm_client=llm_client,
            fs_graph=fs_graph,
            config=config
        )
        
        self._last_result: Optional[Dict[str, Any]] = None
    
    async def run(self, goal: str) -> Dict[str, Any]:
        """
        Execute a complex goal using the manager/worker architecture.
        
        Args:
            goal: High-level goal description
            
        Returns:
            Dictionary with execution status and results
        """
        print(f"[MANAGER {self.agent_id}] Executing goal: {goal}")
        
        if not await self.start():
            return {
                "status": "failed",
                "error": "Manager is already running",
                "results": {}
            }
        
        try:
            self._current_task = goal
            
            # Execute goal through core manager
            result = await self.core.execute_goal(goal)
            
            self._last_result = result
            
            # Record execution
            success = result.get('status') in ['completed', 'completed_with_errors']
            self._record_task_execution(
                task=goal,
                success=success,
                result=result
            )
            
            return result
            
        except Exception as e:
            error_result = {
                "status": "failed",
                "error": str(e),
                "results": {}
            }
            self._last_result = error_result
            
            self._record_task_execution(
                task=goal,
                success=False,
                result=None,
                metadata={"error": str(e)}
            )
            
            raise
            
        finally:
            await self.stop()
            self._current_task = None
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get detailed manager status.
        
        Returns:
            Dictionary with manager status information
        """
        core_status = await self.core.get_status()
        base_stats = self.get_stats()
        
        return {
            **base_stats,
            **core_status,
            "last_result_summary": self._summarize_last_result()
        }
    
    def _summarize_last_result(self) -> Optional[Dict[str, Any]]:
        """Summarize the last execution result."""
        if not self._last_result:
            return None
        
        return {
            "status": self._last_result.get('status'),
            "total_steps": self._last_result.get('total_steps', 0),
            "completed_steps": self._last_result.get('completed_steps', 0),
            "failed_steps": self._last_result.get('failed_steps', 0)
        }
    
    def reset(self):
        """Reset manager state for new goal."""
        self.core.reset()
        self._last_result = None
        print(f"[MANAGER {self.agent_id}] Reset complete")
