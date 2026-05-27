"""Abstract base agent class for all agent types."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict


class BaseAgent(ABC):
    """
    Abstract base class for all agent implementations.
    
    Provides common interface and shared functionality for:
    - Manager agents (orchestrators)
    - Worker agents (specialists)
    - Future agent types
    """
    
    def __init__(self, agent_id: str, llm_client, config):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            llm_client: LLM client for generating responses
            config: Agent configuration object
        """
        self.agent_id = agent_id
        self.llm = llm_client
        self.config = config
        
        # Runtime state
        self._is_running = False
        self._current_task: Optional[str] = None
        self._task_history: list[Dict[str, Any]] = []
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """
        Execute the agent's primary function.
        
        This method must be implemented by subclasses.
        
        Returns:
            Task execution result
        """
        pass
    
    async def start(self) -> bool:
        """
        Start the agent.
        
        Returns:
            True if started successfully
        """
        if self._is_running:
            return False
        
        self._is_running = True
        print(f"[AGENT {self.agent_id}] Started")
        return True
    
    async def stop(self) -> bool:
        """
        Stop the agent.
        
        Returns:
            True if stopped successfully
        """
        if not self._is_running:
            return False
        
        self._is_running = False
        self._current_task = None
        print(f"[AGENT {self.agent_id}] Stopped")
        return True
    
    @property
    def is_running(self) -> bool:
        """Check if agent is currently running."""
        return self._is_running
    
    @property
    def current_task(self) -> Optional[str]:
        """Get current task description."""
        return self._current_task
    
    def _record_task_execution(
        self, 
        task: str, 
        success: bool, 
        result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record task execution in history.
        
        Args:
            task: Task description
            success: Whether task succeeded
            result: Task result
            metadata: Optional additional metadata
        """
        record = {
            "task": task,
            "success": success,
            "result": result,
            "metadata": metadata or {}
        }
        self._task_history.append(record)
        
        # Keep history bounded
        max_history = getattr(self.config, 'max_task_history', 100)
        if len(self._task_history) > max_history:
            self._task_history = self._task_history[-max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary with agent stats
        """
        successful_tasks = sum(
            1 for record in self._task_history 
            if record.get('success', False)
        )
        
        return {
            "agent_id": self.agent_id,
            "is_running": self._is_running,
            "current_task": self._current_task,
            "total_tasks": len(self._task_history),
            "successful_tasks": successful_tasks,
            "success_rate": (
                successful_tasks / len(self._task_history) 
                if self._task_history else 0.0
            )
        }
