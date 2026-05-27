"""A single-purpose, ephemeral worker agent."""

from typing import Any, Optional
from .base_agent import BaseAgent


class WorkerInstance(BaseAgent):
    """
    Ephemeral worker agent for executing specific tasks.
    
    Workers are:
    - Single-purpose: Created for one specific task
    - Context-optimized: Receive only necessary information
    - Short-lived: Discarded after task completion
    - Specialized: Configured with role-specific prompts
    """
    
    def __init__(
        self, 
        worker_id: str, 
        llm_client, 
        system_prompt: str, 
        config
    ):
        """
        Initialize a worker instance.
        
        Args:
            worker_id: Unique identifier for this worker
            llm_client: LLM client for generating responses
            system_prompt: Optimized system prompt for this worker's task
            config: Agent configuration object
        """
        super().__init__(worker_id, llm_client, config)
        self.system_prompt = system_prompt
        self._result: Optional[str] = None
    
    async def run(self, user_input: str = "") -> Any:
        """
        Execute the single assigned task.
        
        Args:
            user_input: Optional additional input from user
            
        Returns:
            Task execution result
        """
        print(f"[WORKER {self.agent_id}] Starting task...")
        
        if not await self.start():
            raise RuntimeError(f"Worker {self.agent_id} is already running")
        
        try:
            self._current_task = "Executing assigned task"
            
            # Prepare prompt
            prompt = user_input or "Execute the task defined in system prompt."
            
            # Generate response
            response = await self.llm.generate(
                prompt=prompt,
                system_message=self.system_prompt,
                max_tokens=getattr(self.config, 'max_llm_tokens_per_task', 2000),
                temperature=getattr(self.config, 'llm_temperature', 0.7),
            )
            
            # Extract content
            if hasattr(response, 'content'):
                result = response.content
            elif hasattr(response, 'data'):
                result = response.data
            else:
                result = str(response)
            
            self._result = result
            
            # Record success
            self._record_task_execution(
                task=self._current_task,
                success=True,
                result=result[:500] if len(result) > 500 else result
            )
            
            print(f"[WORKER {self.agent_id}] Task complete.")
            return result
            
        except Exception as e:
            # Record failure
            self._record_task_execution(
                task=self._current_task,
                success=False,
                result=None,
                metadata={"error": str(e)}
            )
            
            print(f"[WORKER {self.agent_id}] Task failed: {e}")
            raise
            
        finally:
            await self.stop()
            self._current_task = None
    
    @property
    def result(self) -> Optional[str]:
        """Get the result of the last execution."""
        return self._result
    
    def get_worker_info(self) -> dict:
        """
        Get information about this worker.
        
        Returns:
            Dictionary with worker details
        """
        # Extract role from system prompt
        role = "Unknown"
        if "You are an expert" in self.system_prompt:
            lines = self.system_prompt.split('\n')
            for line in lines:
                if "You are an expert" in line:
                    role = line.replace("You are an expert", "").replace(".", "").strip()
                    break
        
        return {
            "worker_id": self.agent_id,
            "role": role,
            "is_running": self._is_running,
            "has_result": self._result is not None,
            "prompt_length": len(self.system_prompt)
        }
