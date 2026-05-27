"""Agent Supervisor: Parent-child relationship management."""

from __future__ import annotations
import asyncio
from typing import Dict, List, Optional, Any, Callable
from .kernel import NexusKernel, AgentHandle, AgentState


class AgentSupervisor:
    """
    Manages parent-child relationships between agents.
    Provides fork, wait, kill operations with callback support.
    """
    
    def __init__(self, kernel: NexusKernel):
        self.kernel = kernel
        self.children: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.exit_callbacks: Dict[str, List[Callable]] = {}  # agent_id -> [callbacks]
        self._wait_handles: Dict[str, asyncio.Event] = {}
    
    async def fork(
        self, 
        agent_type: str, 
        task: Optional[str] = None,
        quota: Optional[Dict[str, float]] = None,
        policy: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None
    ) -> str:
        """
        Fork a new agent, optionally execute initial task.
        
        Args:
            agent_type: Type of agent to create
            task: Initial task to execute (optional)
            quota: Resource limits
            policy: Security/sandbox policy
            parent_id: Parent agent ID (defaults to self if managing)
            
        Returns:
            agent_id of the new child
        """
        # Fork via kernel
        child_id = await self.kernel.fork_agent(
            agent_type=agent_type,
            parent_id=parent_id,
            quota=quota,
            policy=policy
        )
        
        # Track parent-child relationship
        if parent_id:
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(child_id)
            
            # Setup wait event
            self._wait_handles[child_id] = asyncio.Event()
        
        # Execute initial task if provided
        if task:
            asyncio.create_task(self.kernel.exec_agent(child_id, task))
        
        return child_id
    
    async def wait(self, agent_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Block until child terminates. Returns exit status.
        
        Args:
            agent_id: ID of child to wait for
            timeout: Maximum wait time
            
        Returns:
            Exit status dictionary
        """
        if agent_id not in self._wait_handles:
            # Not a tracked child, use kernel wait directly
            return await self.kernel.wait_agent(agent_id, timeout)
        
        event = self._wait_handles[agent_id]
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return {"error": "Wait timeout", "agent_id": agent_id}
        
        # Get final status
        return await self.kernel.wait_agent(agent_id, timeout=0)
    
    async def kill(self, agent_id: str, signal: str = "SIGTERM") -> bool:
        """
        Terminate child, run exit callbacks.
        
        Args:
            agent_id: ID of child to kill
            signal: Termination signal type
            
        Returns:
            True if successful
        """
        success = await self.kernel.kill_agent(agent_id)
        
        if success:
            # Trigger exit callbacks
            await self._run_exit_callbacks(agent_id)
            
            # Signal waiters
            if agent_id in self._wait_handles:
                self._wait_handles[agent_id].set()
            
            # Remove from children tracking
            for parent_id, child_list in self.children.items():
                if agent_id in child_list:
                    child_list.remove(agent_id)
        
        return success
    
    def on_exit(self, agent_id: str, callback: Callable[[str, Dict], Any]):
        """
        Register hook for cleanup/post-processing.
        
        Args:
            agent_id: ID of agent to monitor
            callback: Function to call on exit: callback(agent_id, exit_status)
        """
        if agent_id not in self.exit_callbacks:
            self.exit_callbacks[agent_id] = []
        self.exit_callbacks[agent_id].append(callback)
    
    async def _run_exit_callbacks(self, agent_id: str):
        """Run all registered exit callbacks for an agent."""
        if agent_id not in self.exit_callbacks:
            return
        
        # Get exit status
        exit_status = await self.kernel.wait_agent(agent_id, timeout=0)
        
        for callback in self.exit_callbacks[agent_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, exit_status)
                else:
                    callback(agent_id, exit_status)
            except Exception as e:
                print(f"Exit callback error for {agent_id}: {e}")
        
        # Clean up callbacks
        del self.exit_callbacks[agent_id]
    
    def list_children(self, parent_id: str) -> List[str]:
        """List all children of a parent agent."""
        return self.children.get(parent_id, [])
    
    def get_child_count(self, parent_id: str) -> int:
        """Get number of active children for a parent."""
        return len(self.children.get(parent_id, []))
    
    async def kill_all_children(self, parent_id: str) -> int:
        """
        Kill all children of a parent agent.
        
        Returns:
            Number of children killed
        """
        children = self.children.get(parent_id, [])
        kill_count = 0
        
        for child_id in children[:]:  # Copy list to avoid modification during iteration
            if await self.kill(child_id):
                kill_count += 1
        
        return kill_count
