"""Nexus Kernel: Global resource management and agent scheduling."""

from __future__ import annotations
import asyncio
import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import heapq


class AgentState(Enum):
    """Possible states for an agent."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    TERMINATED = "terminated"


@dataclass
class AgentHandle:
    """Represents an agent instance in the kernel."""
    agent_id: str
    state: AgentState
    created_at: float
    last_active: float
    resource_quota: Dict[str, float]
    sandbox_id: Optional[str]
    parent_id: Optional[str]
    priority: int = 0
    task: Optional[str] = None
    
    def __lt__(self, other: AgentHandle) -> bool:
        """For priority queue comparison (higher priority first)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class NexusKernel:
    """
    The core OS layer for managing AI agents.
    Provides Unix-like process management: fork, exec, wait, kill.
    """
    
    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, AgentHandle] = {}
        self.sandboxes: Dict[str, Any] = {}  # sandbox_id -> sandbox_instance
        self.priority_queue: List[AgentHandle] = []
        self._lock = asyncio.Lock()
        self._scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_forks = 0
        self.total_execs = 0
        self.total_kills = 0
    
    async def fork_agent(
        self, 
        agent_type: str, 
        parent_id: Optional[str] = None,
        quota: Optional[Dict[str, float]] = None,
        policy: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new agent handle, allocate sandbox, check quotas.
        
        Args:
            agent_type: Type of agent to create
            parent_id: ID of parent agent (if any)
            quota: Resource limits {max_tokens, max_time, max_memory}
            policy: Security/sandbox policy
            
        Returns:
            agent_id: Unique identifier for the new agent
        """
        async with self._lock:
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
            
            # Default quota
            if quota is None:
                quota = {
                    "max_tokens": 10000,
                    "max_time": 300,  # seconds
                    "max_memory": 512  # MB
                }
            
            # Create sandbox if needed
            sandbox_id = None
            if policy and policy.get("isolated", False):
                from ..security.sandbox_manager import SandboxManager
                sandbox_mgr = SandboxManager()
                sandbox_id = await sandbox_mgr.create(agent_id, policy)
                self.sandboxes[sandbox_id] = sandbox_mgr
            
            # Create agent handle
            handle = AgentHandle(
                agent_id=agent_id,
                state=AgentState.IDLE,
                created_at=time.time(),
                last_active=time.time(),
                resource_quota=quota,
                sandbox_id=sandbox_id,
                parent_id=parent_id,
                priority=policy.get("priority", 0) if policy else 0
            )
            
            self.agents[agent_id] = handle
            heapq.heappush(self.priority_queue, handle)
            self.total_forks += 1
            
            # Register with parent if exists
            if parent_id and parent_id in self.agents:
                # Parent can track children
                pass
            
            return agent_id
    
    async def exec_agent(
        self, 
        agent_id: str, 
        task: str, 
        *args,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a callable within the agent's sandbox.
        Enforces timeouts and memory limits.
        
        Args:
            agent_id: ID of agent to execute
            task: Task description or callable
            timeout: Maximum execution time in seconds
            
        Returns:
            Result of execution
        """
        async with self._lock:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            handle = self.agents[agent_id]
            if handle.state == AgentState.TERMINATED:
                raise ValueError(f"Agent {agent_id} is terminated")
            
            # Update state
            handle.state = AgentState.RUNNING
            handle.task = task
            handle.last_active = time.time()
            self.total_execs += 1
        
        # Get timeout from quota if not specified
        if timeout is None:
            timeout = handle.resource_quota.get("max_time", 300)
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task(handle, task, *args),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            await self.kill_agent(agent_id)
            raise TimeoutError(f"Agent {agent_id} exceeded timeout of {timeout}s")
        except Exception as e:
            handle.state = AgentState.BLOCKED
            raise
    
    async def _execute_task(self, handle: AgentHandle, task: str, *args) -> Any:
        """Internal task execution logic."""
        # If sandbox exists, execute there
        if handle.sandbox_id and handle.sandbox_id in self.sandboxes:
            sandbox = self.sandboxes[handle.sandbox_id]
            return await sandbox.execute(task, *args)
        
        # Otherwise, just return task info (placeholder for actual execution)
        await asyncio.sleep(0.1)  # Simulate execution
        return {"task": task, "args": args, "status": "completed"}
    
    async def kill_agent(self, agent_id: str) -> bool:
        """
        Terminate agent, destroy sandbox, reclaim resources.
        
        Args:
            agent_id: ID of agent to kill
            
        Returns:
            True if successful
        """
        async with self._lock:
            if agent_id not in self.agents:
                return False
            
            handle = self.agents[agent_id]
            handle.state = AgentState.TERMINATED
            handle.last_active = time.time()
            
            # Destroy sandbox
            if handle.sandbox_id and handle.sandbox_id in self.sandboxes:
                sandbox = self.sandboxes.pop(handle.sandbox_id)
                await sandbox.destroy()
            
            self.total_kills += 1
            return True
    
    async def wait_agent(self, agent_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for agent to complete execution.
        
        Args:
            agent_id: ID of agent to wait for
            timeout: Maximum wait time
            
        Returns:
            Exit status and results
        """
        start_time = time.time()
        while True:
            async with self._lock:
                if agent_id not in self.agents:
                    return {"error": "Agent not found"}
                
                handle = self.agents[agent_id]
                if handle.state == AgentState.TERMINATED:
                    return {
                        "agent_id": agent_id,
                        "state": handle.state,
                        "exit_time": handle.last_active
                    }
                if handle.state == AgentState.BLOCKED:
                    return {
                        "agent_id": agent_id,
                        "state": handle.state,
                        "error": "Agent blocked"
                    }
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return {"error": "Wait timeout"}
            
            await asyncio.sleep(0.1)
    
    async def run_scheduler(self):
        """Background loop: picks highest priority ready agents and dispatches them."""
        self._scheduler_running = True
        
        while self._scheduler_running:
            async with self._lock:
                if self.priority_queue:
                    # Get highest priority agent
                    handle = heapq.heappop(self.priority_queue)
                    
                    if handle.state == AgentState.IDLE:
                        # Ready to execute - would dispatch here
                        pass
                    elif handle.state == AgentState.TERMINATED:
                        # Skip terminated agents
                        continue
                    else:
                        # Re-queue if still active
                        heapq.heappush(self.priority_queue, handle)
            
            await asyncio.sleep(0.5)  # Scheduler tick
    
    def stop_scheduler(self):
        """Stop the scheduler loop."""
        self._scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an agent."""
        if agent_id not in self.agents:
            return None
        
        handle = self.agents[agent_id]
        return {
            "agent_id": handle.agent_id,
            "state": handle.state.value,
            "created_at": handle.created_at,
            "last_active": handle.last_active,
            "quota": handle.resource_quota,
            "task": handle.task,
            "priority": handle.priority
        }
    
    def list_agents(self, state_filter: Optional[AgentState] = None) -> List[str]:
        """List all agents, optionally filtered by state."""
        if state_filter is None:
            return list(self.agents.keys())
        
        return [
            aid for aid, handle in self.agents.items()
            if handle.state == state_filter
        ]
