"""Distributed Actor Model for Multi-Agent Scaling.

This module re-architects the Manager-Worker connection using an asynchronous
Actor framework, enabling true distributed execution across machines with
zero-copy serialization for efficient data sharing.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type
from pathlib import Path

try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None


class MessageType(Enum):
    """Types of messages in the actor system."""
    TASK_SUBMIT = "task_submit"
    TASK_RESULT = "task_result"
    TASK_CANCEL = "task_cancel"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    HEARTBEAT = "heartbeat"
    REGISTER_WORKER = "register_worker"
    DEREGISTER_WORKER = "deregister_worker"
    BROADCAST = "broadcast"


@dataclass
class ActorMessage:
    """Message passed between actors."""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    payload: Any
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None  # For request-response pairing
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self, serializer: str = "json") -> bytes:
        """Serialize message to bytes."""
        if serializer == "protobuf":
            # Would use Protocol Buffers in production
            return self._to_json_bytes()
        elif serializer == "arrow":
            # Would use Apache Arrow for zero-copy in production
            return self._to_json_bytes()
        else:
            return self._to_json_bytes()
    
    def _to_json_bytes(self) -> bytes:
        """Fallback JSON serialization."""
        data = {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes, deserializer: str = "json") -> 'ActorMessage':
        """Deserialize message from bytes."""
        data_dict = json.loads(data.decode('utf-8'))
        return cls(
            message_id=data_dict['message_id'],
            message_type=MessageType(data_dict['message_type']),
            sender_id=data_dict['sender_id'],
            recipient_id=data_dict.get('recipient_id'),
            payload=data_dict['payload'],
            timestamp=data_dict.get('timestamp', time.time()),
            correlation_id=data_dict.get('correlation_id'),
            metadata=data_dict.get('metadata', {})
        )


@dataclass
class TaskSpec:
    """Specification for a task to be executed by a worker."""
    task_id: str
    task_type: str
    input_data: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout_seconds: float = 300.0
    required_capabilities: Set[str] = field(default_factory=set)
    affinity_worker_id: Optional[str] = None  # Prefer specific worker


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    worker_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class Actor(ABC):
    """
    Abstract base class for actors in the distributed system.
    
    Actors are independent units of computation that:
    - Communicate only via message passing
    - Have isolated state
    - Process messages asynchronously
    - Can be distributed across processes or machines
    """
    
    def __init__(self, actor_id: str):
        self.actor_id = actor_id
        self._running = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._handlers: Dict[MessageType, Callable] = {}
        self._setup_handlers()
    
    @abstractmethod
    async def start(self) -> None:
        """Start the actor's message processing loop."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the actor gracefully."""
        pass
    
    @abstractmethod
    async def send_message(self, message: ActorMessage) -> None:
        """Send a message to another actor."""
        pass
    
    def receive_message(self, message: ActorMessage) -> None:
        """Queue a message for processing."""
        self._message_queue.put_nowait(message)
    
    async def process_messages(self) -> None:
        """Main message processing loop."""
        self._running = True
        
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                # Route to appropriate handler
                handler = self._handlers.get(message.message_type)
                if handler:
                    await handler(message)
                else:
                    await self._handle_unknown_message(message)
                
            except asyncio.TimeoutError:
                # Check for shutdown
                continue
            except Exception as e:
                await self._handle_error(e)
        
    def _setup_handlers(self) -> None:
        """Set up message type handlers."""
        self._handlers = {
            MessageType.TASK_SUBMIT: self._handle_task_submit,
            MessageType.TASK_CANCEL: self._handle_task_cancel,
            MessageType.STATUS_REQUEST: self._handle_status_request,
            MessageType.HEARTBEAT: self._handle_heartbeat,
        }
    
    async def _handle_task_submit(self, message: ActorMessage) -> None:
        """Handle incoming task submission."""
        pass
    
    async def _handle_task_cancel(self, message: ActorMessage) -> None:
        """Handle task cancellation request."""
        pass
    
    async def _handle_status_request(self, message: ActorMessage) -> None:
        """Handle status inquiry."""
        pass
    
    async def _handle_heartbeat(self, message: ActorMessage) -> None:
        """Handle heartbeat message."""
        pass
    
    async def _handle_unknown_message(self, message: ActorMessage) -> None:
        """Handle unrecognized message types."""
        print(f"[{self.actor_id}] Unknown message type: {message.message_type}")
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors during message processing."""
        print(f"[{self.actor_id}] Error: {error}")


class WorkerActor(Actor):
    """
    Worker actor that executes tasks.
    
    Workers can be dynamically spawned across machines and register
    themselves with manager actors.
    """
    
    def __init__(
        self,
        actor_id: str,
        capabilities: Set[str] = None,
        max_concurrent_tasks: int = 4
    ):
        super().__init__(actor_id)
        self.capabilities = capabilities or set()
        self.max_concurrent_tasks = max_concurrent_tasks
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._manager_address: Optional[str] = None
        self._last_heartbeat = time.time()
    
    async def start(self) -> None:
        """Start worker actor."""
        print(f"[WORKER {self.actor_id}] Starting...")
        self._running = True
        
        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())
        
        # Start message processing
        await self.process_messages()
    
    async def stop(self) -> None:
        """Stop worker actor gracefully."""
        print(f"[WORKER {self.actor_id}] Stopping...")
        self._running = False
        
        # Cancel active tasks
        for task_id, task in self._active_tasks.items():
            task.cancel()
        
        # Send deregistration to manager
        if self._manager_address:
            await self._send_deregistration()
    
    async def send_message(self, message: ActorMessage) -> None:
        """Send message - implementation depends on transport."""
        # Would be implemented by concrete transport layer
        pass
    
    def set_manager_address(self, address: str) -> None:
        """Set the manager's address for communication."""
        self._manager_address = address
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to manager."""
        while self._running:
            await asyncio.sleep(30)  # 30 second heartbeat interval
            
            if self._manager_address and self._running:
                heartbeat = ActorMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    sender_id=self.actor_id,
                    recipient_id=None,  # Broadcast
                    payload={
                        'active_tasks': len(self._active_tasks),
                        'capabilities': list(self.capabilities),
                        'timestamp': time.time()
                    }
                )
                await self.send_message(heartbeat)
    
    async def _handle_task_submit(self, message: ActorMessage) -> None:
        """Execute submitted task."""
        task_spec: TaskSpec = message.payload
        
        if len(self._active_tasks) >= self.max_concurrent_tasks:
            # Queue is full, reject task
            result = TaskResult(
                task_id=task_spec.task_id,
                success=False,
                error="Worker at capacity"
            )
            await self._send_result(result, message.correlation_id)
            return
        
        # Execute task asynchronously
        async_task = asyncio.create_task(
            self._execute_task(task_spec, message.correlation_id)
        )
        self._active_tasks[task_spec.task_id] = async_task
    
    async def _execute_task(self, task_spec: TaskSpec, correlation_id: str) -> None:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Validate capabilities
            if task_spec.required_capabilities - self.capabilities:
                raise ValueError(
                    f"Missing capabilities: {task_spec.required_capabilities - self.capabilities}"
                )
            
            # Get task handler
            handler = self._get_task_handler(task_spec.task_type)
            if not handler:
                raise ValueError(f"Unknown task type: {task_spec.task_type}")
            
            # Execute task
            output = await handler(task_spec.input_data, task_spec.parameters)
            
            execution_time = (time.time() - start_time) * 1000
            
            result = TaskResult(
                task_id=task_spec.task_id,
                success=True,
                output=output,
                execution_time_ms=execution_time,
                worker_id=self.actor_id,
                metrics={'memory_used_mb': 0}  # Would measure actual usage
            )
            
        except Exception as e:
            result = TaskResult(
                task_id=task_spec.task_id,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                worker_id=self.actor_id
            )
        
        finally:
            # Remove from active tasks
            self._active_tasks.pop(task_spec.task_id, None)
            
            # Send result
            await self._send_result(result, correlation_id)
    
    def _get_task_handler(self, task_type: str) -> Optional[Callable]:
        """Get handler function for task type."""
        # In production, would have registry of task handlers
        # For now, return generic handler
        return self._generic_task_handler
    
    async def _generic_task_handler(
        self,
        input_data: Any,
        parameters: Dict[str, Any]
    ) -> Any:
        """Generic task handler - placeholder."""
        # Simulate some work
        await asyncio.sleep(0.1)
        return input_data
    
    async def _send_result(self, result: TaskResult, correlation_id: str) -> None:
        """Send task result back to manager."""
        if not self._manager_address:
            return
        
        message = ActorMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_RESULT,
            sender_id=self.actor_id,
            recipient_id=None,  # Manager will receive
            payload=result,
            correlation_id=correlation_id
        )
        await self.send_message(message)
    
    async def _send_deregistration(self) -> None:
        """Send deregistration notification to manager."""
        message = ActorMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DEREGISTER_WORKER,
            sender_id=self.actor_id,
            recipient_id=None,
            payload={'reason': 'shutdown'}
        )
        await self.send_message(message)


class ManagerActor(Actor):
    """
    Manager actor that coordinates workers and distributes tasks.
    
    Managers can spawn workers dynamically across machines and maintain
    a registry of available workers with their capabilities.
    """
    
    def __init__(
        self,
        actor_id: str,
        config: Optional[Any] = None
    ):
        super().__init__(actor_id)
        self.config = config
        self._workers: Dict[str, Dict[str, Any]] = {}  # worker_id -> info
        self._pending_tasks: Dict[str, TaskSpec] = {}
        self._task_futures: Dict[str, asyncio.Future] = {}
        self._last_cleanup = time.time()
    
    async def start(self) -> None:
        """Start manager actor."""
        print(f"[MANAGER {self.actor_id}] Starting...")
        self._running = True
        
        # Start cleanup loop
        asyncio.create_task(self._cleanup_loop())
        
        # Start message processing
        await self.process_messages()
    
    async def stop(self) -> None:
        """Stop manager actor gracefully."""
        print(f"[MANAGER {self.actor_id}] Stopping...")
        self._running = False
        
        # Notify all workers
        for worker_id in self._workers:
            await self._notify_worker_shutdown(worker_id)
    
    async def send_message(self, message: ActorMessage) -> None:
        """Send message - implementation depends on transport."""
        pass
    
    async def submit_task(self, task_spec: TaskSpec) -> asyncio.Future:
        """
        Submit a task for execution.
        
        Returns a Future that resolves to TaskResult.
        """
        future = asyncio.Future()
        self._task_futures[task_spec.task_id] = future
        self._pending_tasks[task_spec.task_id] = task_spec
        
        # Try to dispatch immediately
        await self._dispatch_pending_tasks()
        
        return future
    
    async def _handle_register_worker(self, message: ActorMessage) -> None:
        """Register a new worker."""
        worker_id = message.sender_id
        capabilities = message.payload.get('capabilities', [])
        
        self._workers[worker_id] = {
            'capabilities': set(capabilities),
            'registered_at': time.time(),
            'last_heartbeat': time.time(),
            'active_tasks': 0,
            'address': message.payload.get('address')
        }
        
        print(f"[MANAGER {self.actor_id}] Registered worker {worker_id}")
        
        # Try to dispatch pending tasks
        await self._dispatch_pending_tasks()
    
    async def _handle_task_result(self, message: ActorMessage) -> None:
        """Handle task result from worker."""
        result: TaskResult = message.payload
        task_id = result.task_id
        
        # Complete the future
        if task_id in self._task_futures:
            future = self._task_futures[task_id]
            if not future.done():
                future.set_result(result)
            del self._task_futures[task_id]
        
        # Update worker stats
        if result.worker_id in self._workers:
            self._workers[result.worker_id]['active_tasks'] -= 1
        
        # Remove from pending
        self._pending_tasks.pop(task_id, None)
        
        # Try to dispatch more pending tasks
        await self._dispatch_pending_tasks()
    
    async def _handle_heartbeat(self, message: ActorMessage) -> None:
        """Update worker heartbeat."""
        worker_id = message.sender_id
        
        if worker_id in self._workers:
            self._workers[worker_id]['last_heartbeat'] = time.time()
            self._workers[worker_id]['active_tasks'] = message.payload.get('active_tasks', 0)
    
    async def _dispatch_pending_tasks(self) -> None:
        """Dispatch pending tasks to available workers."""
        tasks_to_dispatch = []
        
        for task_id, task_spec in list(self._pending_tasks.items()):
            # Find suitable worker
            worker_id = self._find_suitable_worker(task_spec)
            
            if worker_id:
                tasks_to_dispatch.append((task_spec, worker_id))
        
        # Dispatch tasks
        for task_spec, worker_id in tasks_to_dispatch:
            await self._dispatch_task_to_worker(task_spec, worker_id)
    
    def _find_suitable_worker(self, task_spec: TaskSpec) -> Optional[str]:
        """Find best worker for a task."""
        candidates = []
        
        for worker_id, info in self._workers.items():
            # Check capabilities
            if task_spec.required_capabilities - info['capabilities']:
                continue
            
            # Check capacity (simplified)
            if info['active_tasks'] >= 4:  # Assume max 4 concurrent
                continue
            
            # Calculate score
            score = 0
            
            # Prefer workers with affinity
            if task_spec.affinity_worker_id == worker_id:
                score += 100
            
            # Prefer less loaded workers
            score -= info['active_tasks'] * 10
            
            # Prefer recently active workers
            time_since_heartbeat = time.time() - info['last_heartbeat']
            if time_since_heartbeat < 60:
                score += 20
            
            candidates.append((worker_id, score))
        
        if not candidates:
            return None
        
        # Return highest scored candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    async def _dispatch_task_to_worker(
        self,
        task_spec: TaskSpec,
        worker_id: str
    ) -> None:
        """Dispatch task to specific worker."""
        if worker_id not in self._workers:
            return
        
        message = ActorMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_SUBMIT,
            sender_id=self.actor_id,
            recipient_id=worker_id,
            payload=task_spec,
            correlation_id=task_spec.task_id
        )
        
        await self.send_message(message)
        
        # Update worker stats
        self._workers[worker_id]['active_tasks'] += 1
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up stale workers."""
        while self._running:
            await asyncio.sleep(60)  # Every minute
            
            current_time = time.time()
            stale_workers = []
            
            for worker_id, info in self._workers.items():
                if current_time - info['last_heartbeat'] > 180:  # 3 minutes
                    stale_workers.append(worker_id)
            
            for worker_id in stale_workers:
                print(f"[MANAGER {self.actor_id}] Removing stale worker {worker_id}")
                del self._workers[worker_id]
    
    async def _notify_worker_shutdown(self, worker_id: str) -> None:
        """Notify worker to shut down."""
        message = ActorMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            sender_id=self.actor_id,
            recipient_id=worker_id,
            payload={'command': 'shutdown'}
        )
        await self.send_message(message)


class ZMQTransport:
    """
    ZeroMQ-based transport for actor communication.
    
    Provides high-performance, asynchronous message passing
    suitable for distributed actor systems.
    """
    
    def __init__(self, actor: Actor, bind_address: str = "tcp://*:5555"):
        if not ZMQ_AVAILABLE:
            raise ImportError("pyzmq is required for ZMQTransport")
        
        self.actor = actor
        self.bind_address = bind_address
        self.context = zmq.asyncio.Context()
        self.socket = None
        self._running = False
    
    async def start(self) -> None:
        """Start ZMQ transport."""
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.actor.actor_id)
        self.socket.bind(self.bind_address)
        
        self._running = True
        asyncio.create_task(self._receive_loop())
    
    async def stop(self) -> None:
        """Stop ZMQ transport."""
        self._running = False
        
        if self.socket:
            self.socket.close()
        
        self.context.term()
    
    async def send(self, message: ActorMessage, address: str) -> None:
        """Send message to address."""
        if not self.socket:
            raise RuntimeError("Transport not started")
        
        # Send address envelope followed by message
        await self.socket.send_multipart([
            address.encode('utf-8'),
            message.to_bytes()
        ])
    
    async def _receive_loop(self) -> None:
        """Receive messages from ZMQ socket."""
        while self._running:
            try:
                parts = await self.socket.recv_multipart()
                if len(parts) >= 2:
                    # Second part is the message
                    message = ActorMessage.from_bytes(parts[1])
                    self.actor.receive_message(message)
            except Exception as e:
                if self._running:
                    print(f"ZMQ receive error: {e}")


def create_distributed_actor_system(
    manager_id: str,
    config: Optional[Any] = None,
    use_ray: bool = False,
    use_zmq: bool = True
) -> ManagerActor:
    """
    Factory function to create a distributed actor system.
    
    Args:
        manager_id: ID for the manager actor
        config: System configuration
        use_ray: Whether to use Ray backend (if available)
        use_zmq: Whether to use ZeroMQ transport
        
    Returns:
        Configured ManagerActor
    """
    manager = ManagerActor(manager_id, config)
    
    if use_zmq and ZMQ_AVAILABLE:
        transport = ZMQTransport(manager)
        # Wire up transport
        original_send = manager.send_message
        
        async def send_with_transport(message: ActorMessage) -> None:
            if message.recipient_id and transport.socket:
                await transport.send(message, message.recipient_id)
            else:
                await original_send(message)
        
        manager.send_message = send_with_transport
    
    return manager
