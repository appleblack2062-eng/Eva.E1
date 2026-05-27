"""A2A Protocol: Agent-to-Agent communication and discovery."""

from __future__ import annotations
import asyncio
import uuid
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(Enum):
    """Types of A2A messages."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    DISCOVERY = "discovery"
    REGISTER = "register"


@dataclass
class A2AMessage:
    """A message in the A2A protocol."""
    id: str
    type: MessageType
    sender: str
    target: Optional[str]
    capability: Optional[str]
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'sender': self.sender,
            'target': self.target,
            'capability': self.capability,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AMessage':
        return cls(
            id=data['id'],
            type=MessageType(data['type']),
            sender=data['sender'],
            target=data.get('target'),
            capability=data.get('capability'),
            payload=data.get('payload', {}),
            timestamp=data['timestamp'],
            correlation_id=data.get('correlation_id')
        )


class A2AProtocol:
    """
    Agent-to-Agent communication protocol.
    
    Features:
    - Capability registration and discovery
    - Async request/response messaging
    - Message routing
    - Correlation tracking
    """
    
    def __init__(self, agent_id: str, config=None):
        self.agent_id = agent_id
        self.config = config or {}
        
        # Registered capabilities: name -> schema
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Message handlers: capability_name -> handler_function
        self.handlers: Dict[str, Callable] = {}
        
        # Pending requests: correlation_id -> asyncio.Future
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Known agents: agent_id -> capabilities
        self.known_agents: Dict[str, List[str]] = {}
        
        # Message queue for incoming messages
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Running state
        self._running = False
        self._message_loop_task: Optional[asyncio.Task] = None
    
    def register_capability(self, name: str, schema: Dict[str, Any], handler: Optional[Callable] = None):
        """
        Advertise a capability that this agent provides.
        
        Args:
            name: Capability name
            schema: JSON schema describing inputs/outputs
            handler: Optional async function to handle requests
        """
        self.capabilities[name] = schema
        
        if handler:
            self.handlers[name] = handler
    
    def unregister_capability(self, name: str):
        """Remove a registered capability."""
        self.capabilities.pop(name, None)
        self.handlers.pop(name, None)
    
    async def send_request(
        self, 
        target: str, 
        capability: str, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Send an async request to another agent.
        
        Args:
            target: Target agent ID
            capability: Capability to invoke
            payload: Request payload
            timeout: Request timeout in seconds
            
        Returns:
            Response payload
        """
        correlation_id = str(uuid.uuid4())
        
        # Create request message
        message = A2AMessage(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            sender=self.agent_id,
            target=target,
            capability=capability,
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id
        )
        
        # Create future for response
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_requests[correlation_id] = future
        
        try:
            # Send message (in real implementation, would use network)
            await self._send_message(message)
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {target} timed out after {timeout}s")
        finally:
            # Clean up pending request
            self.pending_requests.pop(correlation_id, None)
    
    async def send_response(
        self, 
        original_message: A2AMessage, 
        payload: Dict[str, Any]
    ):
        """Send a response to a request."""
        response = A2AMessage(
            id=str(uuid.uuid4()),
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            target=original_message.sender,
            capability=original_message.capability,
            payload=payload,
            timestamp=time.time(),
            correlation_id=original_message.id
        )
        
        await self._send_message(response)
    
    async def send_error(
        self, 
        original_message: A2AMessage, 
        error: str
    ):
        """Send an error response."""
        error_msg = A2AMessage(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR,
            sender=self.agent_id,
            target=original_message.sender,
            capability=original_message.capability,
            payload={'error': error},
            timestamp=time.time(),
            correlation_id=original_message.id
        )
        
        await self._send_message(error_msg)
    
    async def _send_message(self, message: A2AMessage):
        """
        Send a message to its destination.
        
        In a real implementation, this would use:
        - HTTP/gRPC for remote agents
        - Shared memory/queue for local agents
        """
        # For local simulation, put in recipient's queue if known
        if message.target and message.target in self.known_agents:
            # Would route to actual agent
            pass
        
        # Broadcast for discovery messages
        if message.type == MessageType.DISCOVERY:
            await self._handle_discovery(message)
    
    async def _handle_discovery(self, message: A2AMessage):
        """Handle a discovery broadcast."""
        # Register the discovering agent
        sender = message.sender
        caps = message.payload.get('capabilities', [])
        self.known_agents[sender] = caps
    
    async def handle_incoming_request(self, message: A2AMessage):
        """
        Handle an incoming request message.
        
        Args:
            message: The incoming request message
        """
        capability = message.capability
        
        # Check if we have this capability
        if capability not in self.handlers:
            await self.send_error(message, f"Capability '{capability}' not found")
            return
        
        try:
            # Call handler
            handler = self.handlers[capability]
            result = handler(message.payload)
            
            # If handler is async, await it
            if asyncio.iscoroutine(result):
                result = await result
            
            # Send response
            await self.send_response(message, {'result': result})
            
        except Exception as e:
            await self.send_error(message, str(e))
    
    async def handle_incoming_response(self, message: A2AMessage):
        """Handle an incoming response message."""
        correlation_id = message.correlation_id
        
        if correlation_id in self.pending_requests:
            future = self.pending_requests[correlation_id]
            
            if message.type == MessageType.ERROR:
                future.set_exception(Exception(message.payload.get('error', 'Unknown error')))
            else:
                future.set_result(message.payload)
    
    async def run_message_loop(self):
        """Background loop to process incoming messages."""
        self._running = True
        
        while self._running:
            try:
                message_data = await self.message_queue.get()
                message = A2AMessage.from_dict(message_data)
                
                if message.type == MessageType.REQUEST:
                    await self.handle_incoming_request(message)
                elif message.type in [MessageType.RESPONSE, MessageType.ERROR]:
                    await self.handle_incoming_response(message)
                elif message.type == MessageType.DISCOVERY:
                    await self._handle_discovery(message)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Message loop error: {e}")
    
    def start(self):
        """Start the A2A protocol message loop."""
        if not self._running:
            self._message_loop_task = asyncio.create_task(self.run_message_loop())
    
    def stop(self):
        """Stop the A2A protocol."""
        self._running = False
        if self._message_loop_task:
            self._message_loop_task.cancel()
    
    async def receive_message(self, message_dict: Dict[str, Any]):
        """Receive a message from the network."""
        await self.message_queue.put(message_dict)
    
    def discover_agents(self) -> List[Dict[str, Any]]:
        """Get list of known agents and their capabilities."""
        return [
            {'agent_id': aid, 'capabilities': caps}
            for aid, caps in self.known_agents.items()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get A2A protocol statistics."""
        return {
            'agent_id': self.agent_id,
            'registered_capabilities': len(self.capabilities),
            'known_agents': len(self.known_agents),
            'pending_requests': len(self.pending_requests),
            'queue_size': self.message_queue.qsize(),
            'running': self._running
        }
