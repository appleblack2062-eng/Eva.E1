"""Agents module for base agent classes and implementations.

This module provides:
- Base agent abstraction
- Worker instances for task execution
- Manager instances for orchestration
- Distributed actor model for multi-agent scaling
"""

from .base_agent import BaseAgent
from .worker_instance import WorkerInstance
from .manager_instance import ManagerInstance
from .distributed_actors import (
    Actor,
    ActorMessage,
    WorkerActor,
    ManagerActor,
    TaskSpec,
    TaskResult,
    MessageType,
    ZMQTransport,
    create_distributed_actor_system,
)

__all__ = [
    "BaseAgent",
    "WorkerInstance",
    "ManagerInstance",
    # Distributed Actors
    "Actor",
    "ActorMessage",
    "WorkerActor",
    "ManagerActor",
    "TaskSpec",
    "TaskResult",
    "MessageType",
    "ZMQTransport",
    "create_distributed_actor_system",
]
