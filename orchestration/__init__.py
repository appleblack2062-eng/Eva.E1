"""Orchestration module for multi-agent coordination and task management."""

# Lazy imports to avoid circular dependency issues
__all__ = [
    "TaskDecomposer",
    "TaskStep",
    "ManagerAgent",
    "WorkerFactory",
]

def __getattr__(name):
    if name in ("TaskDecomposer", "TaskStep"):
        from .task_decomposer import TaskDecomposer, TaskStep
        return TaskDecomposer if name == "TaskDecomposer" else TaskStep
    elif name == "ManagerAgent":
        from .manager_agent import ManagerAgent
        return ManagerAgent
    elif name == "WorkerFactory":
        from .worker_factory import WorkerFactory
        return WorkerFactory
    raise AttributeError(f"module {__name__!r} has no attribute {__name__!r}")
