"""Safety guards for agent execution."""

from __future__ import annotations
import time
import resource
from contextlib import contextmanager

class TimeoutGuard:
    """Raises TimeoutError if block takes too long."""
    
    def __init__(self, seconds: float):
        self.seconds = seconds
    
    def __enter__(self):
        self.start = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if time.time() - self.start > self.seconds:
            raise TimeoutError(f"Operation exceeded {self.seconds}s limit")
        return False

class ResourceMonitor:
    """Monitors memory and CPU usage."""
    
    def __init__(self, config):
        self.max_memory_mb = config.max_memory_usage_mb
    
    def check_memory(self):
        """Check current process memory usage."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        mb_used = usage.ru_maxrss / 1024  # Convert KB to MB
        if mb_used > self.max_memory_mb:
            raise MemoryError(f"Memory limit exceeded: {mb_used:.1f}MB > {self.max_memory_mb}MB")
