"""High-speed LRU cache for execution results and compiled code."""

from __future__ import annotations
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple

class ExecutionCacheStore:
    """In-memory cache with TTL and size limits."""
    
    def __init__(self, config):
        self.max_size = config.cache_size
        self.default_ttl = config.cache_ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float, float]] = OrderedDict()
        # Key -> (Value, CreatedAt, TTL)
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        value, created_at, ttl = self._cache[key]
        if time.time() - created_at > ttl:
            del self._cache[key]
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        if key in self._cache:
            self._cache.move_to_end(key)
        
        self._cache[key] = (value, time.time(), ttl or self.default_ttl)
        
        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def invalidate(self, key: str):
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        self._cache.clear()
