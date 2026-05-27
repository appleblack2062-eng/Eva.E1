"""Hot Tier Memory: LRU cache for immediate working context."""

from __future__ import annotations
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MemoryItem:
    """A single memory item in the hot tier."""
    key: str
    content: str
    metadata: Dict[str, Any]
    created_at: float
    accessed_at: float
    token_count: int = 0


class HotTierMemory:
    """
    LRU cache of immediate working context (~3000 tokens).
    Provides fast access to frequently used information.
    """
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.cache: OrderedDict[str, MemoryItem] = OrderedDict()
        self.current_tokens = 0
        self._lock = None  # Async lock would be used in async context
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def put(self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add item to cache. Evicts oldest if token limit exceeded.
        
        Args:
            key: Unique identifier for the memory
            content: The memory content
            metadata: Optional metadata (tags, importance, etc.)
        """
        # Estimate token count (simple word-based estimation)
        token_count = len(content.split()) * 1.3  # Rough estimate
        
        # If key exists, remove old version first
        if key in self.cache:
            old_item = self.cache.pop(key)
            self.current_tokens -= old_item.token_count
        
        # Create new item
        now = time.time()
        item = MemoryItem(
            key=key,
            content=content,
            metadata=metadata or {},
            created_at=now,
            accessed_at=now,
            token_count=int(token_count)
        )
        
        # Evict if necessary
        while self.current_tokens + token_count > self.max_tokens and self.cache:
            self._evict_oldest()
        
        # Add to cache (end = most recently used)
        self.cache[key] = item
        self.current_tokens += int(token_count)
    
    def get(self, key: str) -> Optional[str]:
        """
        Retrieve item and move to end (MRU).
        
        Args:
            key: Identifier of memory to retrieve
            
        Returns:
            Content string or None if not found
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        item = self.cache.pop(key)  # Remove from current position
        item.accessed_at = time.time()
        self.cache[key] = item  # Re-add at end (most recently used)
        
        self.hits += 1
        return item.content
    
    def peek(self, key: str) -> Optional[MemoryItem]:
        """Get item without updating access time."""
        return self.cache.get(key)
    
    def peek_all(self) -> List[MemoryItem]:
        """
        Return all items for prompt injection.
        Items are ordered from least to most recently used.
        """
        return list(self.cache.values())
    
    def _evict_oldest(self):
        """Remove the least recently used item."""
        if not self.cache:
            return
        
        # First item is oldest (LRU)
        key, item = self.cache.popitem(last=False)
        self.current_tokens -= item.token_count
        self.evictions += 1
    
    def delete(self, key: str) -> bool:
        """Remove a specific item from cache."""
        if key in self.cache:
            item = self.cache.pop(key)
            self.current_tokens -= item.token_count
            return True
        return False
    
    def clear(self):
        """Clear all items from cache."""
        self.cache.clear()
        self.current_tokens = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.current_tokens / self.max_tokens,
            "item_count": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate
        }
    
    def search(self, query: str) -> List[MemoryItem]:
        """
        Simple text search within cache contents.
        
        Args:
            query: Search string
            
        Returns:
            List of matching items
        """
        query_lower = query.lower()
        matches = []
        
        for item in self.cache.values():
            if query_lower in item.content.lower():
                matches.append(item)
            elif any(query_lower in str(v).lower() for v in item.metadata.values()):
                matches.append(item)
        
        return matches
    
    def get_recent(self, count: int = 5) -> List[MemoryItem]:
        """Get the most recently accessed items."""
        items = list(self.cache.values())
        return items[-count:] if len(items) > count else items
    
    def get_by_metadata(self, key_filter: str, value_filter: Any = None) -> List[MemoryItem]:
        """
        Filter items by metadata.
        
        Args:
            key_filter: Metadata key to filter on
            value_filter: Optional value to match (if None, just check key exists)
            
        Returns:
            List of matching items
        """
        matches = []
        for item in self.cache.values():
            if key_filter in item.metadata:
                if value_filter is None or item.metadata[key_filter] == value_filter:
                    matches.append(item)
        return matches
