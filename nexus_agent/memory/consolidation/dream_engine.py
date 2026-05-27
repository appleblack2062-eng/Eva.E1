"""Dream Engine: Background memory consolidation during 'sleep' cycles."""

from __future__ import annotations
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class DreamStats:
    """Statistics from a dream cycle."""
    cycle_id: str
    start_time: float
    end_time: float
    memories_processed: int
    clusters_formed: int
    summaries_created: int
    nodes_pruned: int
    nodes_archived: int


class DreamEngine:
    """
    Background 'sleep' cycle to organize and consolidate memory.
    Mimics biological sleep patterns for memory optimization.
    
    Phases:
    1. Orient: Get current memory stats
    2. Gather: Query high-signal recent memories
    3. Consolidate: Cluster by entity/topic, summarize
    4. Prune: Delete/archive low-importance stale nodes
    """
    
    def __init__(
        self, 
        config, 
        vector_store=None, 
        llm_client=None,
        hot_memory=None,
        warm_memory=None,
        cold_memory=None
    ):
        self.config = config
        self.vector_store = vector_store
        self.llm = llm_client
        self.hot_memory = hot_memory
        self.warm_memory = warm_memory
        self.cold_memory = cold_memory
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_interval = config.get('dream_cycle_interval', 3600)  # Default 1 hour
        self._last_cycle: Optional[DreamStats] = None
        self._cycle_history: List[DreamStats] = []
        
        # Consolidation settings
        self.max_clusters = config.get('max_dream_clusters', 10)
        self.similarity_threshold = config.get('dream_similarity_threshold', 0.75)
        self.prune_age_threshold = config.get('dream_prune_age_hours', 24)
    
    def start(self):
        """Start the background dream cycle loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
    
    def stop(self):
        """Stop the dream engine."""
        self._running = False
        if self._task:
            self._task.cancel()
    
    async def _run_loop(self):
        """Main async loop for dream cycles."""
        while self._running:
            try:
                await self._run_dream_cycle()
            except Exception as e:
                print(f"Dream cycle error: {e}")
            
            # Wait until next cycle
            await asyncio.sleep(self._cycle_interval)
    
    async def _run_dream_cycle(self) -> DreamStats:
        """
        Execute one complete dream cycle.
        
        Returns:
            DreamStats with cycle results
        """
        cycle_id = f"dream_{int(time.time())}"
        start_time = time.time()
        
        stats = DreamStats(
            cycle_id=cycle_id,
            start_time=start_time,
            end_time=0,
            memories_processed=0,
            clusters_formed=0,
            summaries_created=0,
            nodes_pruned=0,
            nodes_archived=0
        )
        
        # Phase 1: Orient - Get stats
        orient_data = await self._orient()
        stats.memories_processed = orient_data.get('total_memories', 0)
        
        # Phase 2: Gather - Collect high-signal memories
        gathered = await self._gather()
        
        # Phase 3: Consolidate - Cluster and summarize
        if gathered and self.llm:
            clusters = await self._consolidate(gathered)
            stats.clusters_formed = len(clusters)
            
            # Create summary nodes
            summaries = await self._create_summaries(clusters)
            stats.summaries_created = len(summaries)
        
        # Phase 4: Prune - Remove stale/low-value nodes
        prune_result = await self._prune()
        stats.nodes_pruned = prune_result.get('pruned', 0)
        stats.nodes_archived = prune_result.get('archived', 0)
        
        # Finalize
        stats.end_time = time.time()
        self._last_cycle = stats
        self._cycle_history.append(stats)
        
        # Keep only last 100 cycles in history
        if len(self._cycle_history) > 100:
            self._cycle_history = self._cycle_history[-100:]
        
        return stats
    
    async def _orient(self) -> Dict[str, Any]:
        """Get current memory statistics and state."""
        result = {
            'total_memories': 0,
            'hot_count': 0,
            'warm_count': 0,
            'cold_count': 0
        }
        
        if self.hot_memory:
            hot_stats = self.hot_memory.get_stats()
            result['hot_count'] = hot_stats.get('item_count', 0)
            result['total_memories'] += result['hot_count']
        
        # Warm and warm tier counts would come from their respective stores
        # This is a simplified implementation
        
        return result
    
    async def _gather(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Gather high-signal recent memories for consolidation.
        
        Args:
            limit: Maximum number of memories to gather
            
        Returns:
            List of memory items with metadata
        """
        gathered = []
        
        # Get recent items from hot memory
        if self.hot_memory:
            recent = self.hot_memory.get_recent(count=limit)
            for item in recent:
                gathered.append({
                    'id': item.key,
                    'content': item.content,
                    'metadata': item.metadata,
                    'accessed_at': item.accessed_at,
                    'created_at': item.created_at,
                    'importance': self._calculate_importance(item)
                })
        
        # Could also gather from warm tier based on access patterns
        if self.warm_memory and len(gathered) < limit:
            # Search for frequently accessed items
            pass
        
        return gathered
    
    def _calculate_importance(self, item) -> float:
        """Calculate importance score for a memory item."""
        # Simple heuristic: recent + frequently accessed = important
        now = time.time()
        age_hours = (now - item.created_at) / 3600
        recency_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
        
        # Metadata-based importance
        importance_meta = item.metadata.get('importance', 0.5)
        
        return (recency_score + importance_meta) / 2
    
    async def _consolidate(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster memories by entity/topic.
        
        Args:
            memories: List of memory items
            
        Returns:
            List of clusters (each cluster is a list of memories)
        """
        if not memories:
            return []
        
        # Simple clustering based on keyword overlap
        # In production, would use embedding similarity
        clusters = []
        used = set()
        
        for i, memory in enumerate(memories):
            if i in used:
                continue
            
            # Start new cluster
            cluster = [memory]
            used.add(i)
            
            # Find similar memories
            for j, other in enumerate(memories):
                if j in used or j == i:
                    continue
                
                if self._are_similar(memory, other):
                    cluster.append(other)
                    used.add(j)
                
                if len(cluster) >= 10:  # Max cluster size
                    break
            
            clusters.append(cluster)
        
        # Limit number of clusters
        return clusters[:self.max_clusters]
    
    def _are_similar(self, mem1: Dict, mem2: Dict) -> bool:
        """Check if two memories are similar enough to cluster."""
        # Simple keyword-based similarity
        words1 = set(mem1['content'].lower().split())
        words2 = set(mem2['content'].lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return False
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= self.similarity_threshold
    
    async def _create_summaries(self, clusters: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Create summary nodes for each cluster using LLM.
        
        Args:
            clusters: List of memory clusters
            
        Returns:
            List of created summary nodes
        """
        summaries = []
        
        for i, cluster in enumerate(clusters):
            if not self.llm:
                continue
            
            # Prepare content for summarization
            contents = [m['content'] for m in cluster]
            combined = "\n\n".join(contents[:5])  # Limit context
            
            prompt = f"""Summarize the following related memories into a concise summary:

{combined}

Provide a 2-3 sentence summary that captures the key information."""
            
            try:
                response = await self.llm.generate(prompt, max_tokens=200)
                summary_content = response.content if hasattr(response, 'content') else str(response)
                
                # Store summary in hot memory
                summary_id = f"summary_{i}_{int(time.time())}"
                if self.hot_memory:
                    self.hot_memory.put(
                        summary_id,
                        summary_content,
                        metadata={
                            'type': 'summary',
                            'source_cluster_size': len(cluster),
                            'created_by': 'dream_engine'
                        }
                    )
                
                summaries.append({
                    'id': summary_id,
                    'content': summary_content,
                    'source_count': len(cluster)
                })
            except Exception as e:
                print(f"Error creating summary: {e}")
        
        return summaries
    
    async def _prune(self) -> Dict[str, int]:
        """
        Remove or archive low-importance stale nodes.
        
        Returns:
            Dictionary with pruned/archived counts
        """
        result = {'pruned': 0, 'archived': 0}
        
        if not self.hot_memory:
            return result
        
        now = time.time()
        age_threshold_seconds = self.prune_age_threshold * 3600
        
        # Find old, low-importance items
        to_remove = []
        to_archive = []
        
        for item in self.hot_memory.peek_all():
            age = now - item.created_at
            importance = item.metadata.get('importance', 0.5)
            
            if age > age_threshold_seconds:
                if importance < 0.3:
                    to_remove.append(item.key)
                elif importance < 0.6:
                    to_archive.append(item.key)
        
        # Archive first
        if to_archive and self.cold_memory:
            for key in to_archive:
                content = self.hot_memory.get(key)
                if content:
                    item = self.hot_memory.peek(key)
                    await self.cold_memory.archive(
                        key,
                        content,
                        item.metadata if item else {}
                    )
                    result['archived'] += 1
        
        # Then remove
        for key in to_remove:
            if self.hot_memory.delete(key):
                result['pruned'] += 1
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dream engine statistics."""
        return {
            'running': self._running,
            'cycle_interval': self._cycle_interval,
            'last_cycle': self._last_cycle.__dict__ if self._last_cycle else None,
            'total_cycles': len(self._cycle_history),
            'avg_processing_time': (
                sum(c.end_time - c.start_time for c in self._cycle_history) / len(self._cycle_history)
                if self._cycle_history else 0
            )
        }
    
    async def run_manual_cycle(self) -> DreamStats:
        """Run a single dream cycle manually."""
        return await self._run_dream_cycle()
