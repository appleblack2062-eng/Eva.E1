"""Merge Agent: Self-organizing memory through node merging."""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class MergeResult:
    """Result of a merge operation."""
    merged_count: int
    new_summary_nodes: int
    archived_nodes: int
    similarity_threshold_used: float


class MergeAgent:
    """
    Merges similar nodes to reduce graph complexity.
    Automatically identifies and consolidates duplicate or highly similar memories.
    """
    
    def __init__(
        self,
        hot_memory=None,
        warm_memory=None,
        llm_client=None,
        similarity_threshold: float = 0.88
    ):
        self.hot_memory = hot_memory
        self.warm_memory = warm_memory
        self.llm = llm_client
        self.similarity_threshold = similarity_threshold
        
        # Statistics
        self.total_merges = 0
        self.total_nodes_processed = 0
    
    async def run_merge_cycle(self, batch_size: int = 100) -> MergeResult:
        """
        Execute one merge cycle.
        
        Args:
            batch_size: Number of recent nodes to process
            
        Returns:
            MergeResult with statistics
        """
        result = MergeResult(
            merged_count=0,
            new_summary_nodes=0,
            archived_nodes=0,
            similarity_threshold_used=self.similarity_threshold
        )
        
        # Get recent nodes to analyze
        nodes = await self._get_recent_nodes(batch_size)
        self.total_nodes_processed += len(nodes)
        
        if len(nodes) < 2:
            return result
        
        # Compute pairwise similarities
        pairs_to_merge = await self._find_similar_pairs(nodes)
        
        # Execute merges
        for pair in pairs_to_merge:
            merge_success = await self._merge_pair(pair)
            if merge_success:
                result.merged_count += 1
                result.new_summary_nodes += 1
        
        return result
    
    async def _get_recent_nodes(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent nodes from memory tiers."""
        nodes = []
        
        # From hot memory
        if self.hot_memory:
            recent = self.hot_memory.get_recent(count=limit)
            for item in recent:
                nodes.append({
                    'id': item.key,
                    'content': item.content,
                    'metadata': item.metadata,
                    'source': 'hot'
                })
        
        # From warm memory
        if self.warm_memory and len(nodes) < limit:
            # Search for recent items
            warm_results = await self.warm_memory.search(
                query="*",  # Get all
                limit=limit - len(nodes)
            )
            for item in warm_results:
                nodes.append({
                    'id': item.get('id'),
                    'content': item.get('content', ''),
                    'metadata': item.get('metadata', {}),
                    'source': 'warm'
                })
        
        return nodes
    
    async def _find_similar_pairs(
        self, 
        nodes: List[Dict[str, Any]]
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Find pairs of nodes that should be merged.
        
        Returns:
            List of tuples: (node1, node2, similarity_score)
        """
        pairs = []
        used = set()
        
        for i, node1 in enumerate(nodes):
            if i in used:
                continue
            
            for j, node2 in enumerate(nodes):
                if j <= i or j in used:
                    continue
                
                similarity = self._compute_similarity(node1, node2)
                
                if similarity >= self.similarity_threshold:
                    pairs.append((node1, node2, similarity))
                    used.add(i)
                    used.add(j)
                    break  # Only merge each node once per cycle
        
        return pairs
    
    def _compute_similarity(self, node1: Dict, node2: Dict) -> float:
        """
        Compute similarity between two nodes.
        
        Uses cosine similarity on embeddings if available,
        otherwise falls back to text-based Jaccard similarity.
        """
        # If embeddings exist in metadata, use cosine similarity
        emb1 = node1.get('metadata', {}).get('embedding')
        emb2 = node2.get('metadata', {}).get('embedding')
        
        if emb1 and emb2:
            return self._cosine_similarity(emb1, emb2)
        
        # Fallback to text-based similarity
        return self._text_similarity(node1['content'], node2['content'])
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall'
        }
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _merge_pair(self, pair: Tuple[Dict, Dict, float]) -> bool:
        """
        Merge two similar nodes into a summary node.
        
        Args:
            pair: Tuple of (node1, node2, similarity)
            
        Returns:
            True if merge was successful
        """
        node1, node2, similarity = pair
        
        if not self.llm:
            # Without LLM, just archive one and keep the other
            return await self._simple_merge(node1, node2)
        
        # Use LLM to create merged summary
        prompt = f"""Merge the following two similar pieces of information into a single comprehensive summary:

Information 1:
{node1['content']}

Information 2:
{node2['content']}

Create a unified summary that preserves all important information without redundancy."""
        
        try:
            response = await self.llm.generate(prompt, max_tokens=300)
            merged_content = response.content if hasattr(response, 'content') else str(response)
            
            # Create summary node ID
            summary_id = f"merged_{node1['id']}_{node2['id']}"
            
            # Store in hot memory
            if self.hot_memory:
                self.hot_memory.put(
                    summary_id,
                    merged_content,
                    metadata={
                        'type': 'merged',
                        'source_nodes': [node1['id'], node2['id']],
                        'similarity': similarity,
                        'merged_at': time.time()
                    }
                )
            
            # Archive original nodes
            if self.warm_memory:
                # Move to warm tier (or cold if available)
                pass
            
            # Remove from hot memory
            if self.hot_memory:
                self.hot_memory.delete(node1['id'])
                self.hot_memory.delete(node2['id'])
            
            self.total_merges += 1
            return True
            
        except Exception as e:
            print(f"Error merging nodes: {e}")
            return False
    
    async def _simple_merge(self, node1: Dict, node2: Dict) -> bool:
        """
        Simple merge without LLM: keep newer, mark older as archived.
        
        Returns:
            True if successful
        """
        # Determine which is newer
        time1 = node1.get('metadata', {}).get('created_at', 0)
        time2 = node2.get('metadata', {}).get('created_at', 0)
        
        newer = node1 if time1 > time2 else node2
        older = node2 if time1 > time2 else node1
        
        # Mark older as merged reference
        if self.hot_memory:
            older_content = self.hot_memory.get(older['id'])
            if older_content:
                self.hot_memory.put(
                    older['id'],
                    older_content,
                    metadata={
                        **older.get('metadata', {}),
                        'merged_into': newer['id'],
                        'merged_at': time.time()
                    }
                )
        
        self.total_merges += 1
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get merge agent statistics."""
        return {
            'total_merges': self.total_merges,
            'total_nodes_processed': self.total_nodes_processed,
            'similarity_threshold': self.similarity_threshold,
            'efficiency_ratio': (
                self.total_merges / self.total_nodes_processed 
                if self.total_nodes_processed > 0 else 0
            )
        }
