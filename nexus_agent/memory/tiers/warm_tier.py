"""Warm Tier Memory: Semantic retrieval via Vector/Graph stores."""

from __future__ import annotations
from typing import List, Dict, Any, Optional


class WarmTierMemory:
    """
    Semantic retrieval layer using vector and graph stores.
    Provides on-demand retrieval of relevant context.
    """
    
    def __init__(self, vector_store=None, graph_store=None):
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Statistics
        self.search_count = 0
        self.avg_results = 0
    
    async def search(
        self, 
        query: str, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search across vector and graph stores.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional filters (e.g., date range, tags)
            
        Returns:
            List of relevant memory items with scores
        """
        self.search_count += 1
        results = []
        
        # Search vector store for semantic similarity
        if self.vector_store:
            vector_results = await self._search_vector(query, limit, filters)
            results.extend(vector_results)
        
        # Search graph store for structural relationships
        if self.graph_store:
            graph_results = await self._search_graph(query, limit, filters)
            results.extend(graph_results)
        
        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for item in sorted(results, key=lambda x: x.get('score', 0), reverse=True):
            item_id = item.get('id')
            if item_id and item_id not in seen:
                seen.add(item_id)
                unique_results.append(item)
        
        # Update statistics
        self.avg_results = (
            (self.avg_results * (self.search_count - 1) + len(unique_results)) 
            / self.search_count
        )
        
        return unique_results[:limit]
    
    async def _search_vector(
        self, 
        query: str, 
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search vector store for semantically similar items."""
        try:
            # Embed query and search
            results = await self.vector_store.similarity_search(
                query=query,
                k=limit,
                filters=filters
            )
            
            # Format results
            formatted = []
            for doc in results:
                formatted.append({
                    'id': doc.metadata.get('id', str(hash(doc.page_content))),
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 0.0),
                    'source': 'vector'
                })
            
            return formatted
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    async def _search_graph(
        self, 
        query: str, 
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search graph store for related nodes."""
        try:
            # Extract keywords from query
            keywords = query.lower().split()
            
            # Find matching nodes
            results = await self.graph_store.find_relevant_nodes(
                keywords=keywords,
                limit=limit,
                filters=filters
            )
            
            # Format results
            formatted = []
            for node in results:
                formatted.append({
                    'id': node.get('id'),
                    'content': node.get('content', ''),
                    'metadata': node.get('metadata', {}),
                    'score': node.get('relevance_score', 0.0),
                    'source': 'graph',
                    'relationships': node.get('connections', [])
                })
            
            return formatted
        except Exception as e:
            print(f"Graph search error: {e}")
            return []
    
    async def get_related(self, item_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get items related to a specific item via graph traversal.
        
        Args:
            item_id: ID of the source item
            depth: How many hops to traverse
            
        Returns:
            List of related items
        """
        if not self.graph_store:
            return []
        
        try:
            related = await self.graph_store.get_neighbors(item_id, hops=depth)
            return related
        except Exception as e:
            print(f"Get related error: {e}")
            return []
    
    async def store(self, item_id: str, content: str, metadata: Dict[str, Any]):
        """
        Store an item in both vector and graph stores.
        
        Args:
            item_id: Unique identifier
            content: Content to store
            metadata: Associated metadata
        """
        tasks = []
        
        if self.vector_store:
            tasks.append(self._store_vector(item_id, content, metadata))
        
        if self.graph_store:
            tasks.append(self._store_graph(item_id, content, metadata))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _store_vector(self, item_id: str, content: str, metadata: Dict[str, Any]):
        """Store in vector store."""
        try:
            await self.vector_store.add_texts(
                texts=[content],
                metadatas=[{**metadata, 'id': item_id}]
            )
        except Exception as e:
            print(f"Vector store error: {e}")
    
    async def _store_graph(self, item_id: str, content: str, metadata: Dict[str, Any]):
        """Store in graph store."""
        try:
            await self.graph_store.add_node(
                node_id=item_id,
                content=content,
                metadata=metadata
            )
        except Exception as e:
            print(f"Graph store error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get warm tier statistics."""
        return {
            "search_count": self.search_count,
            "avg_results": self.avg_results,
            "has_vector_store": self.vector_store is not None,
            "has_graph_store": self.graph_store is not None
        }


# Import asyncio for the store method
import asyncio
