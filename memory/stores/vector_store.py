"""Vector storage for semantic task retrieval using ChromaDB."""

from __future__ import annotations
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np

class TaskVectorStore:
    """Handles semantic indexing of task descriptions and inputs."""
    
    def __init__(self, agent_id: str, config, embedding_provider):
        self.agent_id = agent_id
        self.config = config
        self.embedder = embedding_provider
        
        # Isolated collection per agent
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection(
            name=f"nexus_tasks_{agent_id}",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def add_task(self, task_id: str, description: str, input_data: Any, metadata: Dict[str, Any]):
        """Index a task for future retrieval."""
        # Combine description and stringified input for richer context
        content = f"{description} Input: {str(input_data)[:500]}"
        embedding = self.embedder.embed(content).tolist()
        
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "task_id": task_id,
                "success": metadata.get("success", False),
                "latency_ms": metadata.get("latency_ms", 0),
                "mode": metadata.get("mode", "LLM_ONLY"),
                **{k: str(v) for k, v in metadata.items() if isinstance(v, (str, int, float))}
            }],
            ids=[task_id]
        )
    
    async def search_similar(self, query: str, limit: int = 10, min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Find semantically similar past tasks."""
        embedding = self.embedder.embed(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit * 2,  # Fetch more to filter by similarity
            include=["metadatas", "documents", "distances"]
        )
        
        matches = []
        if results['ids'][0]:
            for i, dist in enumerate(results['distances'][0]):
                similarity = 1.0 - dist
                if similarity >= min_similarity:
                    matches.append({
                        "task_id": results['ids'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "content": results['documents'][0][i],
                        "similarity": similarity
                    })
        
        return matches[:limit]
