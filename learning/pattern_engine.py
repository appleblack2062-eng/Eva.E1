"""Pattern Generalization Engine for meta-learning.

This module provides pattern encoding, similarity matching, and workflow adaptation
capabilities for transferring workflows between similar tasks.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class WorkflowRef:
    """Reference to a workflow with metadata."""
    workflow_id: str
    spec: Dict[str, Any]
    task_description: str
    success_rate: float = 0.9
    usage_count: int = 0


class PatternEncoder:
    """Siamese network for task pattern embedding."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for PatternEncoder. "
                "Install with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
    
    def encode(self, task_desc: str, input_schema: Dict[str, Any]) -> np.ndarray:
        """Combine text + structural features into unified embedding."""
        # Encode text description
        text_emb = self.model.encode(task_desc, convert_to_numpy=True)
        
        # Encode schema structure
        struct_emb = self._encode_schema(input_schema)
        
        # Concatenate embeddings
        return np.concatenate([text_emb, struct_emb])
    
    def _encode_schema(self, schema: Dict[str, Any]) -> np.ndarray:
        """Convert JSON schema to fixed-length vector."""
        if not schema:
            return np.zeros(384)  # Match sentence transformer dimension
        
        # Serialize schema to string for embedding
        schema_str = str(sorted(schema.items()))
        schema_emb = self.model.encode(schema_str, convert_to_numpy=True)
        
        return schema_emb
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


class PatternGraph:
    """Knowledge graph of task patterns with similarity edges."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.encoder = PatternEncoder()
        self._embeddings_cache: Dict[str, np.ndarray] = {}
    
    def add_pattern(
        self, 
        pattern_id: str, 
        embedding: np.ndarray, 
        workflow_ref: WorkflowRef,
        metadata: Dict[str, Any]
    ):
        """Add pattern node and connect to k-nearest neighbors."""
        # Store embedding
        self._embeddings_cache[pattern_id] = embedding
        
        # Add node with attributes
        self.graph.add_node(
            pattern_id, 
            embedding=embedding, 
            workflow=workflow_ref, 
            **metadata
        )
        
        # Connect to k-nearest neighbors in latent space
        neighbors = self._find_knn(embedding, k=5, exclude={pattern_id})
        for nid, sim in neighbors:
            self.graph.add_edge(pattern_id, nid, relation="similar", weight=sim)
            # Add reverse edge for bidirectional similarity
            self.graph.add_edge(nid, pattern_id, relation="similar", weight=sim)
    
    def find_template(
        self, 
        query_embedding: np.ndarray, 
        min_sim: float = 0.75
    ) -> Optional[WorkflowRef]:
        """Find closest pattern and return its workflow as template."""
        candidates = self._find_knn(query_embedding, k=3)
        
        for nid, sim in candidates:
            if sim >= min_sim:
                node_data = self.graph.nodes[nid]
                return node_data.get("workflow")
        
        return None
    
    def adapt_workflow(
        self, 
        template: WorkflowRef, 
        diff_spec: Dict[str, Any],
        llm_client=None
    ) -> Optional[WorkflowRef]:
        """Use LLM to adapt template workflow to new task differences."""
        if llm_client is None:
            # Return template as-is if no LLM available
            return template
        
        prompt = f"""
Adapt this workflow template to handle these differences:

Template: {template.spec}
Differences: {diff_spec}

Return modified workflow spec as JSON.
"""
        # Call LLM and parse response
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(llm_client.generate(prompt, response_format="json"))
        
        # Create new workflow reference
        new_spec = response if isinstance(response, dict) else template.spec
        return WorkflowRef(
            workflow_id=f"{template.workflow_id}_adapted",
            spec=new_spec,
            task_description=template.task_description,
            success_rate=template.success_rate * 0.95,  # Slightly lower confidence
            usage_count=0
        )
    
    def _find_knn(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        exclude: set = None
    ) -> List[Tuple[str, float]]:
        """Find k-nearest neighbors by cosine similarity."""
        exclude = exclude or set()
        
        similarities = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_id in exclude:
                continue
            
            stored_emb = node_data.get("embedding")
            if stored_emb is None:
                continue
            
            sim = self.encoder.compute_similarity(query_embedding, stored_emb)
            similarities.append((node_id, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def get_similar_patterns(
        self, 
        query_embedding: np.ndarray, 
        min_sim: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[str, float, WorkflowRef]]:
        """Retrieve similar patterns with their workflows."""
        neighbors = self._find_knn(query_embedding, k=limit)
        
        results = []
        for nid, sim in neighbors:
            if sim >= min_sim:
                node_data = self.graph.nodes[nid]
                workflow = node_data.get("workflow")
                if workflow:
                    results.append((nid, sim, workflow))
        
        return results


class PatternGeneralizationEngine:
    """Main engine for pattern-based workflow generalization."""
    
    def __init__(self, config, embedding_provider=None):
        self.config = config
        self.pattern_graph = PatternGraph()
        
        # Use provided embedding provider or create default encoder
        if embedding_provider:
            self.encoder = embedding_provider
        else:
            self.encoder = PatternEncoder()
        
        self._pattern_stats: Dict[str, Dict[str, Any]] = {}
    
    async def encode_task_pattern(
        self, 
        task_desc: str, 
        input_schema: Dict[str, Any]
    ) -> np.ndarray:
        """Encode task into latent embedding space."""
        return self.encoder.encode(task_desc, input_schema)
    
    async def find_similar_workflow(
        self, 
        task_desc: str, 
        input_schema: Dict[str, Any],
        min_similarity: float = 0.75
    ) -> Optional[WorkflowRef]:
        """Find and return most similar workflow template."""
        embedding = self.encode_task_pattern(task_desc, input_schema)
        return self.pattern_graph.find_template(embedding, min_similarity)
    
    async def store_pattern(
        self,
        pattern_id: str,
        task_desc: str,
        input_schema: Dict[str, Any],
        workflow_ref: WorkflowRef,
        metadata: Dict[str, Any] = None
    ):
        """Store new pattern in the graph."""
        embedding = self.encode_task_pattern(task_desc, input_schema)
        
        self.pattern_graph.add_pattern(
            pattern_id=pattern_id,
            embedding=embedding,
            workflow_ref=workflow_ref,
            metadata=metadata or {}
        )
        
        # Initialize stats
        self._pattern_stats[pattern_id] = {
            "sample_count": 1,
            "last_used": time.time(),
            "success_count": 0,
            "total_cost": 0.0,
        }
    
    async def adapt_template(
        self,
        template: WorkflowRef,
        diff_spec: Dict[str, Any],
        llm_client=None
    ) -> WorkflowRef:
        """Adapt workflow template to new requirements."""
        return self.pattern_graph.adapt_workflow(template, diff_spec, llm_client)
    
    def update_pattern_stats(self, pattern_id: str, success: bool, cost: float = 0.0):
        """Update statistics for a pattern."""
        if pattern_id not in self._pattern_stats:
            self._pattern_stats[pattern_id] = {
                "sample_count": 0,
                "last_used": 0,
                "success_count": 0,
                "total_cost": 0.0,
            }
        
        stats = self._pattern_stats[pattern_id]
        stats["sample_count"] += 1
        stats["last_used"] = time.time()
        if success:
            stats["success_count"] += 1
        stats["total_cost"] += cost
    
    def get_pattern_stats(self, pattern_id: str) -> Dict[str, Any]:
        """Get statistics for a pattern."""
        if pattern_id not in self._pattern_stats:
            return {
                "sample_count": 0,
                "last_used_hours_ago": float('inf'),
                "success_rate": 0.0,
                "avg_cost": 0.0,
            }
        
        stats = self._pattern_stats[pattern_id]
        hours_since_use = (time.time() - stats["last_used"]) / 3600 if stats["last_used"] > 0 else float('inf')
        
        return {
            "sample_count": stats["sample_count"],
            "last_used_hours_ago": hours_since_use,
            "success_rate": stats["success_count"] / max(1, stats["sample_count"]),
            "avg_cost": stats["total_cost"] / max(1, stats["sample_count"]),
        }


# Import time at module level
import time
