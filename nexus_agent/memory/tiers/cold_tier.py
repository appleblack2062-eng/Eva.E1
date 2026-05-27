"""Cold Tier Memory: Long-term compressed archive."""

from __future__ import annotations
import json
import gzip
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ArchivedNode:
    """A node in cold storage."""
    node_id: str
    content_hash: str
    archive_path: str
    metadata: Dict[str, Any]
    archived_at: float
    original_size: int
    compressed_size: int


class ColdTierMemory:
    """
    Long-term archive storage with O(1) root index lookup.
    Provides compression and efficient retrieval.
    """
    
    def __init__(self, archive_path: str, index_path: Optional[str] = None):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Index for O(1) lookups
        self.index_path = Path(index_path) if index_path else self.archive_path / "index.json"
        self.index: Dict[str, ArchivedNode] = {}
        
        # Load existing index
        self._load_index()
        
        # Statistics
        self.total_archived = len(self.index)
        self.total_restored = 0
        self.compression_ratio = 0.0
    
    def _load_index(self):
        """Load the index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    for node_id, node_data in data.items():
                        self.index[node_id] = ArchivedNode(**node_data)
            except Exception as e:
                print(f"Error loading cold tier index: {e}")
    
    def _save_index(self):
        """Save the index to disk."""
        try:
            with open(self.index_path, 'w') as f:
                data = {
                    node_id: {
                        'node_id': node.node_id,
                        'content_hash': node.content_hash,
                        'archive_path': node.archive_path,
                        'metadata': node.metadata,
                        'archived_at': node.archived_at,
                        'original_size': node.original_size,
                        'compressed_size': node.compressed_size
                    }
                    for node_id, node in self.index.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving cold tier index: {e}")
    
    async def archive(self, node_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Move node to compressed storage.
        
        Args:
            node_id: Unique identifier for the node
            content: Content to archive
            metadata: Optional metadata
            
        Returns:
            Archive path
        """
        import hashlib
        
        # Compute hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Create archive file
        timestamp = int(time.time() * 1000)
        archive_file = self.archive_path / f"{node_id}_{timestamp}.gz"
        
        # Compress and write
        original_size = len(content.encode('utf-8'))
        with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
            f.write(content)
        
        compressed_size = archive_file.stat().st_size
        
        # Create archived node record
        archived_node = ArchivedNode(
            node_id=node_id,
            content_hash=content_hash,
            archive_path=str(archive_file),
            metadata=metadata or {},
            archived_at=time.time(),
            original_size=original_size,
            compressed_size=compressed_size
        )
        
        # Update index
        self.index[node_id] = archived_node
        self.total_archived += 1
        
        # Update compression ratio
        if self.total_archived > 0:
            total_original = sum(n.original_size for n in self.index.values())
            total_compressed = sum(n.compressed_size for n in self.index.values())
            self.compression_ratio = total_compressed / total_original if total_original > 0 else 0
        
        # Save index
        self._save_index()
        
        return str(archive_file)
    
    async def restore(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Load node back from cold storage.
        
        Args:
            node_id: ID of node to restore
            
        Returns:
            Dictionary with content and metadata, or None if not found
        """
        if node_id not in self.index:
            return None
        
        archived_node = self.index[node_id]
        
        try:
            # Read and decompress
            with gzip.open(archived_node.archive_path, 'rt', encoding='utf-8') as f:
                content = f.read()
            
            self.total_restored += 1
            
            return {
                'node_id': node_id,
                'content': content,
                'metadata': archived_node.metadata,
                'content_hash': archived_node.content_hash,
                'archived_at': archived_node.archived_at
            }
        except Exception as e:
            print(f"Error restoring node {node_id}: {e}")
            return None
    
    async def delete(self, node_id: str) -> bool:
        """
        Permanently delete an archived node.
        
        Args:
            node_id: ID of node to delete
            
        Returns:
            True if deleted successfully
        """
        if node_id not in self.index:
            return False
        
        archived_node = self.index[node_id]
        
        try:
            # Remove archive file
            archive_file = Path(archived_node.archive_path)
            if archive_file.exists():
                archive_file.unlink()
            
            # Remove from index
            del self.index[node_id]
            self._save_index()
            
            return True
        except Exception as e:
            print(f"Error deleting node {node_id}: {e}")
            return False
    
    def exists(self, node_id: str) -> bool:
        """Check if a node exists in cold storage."""
        return node_id in self.index
    
    def get_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a node without restoring content."""
        if node_id not in self.index:
            return None
        return self.index[node_id].metadata
    
    def list_archived(self, limit: int = 100, offset: int = 0) -> List[ArchivedNode]:
        """
        List archived nodes.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of archived node records
        """
        nodes = list(self.index.values())
        return nodes[offset:offset + limit]
    
    def search_by_metadata(self, key: str, value: Any = None) -> List[str]:
        """
        Search for nodes by metadata.
        
        Args:
            key: Metadata key to search
            value: Optional value to match
            
        Returns:
            List of matching node IDs
        """
        matches = []
        for node_id, node in self.index.items():
            if key in node.metadata:
                if value is None or node.metadata[key] == value:
                    matches.append(node_id)
        return matches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cold tier statistics."""
        total_original = sum(n.original_size for n in self.index.values())
        total_compressed = sum(n.compressed_size for n in self.index.values())
        
        return {
            "total_archived": self.total_archived,
            "total_restored": self.total_restored,
            "total_nodes": len(self.index),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": self.compression_ratio,
            "space_saved_bytes": total_original - total_compressed,
            "space_saved_percent": (1 - self.compression_ratio) * 100 if self.compression_ratio > 0 else 0
        }
