"""
Global Hive Mind - Centralized Self-Correcting Knowledge Layer

The shared cognitive layer that aggregates refined knowledge from all workspace
"offices". Contains three critical layers:
- Semantic Layer: Domain knowledge and constraints
- Procedural Layer: Workflow patterns and execution profiles
- Refined Worker Model: Optimized instruction sets for worker training

Features:
- Isomorphic graph consolidation for pattern deduplication
- Causal contradiction resolution with confidence scoring
- Continuous worker evolution through verified pattern packaging
- Zero-cold-start workspace bootstrapping
"""

import uuid
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json

from .ecl_format import ECLLedger, ECLEntry, ECLPrimitiveType, ECLParser


class HiveMindLayerType(Enum):
    """Types of layers in the Global Hive Mind"""
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    REFINED_MODEL = "refined_model"


@dataclass
class KnowledgeAsset:
    """
    A single unit of knowledge in the Hive Mind.
    
    Contains abstracted, client-scrubbed information extracted from
    local .ecl deltas.
    """
    asset_id: str
    layer_type: HiveMindLayerType
    content: Any
    source_space_ids: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    execution_count: int = 0
    avg_latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    is_deprecated: bool = False
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_confidence_score(self) -> float:
        """
        Compute confidence score using the formula:
        Confidence = (Success Rate × Execution Count) / Latency(ms)
        """
        if self.avg_latency_ms <= 0:
            return self.success_rate * min(self.execution_count, 10)
        
        raw_score = (self.success_rate * self.execution_count) / self.avg_latency_ms
        # Normalize to 0-1 range with sigmoid-like scaling
        normalized = raw_score / (raw_score + 1)
        return min(1.0, normalized)
    
    def update_metrics(
        self,
        success: bool,
        latency_ms: float,
        execution_time: Optional[datetime] = None
    ) -> None:
        """Update asset metrics with new execution data"""
        old_count = self.execution_count
        
        self.execution_count += 1
        self.success_rate = (
            (self.success_rate * old_count + (1 if success else 0)) /
            self.execution_count
        )
        
        # Exponential moving average for latency
        alpha = 0.3
        self.avg_latency_ms = (
            alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
        )
        
        self.last_updated = execution_time or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'asset_id': self.asset_id,
            'layer_type': self.layer_type.value,
            'content': self.content,
            'source_space_ids': self.source_space_ids,
            'success_rate': self.success_rate,
            'execution_count': self.execution_count,
            'avg_latency_ms': self.avg_latency_ms,
            'confidence_score': self.compute_confidence_score(),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'version': self.version,
            'is_deprecated': self.is_deprecated,
            'superseded_by': self.superseded_by,
            'metadata': self.metadata
        }


@dataclass
class StructuralPattern:
    """
    Represents a structural workflow pattern extracted from .ecl deltas.
    
    Used for isomorphic graph matching and consolidation.
    """
    pattern_id: str
    dfg_signature: str  # Data-Flow Graph signature hash
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[str, str, str]]  # (from_node, to_node, edge_type)
    execution_profile: Dict[str, Any]
    isomorphism_class: str = ""
    variations: List[str] = field(default_factory=list)
    
    def compute_dfg_signature(self) -> str:
        """Compute deterministic signature for isomorphism detection"""
        # Sort nodes and edges for canonical representation
        sorted_nodes = sorted(
            [json.dumps(n, sort_keys=True) for n in self.nodes]
        )
        sorted_edges = sorted(
            [f"{e[0]}->{e[1]}:{e[2]}" for e in self.edges]
        )
        
        canonical = "|".join(sorted_nodes + sorted_edges)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def is_isomorphic_to(self, other: 'StructuralPattern') -> bool:
        """Check if this pattern is structurally isomorphic to another"""
        return self.dfg_signature == other.dfg_signature


class HiveMindLayer:
    """
    Single layer within the Global Hive Mind.
    
    Manages knowledge assets of a specific type (semantic, procedural, or refined model).
    """
    
    def __init__(self, layer_type: HiveMindLayerType):
        self.layer_type = layer_type
        self.assets: Dict[str, KnowledgeAsset] = {}
        self.patterns: Dict[str, StructuralPattern] = {}
        self._lock = threading.RLock()
        
        # Indexes for fast lookup
        self._source_index: Dict[str, Set[str]] = defaultdict(set)  # space_id -> asset_ids
        self._pattern_index: Dict[str, str] = {}  # dfg_signature -> asset_id
    
    def add_asset(self, asset: KnowledgeAsset) -> str:
        """Add knowledge asset to layer"""
        with self._lock:
            self.assets[asset.asset_id] = asset
            
            # Update indexes
            for space_id in asset.source_space_ids:
                self._source_index[space_id].add(asset.asset_id)
            
            return asset.asset_id
    
    def get_asset(self, asset_id: str) -> Optional[KnowledgeAsset]:
        """Retrieve asset by ID"""
        with self._lock:
            return self.assets.get(asset_id)
    
    def deprecate_asset(self, asset_id: str, superseded_by: str) -> None:
        """Mark asset as deprecated in favor of newer version"""
        with self._lock:
            if asset_id in self.assets:
                asset = self.assets[asset_id]
                asset.is_deprecated = True
                asset.superseded_by = superseded_by
    
    def find_by_pattern(self, dfg_signature: str) -> List[KnowledgeAsset]:
        """Find assets matching a structural pattern"""
        with self._lock:
            asset_id = self._pattern_index.get(dfg_signature)
            if asset_id and asset_id in self.assets:
                return [self.assets[asset_id]]
            return []
    
    def get_active_assets(self) -> List[KnowledgeAsset]:
        """Get all non-deprecated assets"""
        with self._lock:
            return [a for a in self.assets.values() if not a.is_deprecated]
    
    def get_top_assets(self, limit: int = 10) -> List[KnowledgeAsset]:
        """Get top assets by confidence score"""
        with self._lock:
            active = self.get_active_assets()
            sorted_assets = sorted(
                active,
                key=lambda a: a.compute_confidence_score(),
                reverse=True
            )
            return sorted_assets[:limit]
    
    def merge_isomorphic_patterns(
        self,
        pattern1: StructuralPattern,
        pattern2: StructuralPattern
    ) -> StructuralPattern:
        """
        Merge two isomorphic patterns into optimized unified pattern.
        
        Combines execution profiles and variations from both patterns.
        """
        if not pattern1.is_isomorphic_to(pattern2):
            raise ValueError("Patterns are not isomorphic")
        
        merged = StructuralPattern(
            pattern_id=f"merged_{pattern1.pattern_id}_{pattern2.pattern_id}",
            dfg_signature=pattern1.dfg_signature,
            nodes=pattern1.nodes,  # Structurally identical
            edges=pattern1.edges,
            execution_profile={
                **pattern1.execution_profile,
                **pattern2.execution_profile
            },
            isomorphism_class=pattern1.isomorphism_class or pattern1.dfg_signature[:8],
            variations=list(set(pattern1.variations + pattern2.variations))
        )
        
        return merged
    
    def stats(self) -> Dict[str, Any]:
        """Get layer statistics"""
        with self._lock:
            active_count = sum(1 for a in self.assets.values() if not a.is_deprecated)
            deprecated_count = sum(1 for a in self.assets.values() if a.is_deprecated)
            
            return {
                'layer_type': self.layer_type.value,
                'total_assets': len(self.assets),
                'active_assets': active_count,
                'deprecated_assets': deprecated_count,
                'patterns': len(self.patterns),
                'unique_sources': len(self._source_index),
                'avg_confidence': (
                    sum(a.compute_confidence_score() for a in self.assets.values()) /
                    len(self.assets) if self.assets else 0
                )
            }


class GlobalHiveMind:
    """
    The centralized Global Hive Mind - shared cognitive layer across all workspaces.
    
    Aggregates refined knowledge from local .ecl deltas, performs isomorphic
    consolidation, and serves as the source for:
    - Continuous worker evolution (training datasets)
    - Zero-cold-start workspace bootstrapping
    """
    
    def __init__(self):
        self.layers: Dict[HiveMindLayerType, HiveMindLayer] = {
            layer_type: HiveMindLayer(layer_type)
            for layer_type in HiveMindLayerType
        }
        
        self._lock = threading.RLock()
        self.created_at = datetime.utcnow()
        self.total_uplinks = 0
        self.total_consolidations = 0
        
        # Bootstrap cache for quick workspace initialization
        self._bootstrap_cache: Dict[str, ECLLedger] = {}
    
    def get_layer(self, layer_type: HiveMindLayerType) -> HiveMindLayer:
        """Get specific layer"""
        return self.layers[layer_type]
    
    def ingest_knowledge(
        self,
        content: Any,
        layer_type: HiveMindLayerType,
        source_space_id: str,
        success_rate: float = 1.0,
        execution_count: int = 1,
        avg_latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingest new knowledge from a local workspace uplink.
        
        Content is scrubbed of client-specific data before ingestion.
        """
        with self._lock:
            asset_id = str(uuid.uuid4())[:8]
            
            asset = KnowledgeAsset(
                asset_id=asset_id,
                layer_type=layer_type,
                content=content,
                source_space_ids=[source_space_id],
                success_rate=success_rate,
                execution_count=execution_count,
                avg_latency_ms=avg_latency_ms,
                metadata=metadata or {}
            )
            
            layer = self.layers[layer_type]
            layer.add_asset(asset)
            
            self.total_uplinks += 1
            
            return asset_id
    
    def ingest_structural_pattern(
        self,
        pattern: StructuralPattern,
        layer_type: HiveMindLayerType,
        source_space_id: str,
        execution_profile: Dict[str, Any]
    ) -> str:
        """
        Ingest a structural workflow pattern.
        
        Performs isomorphic matching against existing patterns and
        consolidates if match found.
        """
        with self._lock:
            # Compute signature
            pattern.dfg_signature = pattern.compute_dfg_signature()
            pattern.isomorphism_class = pattern.dfg_signature[:8]
            
            layer = self.layers[layer_type]
            
            # Check for isomorphic match
            existing_assets = layer.find_by_pattern(pattern.dfg_signature)
            
            if existing_assets:
                # Consolidate with existing
                existing_asset = existing_assets[0]
                
                # Update existing asset with new data
                existing_asset.source_space_ids.append(source_space_id)
                existing_asset.execution_count += 1
                
                # Merge execution profiles
                if 'execution_profile' in existing_asset.metadata:
                    merged_profile = {
                        **existing_asset.metadata['execution_profile'],
                        **execution_profile
                    }
                    existing_asset.metadata['execution_profile'] = merged_profile
                
                self.total_consolidations += 1
                
                return existing_asset.asset_id
            else:
                # New unique pattern
                asset_id = str(uuid.uuid4())[:8]
                
                asset = KnowledgeAsset(
                    asset_id=asset_id,
                    layer_type=layer_type,
                    content={
                        'pattern_id': pattern.pattern_id,
                        'nodes': pattern.nodes,
                        'edges': pattern.edges,
                        'isomorphism_class': pattern.isomorphism_class
                    },
                    source_space_ids=[source_space_id],
                    metadata={'execution_profile': execution_profile}
                )
                
                layer.add_asset(asset)
                layer.patterns[pattern.pattern_id] = pattern
                layer._pattern_index[pattern.dfg_signature] = asset_id
                
                self.total_uplinks += 1
                
                return asset_id
    
    def bootstrap_workspace(self, project_description: str) -> ECLLedger:
        """
        Generate pre-seeded .ecl ledger for new workspace.
        
        Searches historical graph for similar setups and generates
        optimized starting configuration.
        """
        with self._lock:
            # Check cache first
            desc_hash = hashlib.sha256(project_description.encode()).hexdigest()[:16]
            if desc_hash in self._bootstrap_cache:
                # Return cached template
                cached = self._bootstrap_cache[desc_hash]
                return ECLLedger(
                    space_id=f"NEW_{desc_hash}",
                    security_constraints=cached.security_constraints.copy()
                )
            
            # Find relevant patterns from procedural layer
            procedural_layer = self.layers[HiveMindLayerType.PROCEDURAL]
            top_patterns = procedural_layer.get_top_assets(limit=20)
            
            # Find relevant semantic knowledge
            semantic_layer = self.layers[HiveMindLayerType.SEMANTIC]
            top_semantic = semantic_layer.get_top_assets(limit=10)
            
            # Generate pre-seeded ledger
            space_id = f"SPACE_{desc_hash}"
            ledger = ECLLedger(space_id=space_id)
            
            # Add semantic root with common domain keys
            semantic_section = ledger.add_section('SEMANTIC_ROOT::STATE')
            
            # Extract common domain keys from top semantic assets
            common_domains = set()
            for asset in top_semantic:
                if isinstance(asset.content, dict) and 'domain_keys' in asset.content:
                    common_domains.update(asset.content['domain_keys'])
            
            if common_domains:
                entry = ECLEntry(
                    primitive_type=ECLPrimitiveType.DOMAIN_KEY,
                    key="bootstrap_domains",
                    value=list(common_domains)[:10]  # Limit to top 10
                )
                semantic_section.add_entry(entry)
            
            # Add procedural vectors from top patterns
            proc_section = ledger.add_section('PROCEDURAL_VECTORS')
            
            for i, asset in enumerate(top_patterns[:5]):  # Top 5 patterns
                if 'pattern_id' in asset.content:
                    entry = ECLEntry(
                        primitive_type=ECLPrimitiveType.OPERATION,
                        key=f"boot_{i}",
                        value=asset.content.get('isomorphism_class', 'pattern'),
                        metadata={
                            'op_name': 'BootstrapPattern',
                            'node_ref': f'$NODE_REF_BOOT_{i}',
                            'exec_profile': '%EXEC_PROFILE_HIGH_CONFIDENCE',
                            'confidence': asset.compute_confidence_score()
                        }
                    )
                    proc_section.add_entry(entry)
            
            # Cache for future use
            self._bootstrap_cache[desc_hash] = ledger
            
            return ledger
    
    def get_training_dataset(
        self,
        min_confidence: float = 0.7,
        min_executions: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Package highly verified procedural patterns into training datasets.
        
        Used for continuous worker evolution and SLM fine-tuning.
        """
        with self._lock:
            dataset = {
                'procedural': [],
                'semantic': [],
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'min_confidence_threshold': min_confidence,
                    'min_executions_threshold': min_executions
                }
            }
            
            # Extract high-confidence procedural patterns
            proc_layer = self.layers[HiveMindLayerType.PROCEDURAL]
            for asset in proc_layer.get_active_assets():
                if (asset.compute_confidence_score() >= min_confidence and
                    asset.execution_count >= min_executions):
                    dataset['procedural'].append({
                        'pattern': asset.content,
                        'confidence': asset.compute_confidence_score(),
                        'success_rate': asset.success_rate,
                        'execution_count': asset.execution_count,
                        'metadata': asset.metadata
                    })
            
            # Extract high-confidence semantic knowledge
            sem_layer = self.layers[HiveMindLayerType.SEMANTIC]
            for asset in sem_layer.get_active_assets():
                if (asset.compute_confidence_score() >= min_confidence and
                    asset.execution_count >= min_executions):
                    dataset['semantic'].append({
                        'knowledge': asset.content,
                        'confidence': asset.compute_confidence_score(),
                        'domain_context': asset.metadata.get('domain', 'general')
                    })
            
            return dataset
    
    def stats(self) -> Dict[str, Any]:
        """Get overall Hive Mind statistics"""
        with self._lock:
            return {
                'created_at': self.created_at.isoformat(),
                'total_uplinks': self.total_uplinks,
                'total_consolidations': self.total_consolidations,
                'consolidation_ratio': (
                    self.total_consolidations / self.total_uplinks
                    if self.total_uplinks > 0 else 0
                ),
                'layers': {
                    lt.value: layer.stats()
                    for lt, layer in self.layers.items()
                },
                'bootstrap_cache_size': len(self._bootstrap_cache)
            }
