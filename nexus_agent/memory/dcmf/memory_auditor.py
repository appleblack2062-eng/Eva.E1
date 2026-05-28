"""
Memory Auditor Agent & Self-Correcting Engine

Dedicated agent that guards the Global Hive Mind from becoming an unorganized
data swamp. Performs conflict detection, isomorphic consolidation, and causal
contradiction resolution.

Core Mechanisms:
- Isomorphic Graph Consolidation: Merges structurally similar workflows
- Causal Contradiction Resolution: Resolves conflicts using confidence scoring
- Self-Correction Pipeline: Verifies and hot-patches global assets
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime
from enum import Enum
import json

from .ecl_format import ECLLedger, ECLEntry, ECLPrimitiveType
from .hive_mind import (
    GlobalHiveMind, KnowledgeAsset, StructuralPattern, 
    HiveMindLayerType, HiveMindLayer
)


class ConflictType(Enum):
    """Types of conflicts detected by the Auditor"""
    CONTRADICTION = "contradiction"  # New data contradicts existing
    DUPLICATE = "duplicate"  # Exact or near-exact duplicate
    OUTDATED = "outdated"  # Existing asset is outdated by new
    ISOMORPHIC = "isomorphic"  # Structurally identical patterns
    SCHEMA_MISMATCH = "schema_mismatch"  # Incompatible schema definitions


@dataclass
class ConflictReport:
    """Report generated when conflict is detected"""
    conflict_id: str
    conflict_type: ConflictType
    new_asset_id: str
    existing_asset_id: str
    confidence_delta: float
    resolution_recommendation: str
    auto_resolvable: bool
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuditResult:
    """Result of auditing a knowledge uplink"""
    accepted: bool
    action_taken: str  # 'accepted', 'merged', 'rejected', 'pending_review'
    conflict_report: Optional[ConflictReport] = None
    merged_asset_id: Optional[str] = None
    deprecated_asset_id: Optional[str] = None
    confidence_score: float = 0.0
    audit_timestamp: datetime = field(default_factory=datetime.utcnow)


class ConflictResolver:
    """
    Resolves conflicts between new and existing knowledge assets.
    
    Uses confidence scoring and causal analysis to determine optimal resolution.
    """
    
    def __init__(self, hive_mind: GlobalHiveMind):
        self.hive_mind = hive_mind
        self._lock = threading.RLock()
        self.resolution_history: List[AuditResult] = []
    
    def resolve_conflict(
        self,
        new_asset: KnowledgeAsset,
        existing_asset: KnowledgeAsset,
        conflict_type: ConflictType
    ) -> AuditResult:
        """
        Resolve conflict between new and existing assets.
        
        Returns audit result with recommended action.
        """
        with self._lock:
            new_confidence = new_asset.compute_confidence_score()
            existing_confidence = existing_asset.compute_confidence_score()
            confidence_delta = new_confidence - existing_confidence
            
            # Generate conflict report
            conflict_report = ConflictReport(
                conflict_id=hashlib.sha256(
                    f"{new_asset.asset_id}_{existing_asset.asset_id}".encode()
                ).hexdigest()[:8],
                conflict_type=conflict_type,
                new_asset_id=new_asset.asset_id,
                existing_asset_id=existing_asset.asset_id,
                confidence_delta=confidence_delta,
                resolution_recommendation=self._get_resolution_recommendation(
                    conflict_type, confidence_delta
                ),
                auto_resolvable=self._is_auto_resolvable(conflict_type, confidence_delta),
                details={
                    'new_confidence': new_confidence,
                    'existing_confidence': existing_confidence,
                    'new_executions': new_asset.execution_count,
                    'existing_executions': existing_asset.execution_count
                }
            )
            
            # Determine action
            if conflict_type == ConflictType.ISOMORPHIC:
                # Merge isomorphic patterns
                return self._resolve_isomorphic(
                    new_asset, existing_asset, conflict_report
                )
            elif conflict_type == ConflictType.CONTRADICTION:
                # Use confidence-based resolution
                return self._resolve_contradiction(
                    new_asset, existing_asset, conflict_report
                )
            elif conflict_type == ConflictType.DUPLICATE:
                # Reject duplicate
                return self._resolve_duplicate(
                    new_asset, existing_asset, conflict_report
                )
            elif conflict_type == ConflictType.OUTDATED:
                # Deprecate old, accept new
                return self._resolve_outdated(
                    new_asset, existing_asset, conflict_report
                )
            else:
                # Default: accept both with note
                return AuditResult(
                    accepted=True,
                    action_taken='accepted',
                    conflict_report=conflict_report,
                    confidence_score=new_confidence
                )
    
    def _get_resolution_recommendation(
        self,
        conflict_type: ConflictType,
        confidence_delta: float
    ) -> str:
        """Get human-readable resolution recommendation"""
        recommendations = {
            ConflictType.ISOMORPHIC: "Merge patterns and consolidate execution profiles",
            ConflictType.CONTRADICTION: (
                "Accept higher confidence asset" if confidence_delta > 0.1
                else "Review manually - confidence scores too close"
            ),
            ConflictType.DUPLICATE: "Reject duplicate entry",
            ConflictType.OUTDATED: "Deprecate old asset, accept new version"
        }
        return recommendations.get(conflict_type, "Manual review required")
    
    def _is_auto_resolvable(
        self,
        conflict_type: ConflictType,
        confidence_delta: float
    ) -> bool:
        """Determine if conflict can be auto-resolved"""
        if conflict_type == ConflictType.ISOMORPHIC:
            return True
        elif conflict_type == ConflictType.DUPLICATE:
            return True
        elif conflict_type == ConflictType.OUTDATED:
            return True
        elif conflict_type == ConflictType.CONTRADICTION:
            # Auto-resolve only if confidence delta is significant
            return abs(confidence_delta) > 0.2
        return False
    
    def _resolve_isomorphic(
        self,
        new_asset: KnowledgeAsset,
        existing_asset: KnowledgeAsset,
        conflict_report: ConflictReport
    ) -> AuditResult:
        """Resolve isomorphic pattern conflict by merging"""
        layer = self.hive_mind.get_layer(new_asset.layer_type)
        
        # Merge execution profiles
        merged_profile = {
            **existing_asset.metadata.get('execution_profile', {}),
            **new_asset.metadata.get('execution_profile', {})
        }
        
        # Update existing asset
        existing_asset.source_space_ids.extend(
            sid for sid in new_asset.source_space_ids
            if sid not in existing_asset.source_space_ids
        )
        existing_asset.execution_count += new_asset.execution_count
        
        # Recalculate success rate
        total_successes = (
            existing_asset.success_rate * 
            (existing_asset.execution_count - new_asset.execution_count) +
            new_asset.success_rate * new_asset.execution_count
        )
        existing_asset.success_rate = (
            total_successes / existing_asset.execution_count
            if existing_asset.execution_count > 0 else 0
        )
        
        existing_asset.metadata['execution_profile'] = merged_profile
        existing_asset.last_updated = datetime.utcnow()
        
        return AuditResult(
            accepted=True,
            action_taken='merged',
            conflict_report=conflict_report,
            merged_asset_id=existing_asset.asset_id,
            confidence_score=existing_asset.compute_confidence_score()
        )
    
    def _resolve_contradiction(
        self,
        new_asset: KnowledgeAsset,
        existing_asset: KnowledgeAsset,
        conflict_report: ConflictReport
    ) -> AuditResult:
        """Resolve contradiction using confidence scoring"""
        new_confidence = new_asset.compute_confidence_score()
        existing_confidence = existing_asset.compute_confidence_score()
        
        if new_confidence > existing_confidence + 0.1:
            # New asset wins - deprecate old
            layer = self.hive_mind.get_layer(existing_asset.layer_type)
            layer.deprecate_asset(existing_asset.asset_id, new_asset.asset_id)
            
            return AuditResult(
                accepted=True,
                action_taken='accepted',
                conflict_report=conflict_report,
                deprecated_asset_id=existing_asset.asset_id,
                confidence_score=new_confidence
            )
        elif existing_confidence > new_confidence + 0.1:
            # Existing asset wins - reject new
            return AuditResult(
                accepted=False,
                action_taken='rejected',
                conflict_report=conflict_report,
                confidence_score=existing_confidence
            )
        else:
            # Too close to call - mark for review
            conflict_report.auto_resolvable = False
            return AuditResult(
                accepted=True,
                action_taken='pending_review',
                conflict_report=conflict_report,
                confidence_score=max(new_confidence, existing_confidence)
            )
    
    def _resolve_duplicate(
        self,
        new_asset: KnowledgeAsset,
        existing_asset: KnowledgeAsset,
        conflict_report: ConflictReport
    ) -> AuditResult:
        """Reject duplicate entries"""
        return AuditResult(
            accepted=False,
            action_taken='rejected',
            conflict_report=conflict_report,
            confidence_score=existing_asset.compute_confidence_score()
        )
    
    def _resolve_outdated(
        self,
        new_asset: KnowledgeAsset,
        existing_asset: KnowledgeAsset,
        conflict_report: ConflictReport
    ) -> AuditResult:
        """Deprecate outdated asset in favor of new version"""
        layer = self.hive_mind.get_layer(existing_asset.layer_type)
        layer.deprecate_asset(existing_asset.asset_id, new_asset.asset_id)
        
        return AuditResult(
            accepted=True,
            action_taken='accepted',
            conflict_report=conflict_report,
            deprecated_asset_id=existing_asset.asset_id,
            confidence_score=new_asset.compute_confidence_score()
        )


class IsomorphicConsolidator:
    """
    Identifies and consolidates isomorphic structural patterns.
    
    Detects when different workspaces discover structurally identical
    workflows and merges them into unified global templates.
    """
    
    def __init__(self, hive_mind: GlobalHiveMind):
        self.hive_mind = hive_mind
        self._consolidation_cache: Dict[str, str] = {}  # signature -> merged_asset_id
    
    def find_isomorphic_patterns(
        self,
        new_pattern: StructuralPattern,
        layer_type: HiveMindLayerType
    ) -> List[KnowledgeAsset]:
        """Find existing patterns isomorphic to the new pattern"""
        layer = self.hive_mind.get_layer(layer_type)
        
        # Compute signature
        signature = new_pattern.compute_dfg_signature()
        
        # Check cache first
        if signature in self._consolidation_cache:
            asset_id = self._consolidation_cache[signature]
            asset = layer.get_asset(asset_id)
            if asset and not asset.is_deprecated:
                return [asset]
        
        # Search layer's pattern index
        return layer.find_by_pattern(signature)
    
    def consolidate(
        self,
        new_pattern: StructuralPattern,
        existing_assets: List[KnowledgeAsset],
        source_space_id: str,
        execution_profile: Dict[str, Any]
    ) -> Tuple[KnowledgeAsset, bool]:
        """
        Consolidate new pattern with existing isomorphic patterns.
        
        Returns (consolidated_asset, was_merged).
        """
        if not existing_assets:
            # No existing patterns - create new
            asset = KnowledgeAsset(
                asset_id=new_pattern.pattern_id,
                layer_type=HiveMindLayerType.PROCEDURAL,
                content={
                    'pattern_id': new_pattern.pattern_id,
                    'nodes': new_pattern.nodes,
                    'edges': new_pattern.edges,
                    'isomorphism_class': new_pattern.isomorphism_class
                },
                source_space_ids=[source_space_id],
                metadata={'execution_profile': execution_profile}
            )
            return asset, False
        
        # Merge with first existing asset
        existing = existing_assets[0]
        
        # Update source tracking
        if source_space_id not in existing.source_space_ids:
            existing.source_space_ids.append(source_space_id)
        
        # Merge execution profiles
        if 'execution_profile' in existing.metadata:
            merged = {
                **existing.metadata['execution_profile'],
                **execution_profile
            }
            existing.metadata['execution_profile'] = merged
        
        # Update metrics
        existing.execution_count += 1
        
        # Cache signature
        signature = new_pattern.compute_dfg_signature()
        self._consolidation_cache[signature] = existing.asset_id
        
        return existing, True
    
    def get_isomorphism_stats(self) -> Dict[str, Any]:
        """Get statistics about isomorphic consolidations"""
        layer = self.hive_mind.get_layer(HiveMindLayerType.PROCEDURAL)
        
        total_patterns = len(layer.patterns)
        cached_signatures = len(self._consolidation_cache)
        
        # Count assets with multiple sources (indicating consolidation)
        multi_source_count = sum(
            1 for asset in layer.assets.values()
            if len(asset.source_space_ids) > 1
        )
        
        return {
            'total_unique_patterns': total_patterns,
            'cached_signatures': cached_signatures,
            'consolidated_patterns': multi_source_count,
            'consolidation_efficiency': (
                multi_source_count / total_patterns
                if total_patterns > 0 else 0
            )
        }


class MemoryAuditor:
    """
    Main Memory Auditor Agent - gatekeeper of the Global Hive Mind.
    
    Orchestrates conflict resolution, isomorphic consolidation, and
    self-correction pipelines.
    """
    
    def __init__(self, hive_mind: GlobalHiveMind):
        self.hive_mind = hive_mind
        self.conflict_resolver = ConflictResolver(hive_mind)
        self.isomorphic_consolidator = IsomorphicConsolidator(hive_mind)
        
        self._lock = threading.RLock()
        self.audit_log: List[AuditResult] = []
        self.pending_reviews: List[AuditResult] = []
        
        # Metrics
        self.total_audits = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_merged = 0
        self.total_deprecated = 0
    
    def audit_uplink(
        self,
        content: Any,
        layer_type: HiveMindLayerType,
        source_space_id: str,
        success_rate: float = 1.0,
        execution_count: int = 1,
        avg_latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        is_structural_pattern: bool = False
    ) -> AuditResult:
        """
        Audit a knowledge uplink from local workspace.
        
        Full pipeline:
        1. Isolate client-private info
        2. Check for conflicts
        3. Run isomorphic consolidation if applicable
        4. Resolve contradictions
        5. Inject into Global Hive Mind
        """
        with self._lock:
            self.total_audits += 1
            
            # Step 1: Scrub client-private information
            scrubbed_content = self._scrub_private_info(content, source_space_id)
            
            # Create new asset
            new_asset = KnowledgeAsset(
                asset_id="",  # Will be assigned by hive mind
                layer_type=layer_type,
                content=scrubbed_content,
                source_space_ids=[source_space_id],
                success_rate=success_rate,
                execution_count=execution_count,
                avg_latency_ms=avg_latency_ms,
                metadata=metadata or {}
            )
            
            # Step 2-4: Check for conflicts and resolve
            if is_structural_pattern and isinstance(content, dict):
                # Handle structural pattern
                pattern = StructuralPattern(
                    pattern_id=metadata.get('pattern_id', '') if metadata else '',
                    dfg_signature="",
                    nodes=content.get('nodes', []),
                    edges=[tuple(e) for e in content.get('edges', [])],
                    execution_profile=content.get('execution_profile', {})
                )
                
                # Find isomorphic patterns
                isomorphic = self.isomorphic_consolidator.find_isomorphic_patterns(
                    pattern, layer_type
                )
                
                if isomorphic:
                    # Consolidate
                    consolidated, was_merged = self.isomorphic_consolidator.consolidate(
                        pattern, isomorphic, source_space_id,
                        content.get('execution_profile', {})
                    )
                    
                    if was_merged:
                        self.total_merged += 1
                        result = AuditResult(
                            accepted=True,
                            action_taken='merged',
                            confidence_score=consolidated.compute_confidence_score()
                        )
                        self.audit_log.append(result)
                        self.total_accepted += 1
                        return result
            
            # Check for contradictions with existing assets
            layer = self.hive_mind.get_layer(layer_type)
            conflict_found = self._check_for_conflicts(new_asset, layer)
            
            if conflict_found:
                existing_asset = conflict_found
                conflict_type = self._determine_conflict_type(
                    new_asset, existing_asset
                )
                
                result = self.conflict_resolver.resolve_conflict(
                    new_asset, existing_asset, conflict_type
                )
                
                # Track metrics
                if result.action_taken == 'rejected':
                    self.total_rejected += 1
                elif result.action_taken == 'merged':
                    self.total_merged += 1
                    self.total_accepted += 1
                elif result.action_taken == 'pending_review':
                    self.pending_reviews.append(result)
                elif result.deprecated_asset_id:
                    self.total_deprecated += 1
                    self.total_accepted += 1
                
                self.audit_log.append(result)
                return result
            
            # Step 5: No conflicts - accept directly
            asset_id = self.hive_mingest_knowledge(
                scrubbed_content, layer_type, source_space_id,
                success_rate, execution_count, avg_latency_ms, metadata
            )
            
            result = AuditResult(
                accepted=True,
                action_taken='accepted',
                confidence_score=new_asset.compute_confidence_score()
            )
            
            self.audit_log.append(result)
            self.total_accepted += 1
            
            return result
    
    def _scrub_private_info(self, content: Any, source_space_id: str) -> Any:
        """Remove client-specific private information from content"""
        if isinstance(content, dict):
            scrubbed = {}
            for key, value in content.items():
                # Skip keys that might contain private info
                if any(skip in key.lower() for skip in ['client_', 'private_', 'secret_', 'token']):
                    continue
                scrubbed[key] = self._scrub_private_info(value, source_space_id)
            return scrubbed
        elif isinstance(content, list):
            return [self._scrub_private_info(item, source_space_id) for item in content]
        elif isinstance(content, str):
            # Replace space-specific identifiers with generic placeholders
            if source_space_id in content:
                return content.replace(source_space_id, "[SPACE]")
            return content
        else:
            return content
    
    def _check_for_conflicts(
        self,
        new_asset: KnowledgeAsset,
        layer: HiveMindLayer
    ) -> Optional[KnowledgeAsset]:
        """Check if new asset conflicts with existing assets"""
        # Simple heuristic: check assets with similar content hashes
        content_hash = hashlib.sha256(
            json.dumps(new_asset.content, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        for asset in layer.get_active_assets():
            asset_hash = hashlib.sha256(
                json.dumps(asset.content, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            
            if content_hash == asset_hash:
                return asset
        
        return None
    
    def _determine_conflict_type(
        self,
        new_asset: KnowledgeAsset,
        existing_asset: KnowledgeAsset
    ) -> ConflictType:
        """Determine the type of conflict between assets"""
        # Check for exact duplicate
        if new_asset.content == existing_asset.content:
            return ConflictType.DUPLICATE
        
        # Check for structural isomorphism
        if isinstance(new_asset.content, dict) and isinstance(existing_asset.content, dict):
            if 'nodes' in new_asset.content and 'nodes' in existing_asset.content:
                return ConflictType.ISOMORPHIC
        
        # Check for contradiction (same domain, different values)
        if isinstance(new_asset.content, dict) and isinstance(existing_asset.content, dict):
            common_keys = set(new_asset.content.keys()) & set(existing_asset.content.keys())
            for key in common_keys:
                if new_asset.content[key] != existing_asset.content[key]:
                    return ConflictType.CONTRADICTION
        
        # Default to outdated assumption
        return ConflictType.OUTDATED
    
    def hive_mingest_knowledge(
        self,
        content: Any,
        layer_type: HiveMindLayerType,
        source_space_id: str,
        success_rate: float,
        execution_count: int,
        avg_latency_ms: float,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Delegate to hive mind for ingestion"""
        return self.hive_mind.ingest_knowledge(
            content, layer_type, source_space_id,
            success_rate, execution_count, avg_latency_ms, metadata
        )
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """Get auditor statistics"""
        with self._lock:
            return {
                'total_audits': self.total_audits,
                'total_accepted': self.total_accepted,
                'total_rejected': self.total_rejected,
                'total_merged': self.total_merged,
                'total_deprecated': self.total_deprecated,
                'acceptance_rate': (
                    self.total_accepted / self.total_audits
                    if self.total_audits > 0 else 0
                ),
                'pending_reviews': len(self.pending_reviews),
                'isomorphic_stats': self.isomorphic_consolidator.get_isomorphism_stats()
            }
    
    def process_pending_reviews(self) -> List[AuditResult]:
        """Process pending manual reviews (placeholder for human-in-loop)"""
        # In production, this would interface with human reviewers
        # For now, apply default resolution after timeout
        processed = []
        
        for review in self.pending_reviews:
            # Auto-accept after "review period"
            review.action_taken = 'accepted'
            processed.append(review)
        
        self.pending_reviews.clear()
        return processed
