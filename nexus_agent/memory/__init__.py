"""Memory Layer: Tiered memory system with consolidation and DCMF."""

from .tiers.hot_tier import HotTierMemory
from .tiers.warm_tier import WarmTierMemory
from .tiers.cold_tier import ColdTierMemory
from .consolidation.dream_engine import DreamEngine
from .self_organizing.merge_agent import MergeAgent

# Distributed Cognitive Memory Fabric (DCMF)
from .dcmf.ecl_format import ECLFormat, ECLLedger, ECLParser
from .dcmf.micro_context_buffer import MicroContextBuffer, MCBManager, TaskContext
from .dcmf.hive_mind import GlobalHiveMind, HiveMindLayer, HiveMindLayerType, KnowledgeAsset, StructuralPattern
from .dcmf.memory_auditor import MemoryAuditor, ConflictResolver, IsomorphicConsolidator, ConflictType, AuditResult
from .dcmf.uplink_engine import MemoryUplinkEngine, DeltaEventExtractor, DeltaEventType

__all__ = [
    'HotTierMemory',
    'WarmTierMemory', 
    'ColdTierMemory',
    'DreamEngine',
    'MergeAgent',
    # DCMF exports
    'ECLFormat',
    'ECLLedger',
    'ECLParser',
    'MicroContextBuffer',
    'MCBManager',
    'TaskContext',
    'GlobalHiveMind',
    'HiveMindLayer',
    'HiveMindLayerType',
    'KnowledgeAsset',
    'StructuralPattern',
    'MemoryAuditor',
    'ConflictResolver',
    'IsomorphicConsolidator',
    'ConflictType',
    'AuditResult',
    'MemoryUplinkEngine',
    'DeltaEventExtractor',
    'DeltaEventType'
]
