"""
Distributed Cognitive Memory Fabric (DCMF)

A hyper-efficient multi-tenant memory architecture featuring:
- Encapsulated Context Ledger (.ECL) format for token-to-entropy density
- Micro-Context Buffers (MCB) for ephemeral task-specific memory
- Global Hive Mind for cross-space knowledge sharing
- Memory Auditor Agent for self-correction and consolidation
"""

from .ecl_format import ECLFormat, ECLLedger, ECLParser
from .micro_context_buffer import MicroContextBuffer, MCBManager
from .hive_mind import GlobalHiveMind, HiveMindLayer
from .memory_auditor import MemoryAuditor, ConflictResolver, IsomorphicConsolidator
from .uplink_engine import MemoryUplinkEngine, DeltaEventTrigger

__all__ = [
    'ECLFormat',
    'ECLLedger',
    'ECLParser',
    'MicroContextBuffer',
    'MCBManager',
    'GlobalHiveMind',
    'HiveMindLayer',
    'MemoryAuditor',
    'ConflictResolver',
    'IsomorphicConsolidator',
    'MemoryUplinkEngine',
    'DeltaEventTrigger',
]
