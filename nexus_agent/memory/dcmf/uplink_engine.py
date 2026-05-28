"""
Memory Uplink Engine

Background compiler that watches local .ecl files for Delta Event Triggers.
Extracts structural patterns, workflow updates, and discovered facts,
scrubs client data, and queues abstracted knowledge for the Global Hive Mind.

Features:
- Continuous monitoring of local .ecl ledgers
- Delta event detection and extraction
- Client data scrubbing and abstraction
- Priority-based queuing for uplink processing
"""

import hashlib
import threading
import time
import queue
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
import re

from .ecl_format import ECLLedger, ECLEntry, ECLPrimitiveType, ECLParser
from .hive_mind import GlobalHiveMind, StructuralPattern, HiveMindLayerType
from .memory_auditor import MemoryAuditor, AuditResult


class DeltaEventType(Enum):
    """Types of delta events extracted from .ecl ledgers"""
    STRUCTURAL_PATTERN = "structural_pattern"
    WORKFLOW_UPDATE = "workflow_update"
    DISCOVERED_FACT = "discovered_fact"
    ERROR_RESOLUTION = "error_resolution"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SCHEMA_CHANGE = "schema_change"


@dataclass
class DeltaEvent:
    """Represents a delta event extracted from local .ecl"""
    event_id: str
    event_type: DeltaEventType
    source_space_id: str
    source_ledger_path: str
    timestamp: datetime
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Extraction metrics
    confidence: float = 1.0
    execution_count: int = 1
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    
    # Processing state
    is_processed: bool = False
    is_scrubbed: bool = False
    priority: int = 5  # 1=highest, 10=lowest
    
    def compute_priority(self) -> int:
        """Compute priority based on event characteristics"""
        priority = 5  # Default
        
        # High success rate = higher priority
        if self.success_rate >= 0.95:
            priority -= 2
        elif self.success_rate >= 0.8:
            priority -= 1
        
        # High execution count = higher priority (proven reliability)
        if self.execution_count >= 10:
            priority -= 2
        elif self.execution_count >= 5:
            priority -= 1
        
        # Error resolutions are high priority
        if self.event_type == DeltaEventType.ERROR_RESOLUTION:
            priority -= 2
        
        return max(1, min(10, priority))


@dataclass
class DeltaEventTrigger:
    """Trigger condition for delta event extraction"""
    trigger_id: str
    trigger_type: str  # 'entry_added', 'section_modified', 'threshold_reached'
    pattern: str  # Regex or keyword pattern to match
    threshold_value: Optional[float] = None
    is_active: bool = True


class DeltaEventExtractor:
    """
    Extracts delta events from local .ecl ledgers.
    
    Identifies three critical features:
    - Structural Patterns: Recurrent sequences yielding high efficiency
    - Workflow Updates: AST/code modifications bypassing persistent bugs
    - Discovered Facts: Concrete constraints and API limitations
    """
    
    def __init__(self):
        self.triggers: List[DeltaEventTrigger] = self._default_triggers()
        self._lock = threading.RLock()
    
    def _default_triggers(self) -> List[DeltaEventTrigger]:
        """Set up default extraction triggers"""
        return [
            # Structural pattern triggers
            DeltaEventTrigger(
                trigger_id="trig_struct_001",
                trigger_type="entry_added",
                pattern=r"@OP_\d+\(.+\)\s*->\s*\$NODE_REF_\d+\s*->\s*%EXEC_PROFILE_FAST"
            ),
            
            # Error resolution triggers
            DeltaEventTrigger(
                trigger_id="trig_error_001",
                trigger_type="entry_added",
                pattern=r"FIXED_BY_AST_REWRITE|ERROR_RESOLVED|BUG_BYPASSED"
            ),
            
            # Performance optimization triggers
            DeltaEventTrigger(
                trigger_id="trig_perf_001",
                trigger_type="entry_added",
                pattern=r"OPTIMIZED|PERFORMANCE_BOOST|LATENCY_REDUCED"
            ),
            
            # Schema change triggers
            DeltaEventTrigger(
                trigger_id="trig_schema_001",
                trigger_type="section_modified",
                pattern=r"!SCHEMAS"
            ),
            
            # Discovered fact triggers (API limits, constraints)
            DeltaEventTrigger(
                trigger_id="trig_fact_001",
                trigger_type="entry_added",
                pattern=r"RATE_LIMIT|CONSTRAINT|REQUIREMENT|LIMITATION"
            )
        ]
    
    def extract_from_ledger(self, ledger: ECLLedger) -> List[DeltaEvent]:
        """Extract all delta events from a ledger"""
        events = []
        
        with self._lock:
            # Process episodic delta log
            if 'EPISODIC_DELTA_LOG' in ledger.sections:
                section = ledger.sections['EPISODIC_DELTA_LOG']
                
                for entry in section.entries:
                    event = self._analyze_entry(entry, ledger, DeltaEventType.DISCOVERED_FACT)
                    if event:
                        events.append(event)
            
            # Process procedural vectors for structural patterns
            if 'PROCEDURAL_VECTORS' in ledger.sections:
                section = ledger.sections['PROCEDURAL_VECTORS']
                
                for entry in section.entries:
                    if entry.primitive_type == ECLPrimitiveType.OPERATION:
                        exec_profile = entry.metadata.get('exec_profile', '')
                        
                        # Check for fast/optimized profiles
                        if 'FAST' in exec_profile or 'OPTIMIZED' in exec_profile:
                            event = self._analyze_entry(
                                entry, ledger, DeltaEventType.STRUCTURAL_PATTERN
                            )
                            if event:
                                events.append(event)
            
            # Process semantic root for schema changes
            if 'SEMANTIC_ROOT::STATE' in ledger.sections:
                section = ledger.sections['SEMANTIC_ROOT::STATE']
                
                for entry in section.entries:
                    if entry.primitive_type == ECLPrimitiveType.SCHEMA_PTR:
                        event = self._analyze_entry(
                            entry, ledger, DeltaEventType.SCHEMA_CHANGE
                        )
                        if event:
                            events.append(event)
        
        return events
    
    def _analyze_entry(
        self,
        entry: ECLEntry,
        ledger: ECLLedger,
        default_type: DeltaEventType
    ) -> Optional[DeltaEvent]:
        """Analyze single entry for delta event"""
        event_type = default_type
        
        # Determine event type from content
        content_str = entry.serialize_compact()
        
        if any(kw in content_str for kw in ['FIXED_BY', 'RESOLVED', 'BYPASSED']):
            event_type = DeltaEventType.ERROR_RESOLUTION
        elif any(kw in content_str for kw in ['OPTIMIZED', 'FAST', 'BOOST']):
            event_type = DeltaEventType.PERFORMANCE_OPTIMIZATION
        elif any(kw in content_str for kw in ['RATE_LIMIT', 'CONSTRAINT', 'LIMIT']):
            event_type = DeltaEventType.DISCOVERED_FACT
        elif entry.primitive_type == ECLPrimitiveType.OPERATION:
            event_type = DeltaEventType.STRUCTURAL_PATTERN
        
        # Extract metrics from metadata
        execution_count = entry.metadata.get('execution_count', 1)
        success_rate = entry.metadata.get('success_rate', 1.0)
        avg_latency = entry.metadata.get('avg_latency_ms', 0.0)
        
        # Compute confidence
        confidence = success_rate * min(1.0, execution_count / 10)
        
        # Generate event ID
        event_id = hashlib.sha256(
            f"{ledger.space_id}_{entry.key}_{entry.timestamp}".encode()
        ).hexdigest()[:12]
        
        return DeltaEvent(
            event_id=event_id,
            event_type=event_type,
            source_space_id=ledger.space_id,
            source_ledger_path="",  # Not applicable for in-memory ledgers
            timestamp=entry.timestamp,
            content=self._extract_content(entry, event_type),
            metadata={
                'entry_type': entry.primitive_type.value,
                'original_entry': entry.serialize_compact()
            },
            confidence=confidence,
            execution_count=execution_count,
            success_rate=success_rate,
            avg_latency_ms=avg_latency
        )
    
    def _extract_content(
        self,
        entry: ECLEntry,
        event_type: DeltaEventType
    ) -> Any:
        """Extract structured content from entry based on event type"""
        if event_type == DeltaEventType.STRUCTURAL_PATTERN:
            # Build structural pattern representation
            return {
                'operation': entry.metadata.get('op_name', 'unknown'),
                'node_ref': entry.metadata.get('node_ref', ''),
                'exec_profile': entry.metadata.get('exec_profile', ''),
                'pattern_type': 'operation_sequence'
            }
        elif event_type == DeltaEventType.ERROR_RESOLUTION:
            return {
                'resolution_method': entry.value,
                'context': entry.metadata.get('request_id', '')
            }
        elif event_type == DeltaEventType.DISCOVERED_FACT:
            return {
                'fact': entry.value,
                'domain': entry.metadata.get('domain', 'general')
            }
        else:
            return entry.value


class MemoryUplinkEngine:
    """
    Background engine that processes delta events and uplinks to Hive Mind.
    
    Continuously monitors local .ecl files, extracts delta events,
    scrubs client data, and queues knowledge for global injection.
    """
    
    def __init__(
        self,
        hive_mind: GlobalHiveMind,
        auditor: MemoryAuditor,
        polling_interval: float = 30.0
    ):
        self.hive_mind = hive_mind
        self.auditor = auditor
        self.extractor = DeltaEventExtractor()
        
        self.polling_interval = polling_interval
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Tracking
        self.monitored_spaces: Set[str] = set()
        self.ledger_checksums: Dict[str, str] = {}
        self.processed_events: List[str] = []
        
        # Metrics
        self.total_events_extracted = 0
        self.total_events_uplinked = 0
        self.total_events_rejected = 0
    
    def start(self) -> None:
        """Start background monitoring loop"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._worker_thread.start()
    
    def stop(self) -> None:
        """Stop background monitoring"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
    
    def register_space(self, space_id: str, ledger_path: str) -> None:
        """Register a space's .ecl ledger for monitoring"""
        with self._lock:
            self.monitored_spaces.add(space_id)
            
            # Compute initial checksum
            if os.path.exists(ledger_path):
                with open(ledger_path, 'r') as f:
                    content = f.read()
                self.ledger_checksums[space_id] = hashlib.sha256(
                    content.encode()
                ).hexdigest()
    
    def _monitoring_loop(self) -> None:
        """Background loop for monitoring and processing"""
        while self._running:
            try:
                # Check all monitored spaces for changes
                for space_id in list(self.monitored_spaces):
                    self._check_for_deltas(space_id)
                
                # Process queued events
                self._process_queued_events()
                
            except Exception as e:
                pass  # Log error in production
            
            time.sleep(self.polling_interval)
    
    def _check_for_deltas(self, space_id: str) -> None:
        """Check space's ledger for new delta events"""
        # In production, this would read from file system
        # For now, we simulate with in-memory tracking
        
        # Would compare checksums and extract new events
        # Implementation depends on storage backend
        pass
    
    def process_ledger_directly(self, ledger: ECLLedger) -> List[DeltaEvent]:
        """
        Process a ledger directly and queue its delta events.
        
        Used for immediate processing rather than waiting for poll cycle.
        """
        events = self.extractor.extract_from_ledger(ledger)
        
        for event in events:
            event.priority = event.compute_priority()
            
            # Scrub private information
            event.content = self._scrub_content(event.content, event.source_space_id)
            event.is_scrubbed = True
            
            # Add to queue
            self.event_queue.put((event.priority, event.event_id, event))
            self.total_events_extracted += 1
        
        return events
    
    def _process_queued_events(self) -> None:
        """Process events from queue"""
        batch_size = 10
        processed = 0
        
        while not self.event_queue.empty() and processed < batch_size:
            try:
                priority, event_id, event = self.event_queue.get_nowait()
                
                if event.event_id in self.processed_events:
                    continue
                
                # Audit and uplink
                result = self._uplink_event(event)
                
                if result.accepted:
                    self.processed_events.append(event.event_id)
                    self.total_events_uplinked += 1
                else:
                    self.total_events_rejected += 1
                
                processed += 1
                
            except queue.Empty:
                break
    
    def _uplink_event(self, event: DeltaEvent) -> AuditResult:
        """Audit and uplink single event to Hive Mind"""
        # Determine layer type
        layer_type = HiveMindLayerType.PROCEDURAL
        if event.event_type == DeltaEventType.DISCOVERED_FACT:
            layer_type = HiveMindLayerType.SEMANTIC
        elif event.event_type == DeltaEventType.SCHEMA_CHANGE:
            layer_type = HiveMindLayerType.SEMANTIC
        
        # Determine if structural pattern
        is_structural = event.event_type == DeltaEventType.STRUCTURAL_PATTERN
        
        # Prepare metadata
        metadata = {
            'event_type': event.event_type.value,
            'source_space': event.source_space_id,
            'extracted_at': event.timestamp.isoformat(),
            'is_scrubbed': event.is_scrubbed
        }
        
        if is_structural and isinstance(event.content, dict):
            metadata['pattern_id'] = event.event_id
        
        # Submit for audit
        return self.auditor.audit_uplink(
            content=event.content,
            layer_type=layer_type,
            source_space_id=event.source_space_id,
            success_rate=event.success_rate,
            execution_count=event.execution_count,
            avg_latency_ms=event.avg_latency_ms,
            metadata=metadata,
            is_structural_pattern=is_structural
        )
    
    def _scrub_content(self, content: Any, source_space_id: str) -> Any:
        """Remove client-specific information from content"""
        if isinstance(content, dict):
            scrubbed = {}
            for key, value in content.items():
                if any(skip in key.lower() for skip in ['client_', 'private_', 'secret_', 'token', 'api_key']):
                    continue
                scrubbed[key] = self._scrub_content(value, source_space_id)
            return scrubbed
        elif isinstance(content, list):
            return [self._scrub_content(item, source_space_id) for item in content]
        elif isinstance(content, str):
            # Replace space-specific identifiers
            scrubbed = content.replace(source_space_id, "[SPACE]")
            # Remove potential PII patterns
            scrubbed = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', scrubbed)
            return scrubbed
        else:
            return content
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self._lock:
            return {
                'is_running': self._running,
                'monitored_spaces': len(self.monitored_spaces),
                'queue_size': self.event_queue.qsize(),
                'processed_events': len(self.processed_events),
                'total_extracted': self.total_events_extracted,
                'total_uplinked': self.total_events_uplinked,
                'total_rejected': self.total_events_rejected,
                'uplink_success_rate': (
                    self.total_events_uplinked / self.total_events_extracted
                    if self.total_events_extracted > 0 else 0
                )
            }
    
    def force_process_all(self) -> int:
        """Force process all queued events immediately"""
        count = 0
        while not self.event_queue.empty():
            self._process_queued_events()
            count += 1
        return count
