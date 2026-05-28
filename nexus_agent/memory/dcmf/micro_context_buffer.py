"""
Micro-Context Buffer (MCB)

Ephemeral, task-specific memory slices that are dynamically provisioned
when a Manager breaks down a project into atomic sub-tasks.

Features:
- Dynamic provisioning on sub-task creation
- Intelligent hydration from parent .ecl ledger
- Isolated execution as high-speed scratchpad
- Automatic dissolution after task completion with lesson synthesis
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime
from collections import OrderedDict
import threading

from .ecl_format import ECLLedger, ECLEntry, ECLPrimitiveType, ECLSection


@dataclass
class MCBEntry:
    """Single entry in a Micro-Context Buffer"""
    key: str
    value: Any
    entry_type: str  # 'variable', 'execution_attempt', 'telemetry', 'context_snippet'
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class TaskContext:
    """Context for a specific sub-task"""
    task_id: str
    task_description: str
    required_schemas: List[str] = field(default_factory=list)
    failure_profiles_to_watch: List[str] = field(default_factory=list)
    parent_space_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


class MicroContextBuffer:
    """
    Ephemeral memory slice for a single atomic sub-task.
    
    Acts as a high-speed scratchpad for workers, logging local variables,
    execution attempts, and runtime telemetry. Automatically dissolved
    after task completion with core lessons synthesized back to .ecl.
    """
    
    def __init__(
        self,
        task_context: TaskContext,
        hydration_source: Optional[ECLLedger] = None,
        max_entries: int = 1000,
        ttl_seconds: int = 3600  # 1 hour default TTL
    ):
        self.buffer_id = str(uuid.uuid4())[:8]
        self.task_context = task_context
        self.entries: OrderedDict[str, MCBEntry] = OrderedDict()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.created_at = datetime.utcnow()
        self.expires_at = datetime.utcnow()
        self.is_active = True
        self._lock = threading.RLock()
        
        # Telemetry tracking
        self.telemetry: Dict[str, Any] = {
            'creation_time': time.time(),
            'total_writes': 0,
            'total_reads': 0,
            'peak_memory_entries': 0,
            'execution_attempts': 0,
            'success': False,
        }
        
        # Hydrate from source if provided
        if hydration_source:
            self._hydrate_from_ledger(hydration_source)
    
    def _hydrate_from_ledger(self, ledger: ECLLedger) -> None:
        """
        Intelligently hydrate MCB with targeted snippets from parent .ecl ledger.
        
        Extracts only:
        - Required schemas specified in task context
        - Failure profile nodes to watch
        - Relevant domain keys
        """
        with self._lock:
            # Extract differential context from ledger
            differential = ledger.extract_differential_context(
                required_schemas=self.task_context.required_schemas,
                failure_profiles=self.task_context.failure_profiles_to_watch,
                strip_metadata=True
            )
            
            # Parse and add relevant entries
            if 'SEMANTIC_ROOT::STATE' in ledger.sections:
                section = ledger.sections['SEMANTIC_ROOT::STATE']
                
                # Add domain keys
                for entry in section.get_entries_by_type(ECLPrimitiveType.DOMAIN_KEY):
                    self._add_entry_internal(
                        key=f"domain_{entry.key}",
                        value=entry.value,
                        entry_type='context_snippet',
                        metadata={'source': 'semantic_root'}
                    )
                
                # Add filtered schemas
                for entry in section.get_entries_by_type(ECLPrimitiveType.SCHEMA_PTR):
                    filtered = {
                        k: v for k, v in entry.value.items()
                        if k in self.task_context.required_schemas
                    }
                    if filtered:
                        self._add_entry_internal(
                            key=f"schema_{entry.key}",
                            value=filtered,
                            entry_type='context_snippet',
                            metadata={'source': 'semantic_root'}
                        )
            
            # Add relevant procedural vectors
            if 'PROCEDURAL_VECTORS' in ledger.sections:
                section = ledger.sections['PROCEDURAL_VECTORS']
                
                for entry in section.entries:
                    if entry.primitive_type == ECLPrimitiveType.OPERATION:
                        exec_profile = entry.metadata.get('exec_profile', '')
                        
                        # Add operations with failure profiles we're watching
                        if any(fp in exec_profile for fp in self.task_context.failure_profiles_to_watch):
                            self._add_entry_internal(
                                key=f"op_{entry.key}_{entry.value}",
                                value=entry.metadata,
                                entry_type='context_snippet',
                                metadata={
                                    'source': 'procedural_vectors',
                                    'watch_for_failure': True
                                }
                            )
    
    def _add_entry_internal(
        self,
        key: str,
        value: Any,
        entry_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MCBEntry:
        """Internal method to add entry without lock (caller must hold lock)"""
        # Evict oldest if at capacity
        while len(self.entries) >= self.max_entries:
            oldest_key = next(iter(self.entries))
            del self.entries[oldest_key]
        
        entry = MCBEntry(
            key=key,
            value=value,
            entry_type=entry_type,
            metadata=metadata or {}
        )
        self.entries[key] = entry
        
        self.telemetry['total_writes'] += 1
        self.telemetry['peak_memory_entries'] = max(
            self.telemetry['peak_memory_entries'],
            len(self.entries)
        )
        
        return entry
    
    def set(self, key: str, value: Any, entry_type: str = 'variable', 
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set a value in the buffer"""
        with self._lock:
            if not self.is_active:
                raise RuntimeError("MCB has been dissolved and is no longer active")
            
            if key in self.entries:
                # Update existing
                entry = self.entries[key]
                entry.value = value
                entry.entry_type = entry_type
                if metadata:
                    entry.metadata.update(metadata)
                entry.touch()
            else:
                # Add new
                self._add_entry_internal(key, value, entry_type, metadata)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the buffer"""
        with self._lock:
            if not self.is_active:
                raise RuntimeError("MCB has been dissolved and is no longer active")
            
            if key in self.entries:
                entry = self.entries[key]
                entry.touch()
                self.telemetry['total_reads'] += 1
                return entry.value
            
            return default
    
    def delete(self, key: str) -> bool:
        """Delete a key from the buffer"""
        with self._lock:
            if key in self.entries:
                del self.entries[key]
                return True
            return False
    
    def log_execution_attempt(
        self,
        attempt_id: str,
        success: bool,
        error: Optional[str] = None,
        duration_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an execution attempt with telemetry"""
        with self._lock:
            self.telemetry['execution_attempts'] += 1
            
            self._add_entry_internal(
                key=f"exec_{attempt_id}",
                value={
                    'success': success,
                    'error': error,
                    'duration_ms': duration_ms,
                    'timestamp': datetime.utcnow().isoformat()
                },
                entry_type='execution_attempt',
                metadata=metadata or {}
            )
    
    def log_telemetry(self, metric_name: str, value: Any) -> None:
        """Log runtime telemetry"""
        with self._lock:
            self.telemetry[metric_name] = value
            
            self._add_entry_internal(
                key=f"telemetry_{metric_name}",
                value=value,
                entry_type='telemetry',
                metadata={'logged_at': datetime.utcnow().isoformat()}
            )
    
    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variable-type entries"""
        with self._lock:
            return {
                k: v.value for k, v in self.entries.items()
                if v.entry_type == 'variable'
            }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get all execution attempt records"""
        with self._lock:
            return [
                v.value for k, v in self.entries.items()
                if v.entry_type == 'execution_attempt'
            ]
    
    def get_lessons_learned(self) -> List[Dict[str, Any]]:
        """
        Synthesize lessons learned from execution history.
        
        Called before dissolution to extract core insights for .ecl integration.
        """
        with self._lock:
            lessons = []
            execution_history = self.get_execution_history()
            
            if not execution_history:
                return lessons
            
            # Analyze patterns
            successes = [e for e in execution_history if e.get('success')]
            failures = [e for e in execution_history if not e.get('success')]
            
            # Lesson: Final successful approach
            if successes:
                final_success = successes[-1]
                lessons.append({
                    'type': 'successful_pattern',
                    'description': 'Final successful execution pattern',
                    'data': final_success,
                    'confidence': 0.9 if len(successes) > 1 else 0.7
                })
            
            # Lesson: Error patterns to avoid
            error_patterns = {}
            for failure in failures:
                error = failure.get('error', 'unknown')
                if error not in error_patterns:
                    error_patterns[error] = 0
                error_patterns[error] += 1
            
            for error, count in error_patterns.items():
                lessons.append({
                    'type': 'error_pattern',
                    'description': f'Error pattern to avoid: {error}',
                    'occurrence_count': count,
                    'confidence': min(0.95, 0.5 + (count * 0.1))
                })
            
            # Lesson: Performance insights
            if execution_history:
                durations = [
                    e.get('duration_ms', 0) for e in execution_history
                    if e.get('duration_ms', 0) > 0
                ]
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    lessons.append({
                        'type': 'performance_insight',
                        'description': 'Average execution duration',
                        'avg_duration_ms': avg_duration,
                        'min_duration_ms': min(durations),
                        'max_duration_ms': max(durations)
                    })
            
            return lessons
    
    def dissolve(self) -> List[Dict[str, Any]]:
        """
        Dissolve the MCB, synthesizing lessons for .ecl integration.
        
        Returns list of lessons learned to be appended to parent .ecl ledger.
        """
        with self._lock:
            if not self.is_active:
                raise RuntimeError("MCB already dissolved")
            
            self.is_active = False
            lessons = self.get_lessons_learned()
            
            # Log dissolution
            self.telemetry['dissolution_time'] = time.time()
            self.telemetry['lifetime_seconds'] = (
                self.telemetry['dissolution_time'] - self.telemetry['creation_time']
            )
            
            return lessons
    
    def is_expired(self) -> bool:
        """Check if MCB has exceeded TTL"""
        return datetime.utcnow() > self.expires_at
    
    def extend_ttl(self, seconds: int) -> None:
        """Extend the TTL"""
        with self._lock:
            self.expires_at = datetime.utcnow()
            self.expires_at = self.expires_at.replace(
                second=self.expires_at.second + seconds % 60,
                minute=self.expires_at.minute + seconds // 60
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                'buffer_id': self.buffer_id,
                'task_id': self.task_context.task_id,
                'is_active': self.is_active,
                'entry_count': len(self.entries),
                'max_entries': self.max_entries,
                'utilization_percent': (len(self.entries) / self.max_entries) * 100,
                'telemetry': self.telemetry.copy(),
                'age_seconds': (datetime.utcnow() - self.created_at).total_seconds()
            }


class MCBManager:
    """
    Manager for Micro-Context Buffers.
    
    Handles lifecycle management, provisioning, and coordination
    of multiple MCBs across concurrent tasks.
    """
    
    def __init__(self, default_max_entries: int = 1000, default_ttl: int = 3600):
        self.buffers: Dict[str, MicroContextBuffer] = {}
        self.default_max_entries = default_max_entries
        self.default_ttl = default_ttl
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
    
    def create_buffer(
        self,
        task_context: TaskContext,
        hydration_source: Optional[ECLLedger] = None,
        max_entries: Optional[int] = None,
        ttl_seconds: Optional[int] = None
    ) -> MicroContextBuffer:
        """
        Create a new MCB for a sub-task.
        
        Dynamically provisions ephemeral memory slice with intelligent
        hydration from parent .ecl ledger.
        """
        with self._lock:
            buffer = MicroContextBuffer(
                task_context=task_context,
                hydration_source=hydration_source,
                max_entries=max_entries or self.default_max_entries,
                ttl_seconds=ttl_seconds or self.default_ttl
            )
            
            self.buffers[buffer.buffer_id] = buffer
            return buffer
    
    def get_buffer(self, buffer_id: str) -> Optional[MicroContextBuffer]:
        """Retrieve buffer by ID"""
        with self._lock:
            return self.buffers.get(buffer_id)
    
    def dissolve_buffer(self, buffer_id: str) -> List[Dict[str, Any]]:
        """
        Dissolve a buffer and return lessons learned.
        
        Core lessons are meant to be synthesized back to .ecl ledger.
        """
        with self._lock:
            if buffer_id not in self.buffers:
                raise ValueError(f"Buffer {buffer_id} not found")
            
            buffer = self.buffers[buffer_id]
            lessons = buffer.dissolve()
            
            del self.buffers[buffer_id]
            return lessons
    
    def cleanup_expired(self) -> List[str]:
        """Remove expired buffers, returning their IDs"""
        with self._lock:
            expired_ids = []
            
            for buffer_id, buffer in list(self.buffers.items()):
                if buffer.is_expired():
                    try:
                        buffer.dissolve()
                    except:
                        pass  # Already dissolved
                    expired_ids.append(buffer_id)
                    del self.buffers[buffer_id]
            
            return expired_ids
    
    def start_cleanup_loop(self, interval_seconds: int = 300) -> None:
        """Start background cleanup thread"""
        if self._running:
            return
        
        self._running = True
        
        def cleanup_loop():
            while self._running:
                try:
                    self.cleanup_expired()
                except Exception as e:
                    pass  # Log error in production
                time.sleep(interval_seconds)
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def stop_cleanup_loop(self) -> None:
        """Stop background cleanup thread"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def get_active_count(self) -> int:
        """Get count of active buffers"""
        with self._lock:
            return sum(1 for b in self.buffers.values() if b.is_active)
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get stats for all buffers"""
        with self._lock:
            return [b.get_stats() for b in self.buffers.values()]
    
    def integrate_lessons_to_ledger(
        self,
        lessons: List[Dict[str, Any]],
        ledger: ECLLedger,
        task_id: str
    ) -> None:
        """
        Integrate lessons learned back into parent .ecl ledger.
        
        Converts MCB lessons into proper ECL entries for episodic delta log.
        """
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        
        for lesson in lessons:
            lesson_type = lesson.get('type', 'unknown')
            description = lesson.get('description', '')
            
            # Create episodic delta entry
            delta_value = f"{lesson_type.upper()}:{description}"
            
            entry = ECLEntry(
                primitive_type=ECLPrimitiveType.TEMPORAL_MARKER,
                key=timestamp,
                value=delta_value,
                metadata={
                    'request_id': task_id,
                    'lesson_data': lesson,
                    'confidence': lesson.get('confidence', 0.5)
                }
            )
            
            ledger.add_entry('EPISODIC_DELTA_LOG', entry)
