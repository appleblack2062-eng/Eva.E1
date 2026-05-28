"""Telemetry Recorder for Polyglot DAG Execution."""

from __future__ import annotations
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

# Direct imports to avoid relative import issues
try:
    from orchestration.polyglot.models import NodeTelemetry, ExecutionTrace
except ImportError:
    try:
        import sys
        from pathlib import Path as PPath
        sys.path.insert(0, str(PPath(__file__).parent.parent))
        from orchestration.polyglot.models import NodeTelemetry, ExecutionTrace
    except:
        # Fallback: define minimal classes
        class NodeTelemetry:
            @classmethod
            def from_execution_logs(cls, node_id, logs):
                return cls(node_id=node_id, p50_ms=0, p95_ms=0, variance_ms=0, 
                          error_rate=0, determinism_score=1, tokens_used=0,
                          cost_usd=0, cpu_peak_mb=0)
        
        class ExecutionTrace:
            pass


class TelemetryRecorder:
    """
    Records and retrieves execution telemetry for DAG nodes.
    
    Stores per-node metrics including latency, error rates, determinism scores,
    and resource usage in SQLite for efficient querying.
    """
    
    def __init__(self, storage_path: str = "./nexus_data/telemetry.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        # Node execution logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS node_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                dag_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                success INTEGER NOT NULL,
                latency_ms REAL,
                output_hash TEXT,
                error_message TEXT,
                tokens_used INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                cpu_mb REAL DEFAULT 0,
                input_hash TEXT,
                runtime_type TEXT
            )
        """)
        
        # Execution traces table for memory-driven optimization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_traces (
                trace_id TEXT PRIMARY KEY,
                dag_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT,
                telemetry_json TEXT,
                error_trace TEXT,
                timestamp REAL NOT NULL
            )
        """)
        
        # Index for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_node_id 
            ON node_executions(node_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dag_id 
            ON node_executions(dag_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_input_hash 
            ON execution_traces(input_hash)
        """)
        
        conn.commit()
        conn.close()
    
    def record_execution(
        self,
        node_id: str,
        dag_id: str,
        success: bool,
        latency_ms: float,
        output_hash: Optional[str] = None,
        error_message: Optional[str] = None,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        cpu_mb: float = 0.0,
        input_hash: Optional[str] = None,
        runtime_type: Optional[str] = None
    ):
        """Record a single node execution."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO node_executions (
                node_id, dag_id, timestamp, success, latency_ms,
                output_hash, error_message, tokens_used, cost_usd,
                cpu_mb, input_hash, runtime_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node_id, dag_id, time.time(),
            1 if success else 0, latency_ms,
            output_hash, error_message, tokens_used, cost_usd,
            cpu_mb, input_hash, runtime_type
        ))
        
        conn.commit()
        conn.close()
    
    def record_trace(self, trace: ExecutionTrace):
        """Record a complete execution trace."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO execution_traces (
                trace_id, dag_id, node_id, node_type, input_hash,
                output_hash, telemetry_json, error_trace, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.trace_id, trace.dag_id, trace.node_id,
            trace.node_type, trace.input_hash, trace.output_hash,
            json.dumps(trace.telemetry), trace.error_trace, trace.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_telemetry(
        self, 
        node_id: str, 
        window: int = 100
    ) -> NodeTelemetry:
        """
        Get aggregated telemetry for a node from recent executions.
        
        Args:
            node_id: Node identifier
            window: Number of recent executions to consider
            
        Returns:
            Aggregated NodeTelemetry
        """
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT success, latency_ms, tokens_used, cost_usd, cpu_mb, output_hash
            FROM node_executions
            WHERE node_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (node_id, window))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return NodeTelemetry(
                node_id=node_id,
                p50_ms=0.0,
                p95_ms=0.0,
                variance_ms=0.0,
                error_rate=0.0,
                determinism_score=1.0,
                tokens_used=0,
                cost_usd=0.0,
                cpu_peak_mb=0.0
            )
        
        logs = []
        for row in rows:
            logs.append({
                'success': bool(row[0]),
                'latency_ms': row[1],
                'tokens_used': row[2],
                'cost_usd': row[3],
                'cpu_mb': row[4],
                'output_hash': row[5]
            })
        
        return NodeTelemetry.from_execution_logs(node_id, logs)
    
    def get_dag_stats(self, dag_id: str, window: int = 100) -> Dict[str, Any]:
        """Get aggregated statistics for an entire DAG."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency,
                SUM(cost_usd) as total_cost,
                COUNT(*) as total_runs,
                AVG(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_rate
            FROM node_executions
            WHERE dag_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (dag_id, window))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or not row[0]:
            return {
                "avg_latency": 0.0,
                "max_latency": 0.0,
                "total_p95": 0.0,
                "total_cost": 0.0,
                "success_rate": 0.0
            }
        
        return {
            "avg_latency": row[0] or 0.0,
            "max_latency": row[1] or 0.0,
            "total_p95": row[1] or 0.0,  # Approximate p95 as max
            "total_cost": row[2] or 0.0,
            "success_rate": row[4] or 0.0
        }
    
    def query_similar_traces(
        self, 
        input_hash: str, 
        limit: int = 10
    ) -> List[ExecutionTrace]:
        """Find historical traces with similar inputs."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT trace_id, dag_id, node_id, node_type, input_hash,
                   output_hash, telemetry_json, error_trace, timestamp
            FROM execution_traces
            WHERE input_hash = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (input_hash, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        traces = []
        for row in rows:
            traces.append(ExecutionTrace(
                trace_id=row[0],
                dag_id=row[1],
                node_id=row[2],
                node_type=row[3],
                input_hash=row[4],
                output_hash=row[5],
                telemetry=json.loads(row[6]) if row[6] else {},
                error_trace=row[7],
                timestamp=row[8]
            ))
        
        return traces
    
    def get_node_error_patterns(
        self, 
        node_id: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent error patterns for a node."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT error_message, timestamp, runtime_type
            FROM node_executions
            WHERE node_id = ? AND success = 0
            ORDER BY timestamp DESC
            LIMIT ?
        """, (node_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            patterns.append({
                'error_message': row[0],
                'timestamp': row[1],
                'runtime_type': row[2]
            })
        
        return patterns
    
    def clear_old_data(self, max_age_days: int = 30):
        """Clear telemetry data older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM node_executions WHERE timestamp < ?
        """, (cutoff_time,))
        
        cursor.execute("""
            DELETE FROM execution_traces WHERE timestamp < ?
        """, (cutoff_time,))
        
        deleted_nodes = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_nodes


class InstrumentedNodeExecutor:
    """Wrapper that adds telemetry recording to node execution."""
    
    def __init__(self, recorder: TelemetryRecorder):
        self.recorder = recorder
    
    async def execute_with_telemetry(
        self, 
        node: Any, 
        executor_func: callable
    ) -> Dict[str, Any]:
        """Execute a node and record telemetry."""
        import hashlib
        import json
        
        start_time = time.time()
        input_hash = None
        output_hash = None
        tokens_used = 0
        cost_usd = 0.0
        cpu_mb = 0.0
        
        try:
            # Compute input hash
            # Note: We'll get actual inputs during execution
            result = await executor_func()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract metadata from result if available
            if isinstance(result, dict):
                output_hash = result.get('_output_hash')
                tokens_used = result.get('tokens_used', 0)
                cost_usd = result.get('cost_usd', 0.0)
                cpu_mb = result.get('cpu_mb', 0.0)
            
            # Record successful execution
            self.recorder.record_execution(
                node_id=node.node_id,
                dag_id=getattr(node, 'dag_id', 'unknown'),
                success=True,
                latency_ms=latency_ms,
                output_hash=output_hash,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                cpu_mb=cpu_mb,
                input_hash=input_hash,
                runtime_type=node.runtime
            )
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Record failed execution
            self.recorder.record_execution(
                node_id=node.node_id,
                dag_id=getattr(node, 'dag_id', 'unknown'),
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
                input_hash=input_hash,
                runtime_type=node.runtime
            )
            
            raise
