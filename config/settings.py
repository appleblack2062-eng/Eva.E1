"""Configuration settings for the agent."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    
    # Identity
    agent_id: str = "default_agent"
    
    # Task execution limits
    max_task_timeout: float = 60.0
    max_llm_tokens_per_task: int = 2000
    llm_temperature: float = 0.1
    max_workflow_execution_time_seconds: float = 30.0
    
    # Allowed operations for workflows
    allowed_operations: List[str] = field(default_factory=lambda: [
        "FILTER", "TRANSFORM", "MAP", "REDUCE", "RETURN",
        "VALIDATE", "PARSE", "FORMAT", "AGGREGATE", "SORT",
        "SEARCH", "HTTP_GET", "HTTP_POST", "PARSE_JSON", "FORMAT_STRING"
    ])
    
    # Forbidden operations (security)
    forbidden_operations: List[str] = field(default_factory=lambda: [
        "EXEC_OS", "DELETE_FILE", "FORMAT_DISK", "EXEC", "EVAL", "SYSTEM", "IMPORT", "OPEN"
    ])
    
    # Validation settings
    max_validation_tests: int = 10
    validation_timeout: float = 10.0
    min_test_pass_rate: float = 0.9
    min_validation_confidence: float = 0.85
    min_test_pass_rate_for_deployment: float = 0.95
    
    # Memory settings
    max_memory_usage_mb: int = 512
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Base storage path
    base_storage_path: str = "./nexus_data"
    
    # Optimization thresholds
    min_task_repetitions_for_synthesis: int = 3
    
    # Operation to tool mapping
    operation_to_tool_map: Dict[str, str] = field(default_factory=lambda: {
        "FILTER": "filter_data",
        "TRANSFORM": "transform_data",
        "MAP": "transform_data",
        "REDUCE": "reduce_tool",
        "SORT": "sort_data",
        "VALIDATE": "validate_tool",
        "PARSE": "parse_tool",
        "FORMAT": "format_tool",
        "HTTP_GET": "http_get",
        "PARSE_JSON": "parse_json",
    })


@dataclass
class GlobalConfig:
    """Global configuration shared across agents."""
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Storage paths
    storage_path: str = "./nexus_data"
    cache_path: str = "./nexus_cache"
    
    # Performance settings
    enable_caching: bool = True
    enable_profiling: bool = True
    
    # Safety settings
    sandbox_enabled: bool = True
    resource_monitoring_enabled: bool = True
    
    # Debug mode
    debug_mode: bool = False
