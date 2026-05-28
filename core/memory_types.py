"""Core execution types and memory structures."""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid


class ExecutionMode(Enum):
    """How a task should be executed."""
    LLM_ONLY = auto()              # Pure LLM inference
    LLM_GUIDED = auto()            # LLM plans, tools execute
    WORKFLOW_DRAFT = auto()        # Synthesized workflow, interpreted
    WORKFLOW_COMPILED = auto()     # Optimized Python, cached
    WORKFLOW_JIT = auto()          # JIT-compiled hot path
    HYBRID_FALLBACK = auto()       # Workflow + LLM fallback on error


@dataclass
class TaskPattern:
    """Abstract representation of a task pattern."""
    intent: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    complexity: str  # simple, moderate, complex
    domain: str


@dataclass
class ToolSpec:
    """Specification for a tool."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    code: Optional[str] = None  # Code implementation
    implementation: Optional[str] = None  # Code or reference
    is_builtin: bool = False
    performance_gain: float = 0.0


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    step_number: int
    sub_step: str = ""
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    indent_level: int = 0
    is_terminal: bool = False
    return_value: Optional[Any] = None


@dataclass
class WorkflowSpec:
    """Complete workflow specification."""
    id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    steps: List[WorkflowStep] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    error_handling: str = "fallback"
    
    # Optional compiled code
    compiled_code: Optional[str] = None
    
    # Metadata
    version: str = "1.0.0"
    complexity: str = "moderate"
    optimization_hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    has_custom_operations: bool = False
    profile: Dict[str, Any] = field(default_factory=dict)
    estimated_speedup: float = 1.0
    estimated_token_reduction: float = 0.0
    estimated_latency_reduction: float = 0.0


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.LLM_ONLY
    latency_ms: float = 0.0
    tokens_used: int = 0
    fallback_triggered: bool = False


# Eva.E1 Meta-Learning Types

@dataclass
class ComponentSpec:
    """Specification for a reusable atomic component."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    code: str = ""
    dependencies: List[str] = field(default_factory=list)
    success_rate: float = 0.9
    avg_latency_ms: float = 50.0
    avg_cost_usd: float = 0.001
    tags: List[str] = field(default_factory=list)


@dataclass
class FeedbackSignal:
    """Signal extracted from user feedback."""
    task_context: Dict[str, Any] = field(default_factory=dict)
    original: str = ""
    edited: str = ""
    diff: Dict[str, Any] = field(default_factory=dict)
    change_type: str = "other"
    preference: Dict[str, Optional[str]] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class WorkflowVariant:
    """A variant of a workflow with performance metrics."""
    workflow_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    utility: float = 0.0
    constraints_satisfied: bool = True
