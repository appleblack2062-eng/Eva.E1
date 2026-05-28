"""Eva.E1 Meta-Learning Architecture.

This package provides self-improving agent capabilities:
- Pattern generalization for few-shot learning
- Causal debugging for root-cause analysis
- Meta-strategy control for optimization decisions
- Self-reflection and case-based reasoning
- Drift detection and adaptation
- Implicit feedback learning
"""

from .pattern_engine import (
    PatternEncoder,
    PatternGraph,
    PatternGeneralizationEngine,
    WorkflowRef,
)

from .causal_debugger import (
    CausalGraph,
    CausalDebugger,
    WorkflowStep,
)

from .meta_controller import (
    ContextualBandit,
    MetaStrategyController,
    PendingUpdate,
)

from .reflection_engine import (
    ReflectionEngine,
    Case,
)

from .drift_adapter import (
    PageHinkleyTest,
    DriftDetector,
    DriftAdapter,
)

from .implicit_feedback import (
    ImplicitFeedbackParser,
    FeedbackSignal,
)

__all__ = [
    # Pattern Engine
    "PatternEncoder",
    "PatternGraph", 
    "PatternGeneralizationEngine",
    "WorkflowRef",
    
    # Causal Debugger
    "CausalGraph",
    "CausalDebugger",
    "WorkflowStep",
    
    # Meta Controller
    "ContextualBandit",
    "MetaStrategyController",
    "PendingUpdate",
    
    # Reflection Engine
    "ReflectionEngine",
    "Case",
    
    # Drift Adapter
    "PageHinkleyTest",
    "DriftDetector",
    "DriftAdapter",
    
    # Implicit Feedback
    "ImplicitFeedbackParser",
    "FeedbackSignal",
]
