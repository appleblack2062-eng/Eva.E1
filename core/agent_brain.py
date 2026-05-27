"""Main orchestrator for self-optimizing agentic memory."""

from __future__ import annotations
import asyncio
import time
import hashlib
from typing import Optional, List, Dict, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from ..config.settings import AgentConfig, GlobalConfig
from ..config.thresholds import OptimizationThresholds, EfficiencyTargets
from ..memory.layers.task_memory import TaskMemoryLayer
from ..memory.layers.workflow_memory import WorkflowMemoryLayer
from ..memory.layers.tool_memory import ToolMemoryLayer
from ..memory.layers.strategy_memory import StrategyMemoryLayer
from ..memory.layers.feedback_memory import FeedbackMemoryLayer
from ..memory.stores.vector_store import TaskVectorStore
from ..memory.stores.graph_store import WorkflowGraphStore
from ..memory.stores.cache_store import ExecutionCacheStore
from ..memory.retrieval.hybrid_router import HybridExecutionRouter
from ..synthesis.workflow_generator import WorkflowSynthesizer
from ..synthesis.validator import WorkflowValidator
from ..synthesis.optimizer import WorkflowOptimizer
from ..synthesis.tool_builder import ToolBuilder
from ..execution.orchestrator import ExecutionOrchestrator
from ..execution.fallback_handler import FallbackHandler
from ..execution.profiler import ExecutionProfiler
from ..learning.feedback_processor import FeedbackProcessor
from ..learning.workflow_refiner import WorkflowRefiner
from ..learning.strategy_learner import StrategyLearner
from ..tools.registry import ToolRegistry
from ..tools.sandbox import SafeExecutionSandbox
from ..utils.safety import ResourceMonitor, TimeoutGuard

T = TypeVar('T')

class ExecutionMode(Enum):
    """How a task should be executed."""
    LLM_ONLY = auto()              # Pure LLM inference
    LLM_GUIDED = auto()            # LLM plans, tools execute
    WORKFLOW_DRAFT = auto()        # Synthesized workflow, interpreted
    WORKFLOW_COMPILED = auto()     # Optimized Python, cached
    WORKFLOW_JIT = auto()          # JIT-compiled hot path
    HYBRID_FALLBACK = auto()       # Workflow + LLM fallback on error

class TaskStatus(Enum):
    """Lifecycle status of a task."""
    RECEIVED = auto()
    ROUTED = auto()
    EXECUTING = auto()
    VALIDATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    FALLBACK_USED = auto()
    OPTIMIZED = auto()

@dataclass
class TaskResult(Generic[T]):
    """Result of task execution with metadata."""
    success: bool
    output: Optional[T]
    execution_mode: ExecutionMode
    latency_ms: float
    tokens_used: int
    workflow_id: Optional[str]
    error: Optional[str] = None
    fallback_triggered: bool = False
    optimization_applied: bool = False

class AgentBrain:
    """
    Self-optimizing agent memory and execution engine.
    
    Core loop:
    1. Receive task → retrieve similar past tasks
    2. Router decides: LLM vs workflow vs hybrid
    3. Execute with appropriate mode + safety guards
    4. Profile results → extract feedback signals
    5. If pattern detected: synthesize → validate → optimize workflow
    6. Update routing strategy based on performance
    7. Continuously reduce LLM dependency while maintaining quality
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        global_config: GlobalConfig,
        llm_client,
        embedding_provider,
    ):
        self.agent_id = agent_id
        self.config = config
        self.thresholds = OptimizationThresholds()
        self.targets = EfficiencyTargets()
        
        # Core dependencies
        self.llm = llm_client
        self.embedder = embedding_provider
        self.sandbox = SafeExecutionSandbox(config)
        self.resource_monitor = ResourceMonitor(config)
        
        # Memory layers
        self.task_memory = TaskMemoryLayer(agent_id, config)
        self.workflow_memory = WorkflowMemoryLayer(agent_id, config)
        self.tool_memory = ToolMemoryLayer(agent_id, config)
        self.strategy_memory = StrategyMemoryLayer(agent_id, config)
        self.feedback_memory = FeedbackMemoryLayer(agent_id, config)
        
        # Storage backends
        self.vector_store = TaskVectorStore(agent_id, config, embedding_provider)
        self.graph_store = WorkflowGraphStore(agent_id, config)
        self.cache_store = ExecutionCacheStore(config)
        
        # Synthesis & optimization pipeline
        self.synthesizer = WorkflowSynthesizer(llm_client, config)
        self.validator = WorkflowValidator(config, self.sandbox)
        self.optimizer = WorkflowOptimizer(config, self.sandbox)
        self.tool_builder = ToolBuilder(config, self.sandbox)
        
        # Execution system
        self.router = HybridExecutionRouter(
            config, self.strategy_memory, self.workflow_memory
        )
        self.orchestrator = ExecutionOrchestrator(
            config, self.llm, self.sandbox, self.tool_registry
        )
        self.fallback = FallbackHandler(config, self.llm)
        self.profiler = ExecutionProfiler(config)
        
        # Learning systems
        self.feedback_processor = FeedbackProcessor(config)
        self.workflow_refiner = WorkflowRefiner(config, self.optimizer)
        self.strategy_learner = StrategyLearner(config)
        
        # Runtime state
        self._task_counter = 0
        self._optimization_queue: asyncio.Queue = asyncio.Queue()
        self._is_optimizing = False
        self._shutdown = False
        
        # Start background optimization loop
        asyncio.create_task(self._optimization_loop())
    
    @property
    def tool_registry(self) -> ToolRegistry:
        """Lazy initialization of tool registry."""
        if not hasattr(self, '_tool_registry'):
            self._tool_registry = ToolRegistry(
                config=self.config,
                sandbox=self.sandbox,
                tool_memory=self.tool_memory
            )
        return self._tool_registry
    
    # -----------------------------------------------------------------------
    # Public API: Task Execution
    # -----------------------------------------------------------------------
    
    async def execute_task(
        self,
        task_description: str,
        task_input: Any,
        expected_output_type: type = Any,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "normal",  # low, normal, high, critical
        timeout_seconds: Optional[float] = None,
    ) -> TaskResult:
        """
        Execute a task with automatic optimization and fallback.
        
        This is the main entry point for all agent work.
        """
        task_id = self._generate_task_id(task_description, task_input)
        start_time = time.time()
        
        self._task_counter += 1
        task_meta = {
            "task_id": task_id,
            "description": task_description,
            "input_hash": hashlib.sha256(str(task_input).encode()).hexdigest()[:16],
            "timestamp": time.time(),
            "priority": priority,
        }
        
        try:
            # 1. Retrieve similar past tasks for context
            similar_tasks = await self.task_memory.find_similar(
                query=task_description,
                input_context=task_input,
                limit=10,
                min_similarity=0.7
            )
            
            # 2. Router decides execution mode
            routing_decision = await self.router.decide_execution_mode(
                task_description=task_description,
                task_input=task_input,
                similar_tasks=similar_tasks,
                context=context,
                thresholds=self.thresholds
            )
            
            # 3. Execute with appropriate mode + safety guards
            with TimeoutGuard(timeout_seconds or self.config.max_task_timeout):
                result = await self._execute_with_mode(
                    task_id=task_id,
                    description=task_description,
                    input_data=task_input,
                    output_type=expected_output_type,
                    mode=routing_decision.mode,
                    workflow_id=routing_decision.workflow_id,
                    context=context,
                    similar_tasks=similar_tasks,
                )
            
            # 4. Profile and record execution
            latency_ms = (time.time() - start_time) * 1000
            profile = self.profiler.record_execution(
                task_id=task_id,
                mode=routing_decision.mode,
                latency_ms=latency_ms,
                tokens_used=result.tokens_used,
                success=result.success,
                workflow_id=result.workflow_id,
            )
            
            # 5. Extract feedback for learning
            feedback = self.feedback_processor.extract_signals(
                task_meta=task_meta,
                execution_result=result,
                profile=profile,
                routing_decision=routing_decision,
            )
            await self.feedback_memory.store(feedback)
            
            # 6. Queue for optimization if pattern detected
            if self._should_queue_for_optimization(result, similar_tasks):
                await self._optimization_queue.put({
                    "task_id": task_id,
                    "description": task_description,
                    "input_sample": task_input,
                    "result": result,
                    "similar_tasks": similar_tasks,
                    "feedback": feedback,
                })
            
            # 7. Update routing strategy based on outcome
            await self.strategy_learner.update_policy(
                task_pattern=routing_decision.task_pattern,
                outcome=result.success,
                latency_ms=latency_ms,
                tokens_used=result.tokens_used,
                mode_used=routing_decision.mode,
            )
            
            # 8. Store task result for future retrieval
            await self.task_memory.store_result(
                task_id=task_id,
                description=task_description,
                input_data=task_input,
                output=result.output,
                execution_mode=routing_decision.mode,
                success=result.success,
                latency_ms=latency_ms,
                tokens_used=result.tokens_used,
                workflow_id=result.workflow_id,
            )
            
            return result
            
        except TimeoutError as e:
            return TaskResult(
                success=False,
                output=None,
                execution_mode=ExecutionMode.LLM_ONLY,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                workflow_id=None,
                error=f"Timeout: {str(e)}",
            )
        except Exception as e:
            # Critical error: log and fallback
            await self.feedback_memory.store_critical_error(
                task_id=task_id,
                error=repr(e),
                traceback=True,
            )
            return await self.fallback.handle_critical_failure(
                task_description=task_description,
                error=e,
                context=context,
            )
    
    async def _execute_with_mode(
        self,
        task_id: str,
        description: str,
        input_data: Any,
        output_type: type,
        mode: ExecutionMode,
        workflow_id: Optional[str],
        context: Optional[Dict[str, Any]],
        similar_tasks: List[Dict[str, Any]],
    ) -> TaskResult:
        """Execute task using the specified mode with safety guards."""
        
        if mode == ExecutionMode.LLM_ONLY:
            return await self._execute_llm_only(
                task_id, description, input_data, output_type, context
            )
        
        elif mode == ExecutionMode.LLM_GUIDED:
            return await self.orchestrator.execute_llm_guided(
                task_id=task_id,
                description=description,
                input_data=input_data,
                output_type=output_type,
                context=context,
                tool_registry=self.tool_registry,
            )
        
        elif mode in [ExecutionMode.WORKFLOW_DRAFT, ExecutionMode.WORKFLOW_COMPILED, ExecutionMode.WORKFLOW_JIT]:
            if not workflow_id:
                # Fallback if workflow ID missing
                return await self._execute_llm_only(
                    task_id, description, input_data, output_type, context
                )
            
            workflow = await self.workflow_memory.get_workflow(workflow_id)
            if not workflow:
                return await self._execute_llm_only(
                    task_id, description, input_data, output_type, context
                )
            
            # Execute workflow with fallback protection
            return await self._execute_workflow_with_fallback(
                task_id=task_id,
                description=description,
                input_data=input_data,
                output_type=output_type,
                workflow=workflow,
                mode=mode,
                context=context,
            )
        
        elif mode == ExecutionMode.HYBRID_FALLBACK:
            # Try workflow first, fallback to LLM on error
            if workflow_id:
                workflow = await self.workflow_memory.get_workflow(workflow_id)
                if workflow:
                    result = await self._execute_workflow_with_fallback(
                        task_id, description, input_data, output_type,
                        workflow, ExecutionMode.WORKFLOW_COMPILED, context
                    )
                    if result.success or not result.fallback_triggered:
                        return result
            
            # Fallback to LLM
            return await self._execute_llm_only(
                task_id, description, input_data, output_type, context
            )
        
        else:
            # Unknown mode: safest to use LLM
            return await self._execute_llm_only(
                task_id, description, input_data, output_type, context
            )
    
    async def _execute_llm_only(
        self,
        task_id: str,
        description: str,
        input_data: Any,
        output_type: type,
        context: Optional[Dict[str, Any]],
    ) -> TaskResult:
        """Pure LLM execution path."""
        start = time.time()
        
        # Build prompt with retrieved context
        prompt = await self._build_llm_prompt(description, input_data, context)
        
        # Execute with token/latency limits
        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=self.config.max_llm_tokens_per_task,
            temperature=self.config.llm_temperature,
        )
        
        # Parse and validate output
        try:
            output = self._parse_llm_output(response, output_type)
            success = True
            error = None
        except Exception as e:
            output = None
            success = False
            error = f"Output parsing failed: {str(e)}"
        
        latency_ms = (time.time() - start) * 1000
        
        return TaskResult(
            success=success,
            output=output,
            execution_mode=ExecutionMode.LLM_ONLY,
            latency_ms=latency_ms,
            tokens_used=response.token_count,
            workflow_id=None,
            error=error,
        )
    
    async def _execute_workflow_with_fallback(
        self,
        task_id: str,
        description: str,
        input_data: Any,
        output_type: type,
        workflow,
        mode: ExecutionMode,
        context: Optional[Dict[str, Any]],
    ) -> TaskResult:
        """Execute workflow with automatic LLM fallback on failure."""
        start = time.time()
        fallback_triggered = False
        
        try:
            # Execute workflow in sandbox
            output = await self.orchestrator.run_workflow(
                workflow=workflow,
                input_data=input_data,
                output_type=output_type,
                context=context,
                mode=mode,
            )
            
            # Validate output matches expected type
            if not self._validate_output_type(output, output_type):
                raise TypeError(f"Output type mismatch: expected {output_type}, got {type(output)}")
            
            success = True
            error = None
            
        except Exception as e:
            # Workflow failed: trigger fallback
            fallback_triggered = True
            self._log_workflow_failure(task_id, workflow.id, e)
            
            # Attempt LLM fallback
            fallback_result = await self.fallback.handle_workflow_failure(
                task_description=description,
                input_data=input_data,
                workflow_error=e,
                context=context,
                output_type=output_type,
            )
            
            success = fallback_result.success
            output = fallback_result.output
            error = fallback_result.error if not success else None
        
        latency_ms = (time.time() - start) * 1000
        
        return TaskResult(
            success=success,
            output=output,
            execution_mode=mode if not fallback_triggered else ExecutionMode.HYBRID_FALLBACK,
            latency_ms=latency_ms,
            tokens_used=0 if not fallback_triggered else getattr(output, 'tokens_used', 0),
            workflow_id=workflow.id,
            error=error,
            fallback_triggered=fallback_triggered,
            optimization_applied=mode != ExecutionMode.WORKFLOW_DRAFT,
        )
    
    # -----------------------------------------------------------------------
    # Background Optimization Loop
    # -----------------------------------------------------------------------
    
    async def _optimization_loop(self):
        """Background task: continuously optimize workflows from queue."""
        while not self._shutdown:
            try:
                # Get next optimization candidate
                item = await asyncio.wait_for(
                    self._optimization_queue.get(),
                    timeout=60.0  # Wait up to 1 minute for new work
                )
                
                await self._optimize_task_pattern(item)
                self._optimization_queue.task_done()
                
            except asyncio.TimeoutError:
                # No new work: idle cycle
                await asyncio.sleep(5.0)
            except Exception as e:
                # Log error but keep loop running
                await self.feedback_memory.store_optimization_error(repr(e))
                await asyncio.sleep(1.0)
    
    async def _optimize_task_pattern(self, item: Dict[str, Any]):
        """Synthesize → validate → optimize workflow for a task pattern."""
        
        task_id = item["task_id"]
        description = item["description"]
        input_sample = item["input_sample"]
        similar_tasks = item["similar_tasks"]
        
        # Skip if already optimized recently
        if await self.workflow_memory.has_recent_optimization(description, hours=24):
            return
        
        # Step 1: Synthesize workflow from examples
        workflow_draft = await self.synthesizer.generate_workflow(
            task_description=description,
            examples=similar_tasks,
            input_sample=input_sample,
        )
        
        if not workflow_draft:
            return
        
        # Step 2: Validate workflow correctness
        validation_result = await self.validator.validate_workflow(
            workflow=workflow_draft,
            test_inputs=self._generate_test_inputs(similar_tasks),
            expected_behavior=self._infer_expected_behavior(similar_tasks),
        )
        
        if not validation_result.passed:
            # Refine and retry up to N times
            for attempt in range(3):
                workflow_draft = await self.workflow_refiner.refine(
                    workflow=workflow_draft,
                    failures=validation_result.failures,
                    examples=similar_tasks,
                )
                validation_result = await self.validator.validate_workflow(
                    workflow=workflow_draft,
                    test_inputs=self._generate_test_inputs(similar_tasks),
                    expected_behavior=self._infer_expected_behavior(similar_tasks),
                )
                if validation_result.passed:
                    break
            
            if not validation_result.passed:
                await self.feedback_memory.store_validation_failure(
                    task_description=description,
                    workflow_id=workflow_draft.id,
                    failures=validation_result.failures,
                )
                return
        
        # Step 3: Optimize workflow performance
        optimized_workflow = await self.optimizer.optimize_workflow(
            workflow=workflow_draft,
            profile_data=self._collect_profile_data(similar_tasks),
            optimization_targets=self.targets,
        )
        
        # Step 4: Build optimized tools if applicable
        if optimized_workflow.has_custom_operations:
            custom_tools = await self.tool_builder.build_tools_from_workflow(
                workflow=optimized_workflow,
                performance_profile=optimized_workflow.profile,
            )
            for tool in custom_tools:
                await self.tool_memory.register_tool(tool)
                self.tool_registry.register(tool)
        
        # Step 5: Deploy workflow to memory
        deployment_id = await self.workflow_memory.deploy_workflow(
            workflow=optimized_workflow,
            task_pattern=self._extract_task_pattern(description, similar_tasks),
            performance_baseline=self._measure_baseline_performance(similar_tasks),
        )
        
        # Step 6: Update router to use new workflow
        await self.router.register_workflow_pattern(
            task_pattern=self._extract_task_pattern(description, similar_tasks),
            workflow_id=deployment_id,
            confidence=validation_result.confidence,
            expected_speedup=optimized_workflow.estimated_speedup,
        )
        
        # Step 7: Log optimization success
        await self.feedback_memory.store_optimization_success(
            task_description=description,
            workflow_id=deployment_id,
            original_mode=ExecutionMode.LLM_ONLY,
            new_mode=ExecutionMode.WORKFLOW_COMPILED,
            estimated_token_savings=optimized_workflow.estimated_token_reduction,
            estimated_latency_reduction=optimized_workflow.estimated_latency_reduction,
        )
    
    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------
    
    def _generate_task_id(self, description: str, input_data: Any) -> str:
        """Generate unique task ID from description + input hash."""
        input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()[:16]
        desc_hash = hashlib.sha256(description.encode()).hexdigest()[:16]
        return f"task_{desc_hash}_{input_hash}_{int(time.time())}"
    
    def _should_queue_for_optimization(
        self,
        result: TaskResult,
        similar_tasks: List[Dict[str, Any]],
    ) -> bool:
        """Determine if this task execution should trigger optimization."""
        
        # Must be successful execution
        if not result.success:
            return False
        
        # Must have enough similar tasks to establish pattern
        if len(similar_tasks) < self.thresholds.min_task_repetitions_for_synthesis:
            return False
        
        # Prefer tasks that used LLM (opportunity to optimize)
        if result.execution_mode != ExecutionMode.LLM_ONLY:
            return False
        
        # Check if pattern already being optimized
        # (Would query workflow_memory for pending optimizations)
        
        return True
    
    def _extract_task_pattern(
        self,
        description: str,
        similar_tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract abstract pattern from task description and examples."""
        # Would use LLM or clustering to identify task class
        return {
            "intent": self._classify_intent(description),
            "input_schema": self._infer_input_schema(similar_tasks),
            "output_schema": self._infer_output_schema(similar_tasks),
            "complexity": self._estimate_complexity(description, similar_tasks),
            "domain": self._identify_domain(description),
        }
    
    async def _build_llm_prompt(
        self,
        description: str,
        input_data: Any,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Construct optimized prompt for LLM execution."""
        # Would include retrieved examples, constraints, output format spec
        return f"Task: {description}\nInput: {input_data}\nContext: {context}\nRespond with JSON:"
    
    def _parse_llm_output(self, response: Any, expected_type: type) -> Any:
        """Parse and validate LLM output against expected type."""
        # Would use Pydantic or type validation
        return response.content if hasattr(response, 'content') else response
    
    def _validate_output_type(self, output: Any, expected_type: type) -> bool:
        """Check if output matches expected type."""
        if expected_type == Any:
            return True
        return isinstance(output, expected_type)
    
    def _generate_test_inputs(self, similar_tasks: List[Dict[str, Any]]) -> List[Any]:
        """Generate diverse test inputs from historical tasks."""
        # Would sample and vary inputs from similar tasks
        return [t.get("input") for t in similar_tasks[:5]]
    
    def _infer_expected_behavior(self, similar_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer expected output characteristics from examples."""
        # Would analyze output patterns from successful tasks
        if not similar_tasks:
            return {}
        
        successful = [t for t in similar_tasks if t.get("success")]
        if not successful:
            return {}
        
        return {
            "output_type": type(successful[0].get("output")),
            "common_properties": self._extract_common_properties([t.get("output") for t in successful]),
        }
    
    def _collect_profile_data(self, similar_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect performance metrics from historical executions."""
        if not similar_tasks:
            return {}
        
        latencies = [t.get("latency_ms", 0) for t in similar_tasks if t.get("latency_ms")]
        tokens = [t.get("tokens_used", 0) for t in similar_tasks if t.get("tokens_used")]
        
        return {
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
            "success_rate": sum(1 for t in similar_tasks if t.get("success")) / len(similar_tasks),
        }
    
    def _measure_baseline_performance(self, similar_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Establish performance baseline for optimization comparison."""
        profile = self._collect_profile_data(similar_tasks)
        return {
            "latency_baseline": profile.get("avg_latency_ms", 1000),
            "token_baseline": profile.get("avg_tokens", 2000),
            "success_baseline": profile.get("success_rate", 0.9),
        }
    
    def _classify_intent(self, description: str) -> str:
        """Classify task intent (e.g., 'data_transform', 'api_call', 'reasoning')."""
        # Would use embedding similarity or LLM classification
        keywords = description.lower()
        if any(k in keywords for k in ['filter', 'sort', 'transform', 'aggregate']):
            return 'data_transform'
        elif any(k in keywords for k in ['fetch', 'request', 'http', 'api']):
            return 'api_call'
        elif any(k in keywords for k in ['calculate', 'compute', 'analyze']):
            return 'computation'
        else:
            return 'general_reasoning'
    
    def _infer_input_schema(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer input structure from historical task inputs."""
        # Would use schema inference from examples
        return {"type": "object", "properties": {}}  # Placeholder
    
    def _infer_output_schema(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer output structure from historical task outputs."""
        return {"type": "object", "properties": {}}  # Placeholder
    
    def _estimate_complexity(self, description: str, tasks: List[Dict[str, Any]]) -> str:
        """Estimate task complexity: simple, moderate, complex."""
        # Would analyze description length, nesting, dependencies
        return 'moderate'
    
    def _identify_domain(self, description: str) -> str:
        """Identify task domain: data, web, math, text, etc."""
        return 'general'
    
    def _extract_common_properties(self, outputs: List[Any]) -> List[str]:
        """Extract common properties from a list of outputs."""
        if not outputs:
            return []
        # Would analyze output structure for common fields
        return []
    
    def _log_workflow_failure(self, task_id: str, workflow_id: str, error: Exception):
        """Log workflow execution failure for debugging."""
        print(f"[Workflow Failure] task={task_id} workflow={workflow_id} error={repr(error)}")
    
    # -----------------------------------------------------------------------
    # Introspection & Management API
    # -----------------------------------------------------------------------
    
    async def get_agent_metrics(self) -> Dict[str, Any]:
        """Return comprehensive agent performance metrics."""
        return {
            "total_tasks": self._task_counter,
            "llm_offload_ratio": await self.strategy_memory.compute_llm_offload_ratio(),
            "avg_latency_ms": await self.profiler.get_average_latency(),
            "avg_tokens_per_task": await self.profiler.get_average_tokens(),
            "workflow_reuse_rate": await self.workflow_memory.compute_reuse_rate(),
            "optimization_queue_size": self._optimization_queue.qsize(),
            "active_workflows": await self.workflow_memory.count_active_workflows(),
            "cached_tools": self.tool_registry.count_registered(),
        }
    
    async def list_active_workflows(self, pattern_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List deployed workflows, optionally filtered by pattern."""
        return await self.workflow_memory.list_workflows(pattern_filter)
    
    async def force_optimize_pattern(self, task_pattern: str):
        """Manually trigger optimization for a specific task pattern."""
        # Would queue all historical tasks matching pattern for optimization
        pass
    
    async def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Export a workflow for inspection or external use."""
        workflow = await self.workflow_memory.get_workflow(workflow_id)
        if not workflow:
            return {}
        
        return {
            "id": workflow.id,
            "pattern": workflow.task_pattern,
            "version": workflow.version,
            "code": workflow.compiled_code,
            "tools_used": workflow.required_tools,
            "performance": workflow.performance_metrics,
            "validation_results": workflow.validation_history,
        }
    
    async def shutdown(self):
        """Gracefully shut down the agent."""
        self._shutdown = True
        await self._optimization_queue.join()  # Wait for pending optimizations
        await self.task_memory.flush()
        await self.workflow_memory.flush()
        await self.feedback_memory.flush()
