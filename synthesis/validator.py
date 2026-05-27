"""Validate synthesized workflows before deployment."""

from __future__ import annotations
import asyncio
import ast
import traceback
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config.settings import AgentConfig
from ..core.memory_types import WorkflowSpec, WorkflowStep
from ..tools.sandbox import SafeExecutionSandbox

class ValidationStatus(Enum):
    PASSED = auto()
    FAILED = auto()
    WARNING = auto()

@dataclass
class ValidationResult:
    """Result of workflow validation."""
    status: ValidationStatus
    passed: bool
    confidence: float
    failures: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    static_analysis: Dict[str, Any] = field(default_factory=dict)

class WorkflowValidator:
    """
    Multi-stage workflow validation:
    1. Static analysis: syntax, type hints, allowed operations
    2. Test generation: auto-generate test cases from examples
    3. Sandbox execution: run tests in isolated environment
    4. Behavior verification: check outputs match expectations
    5. Safety audit: resource limits, infinite loops, side effects
    """
    
    def __init__(self, config: AgentConfig, sandbox: SafeExecutionSandbox):
        self.config = config
        self.sandbox = sandbox
    
    async def validate_workflow(
        self,
        workflow: WorkflowSpec,
        test_inputs: List[Any],
        expected_behavior: Dict[str, Any],
    ) -> ValidationResult:
        """Run full validation pipeline on a workflow."""
        
        results = []
        
        # Stage 1: Static analysis
        static_result = await self._static_analysis(workflow)
        results.append(("static", static_result))
        
        if static_result["status"] == "critical_failure":
            return ValidationResult(
                status=ValidationStatus.FAILED,
                passed=False,
                confidence=0.0,
                failures=[{"stage": "static", "error": static_result["error"]}],
            )
        
        # Stage 2: Generate and run tests
        test_results = await self._run_generated_tests(
            workflow=workflow,
            test_inputs=test_inputs,
            expected_behavior=expected_behavior,
        )
        results.append(("tests", test_results))
        
        # Stage 3: Safety audit
        safety_result = await self._safety_audit(workflow)
        results.append(("safety", safety_result))
        
        # Aggregate results
        return self._aggregate_validation_results(results, workflow)
    
    async def _static_analysis(self, workflow: WorkflowSpec) -> Dict[str, Any]:
        """Perform static code analysis on workflow."""
        
        issues = []
        warnings = []
        
        # Check syntax if compiled code exists
        if workflow.compiled_code:
            try:
                ast.parse(workflow.compiled_code)
            except SyntaxError as e:
                return {
                    "status": "critical_failure",
                    "error": f"Syntax error: {str(e)}",
                    "line": e.lineno,
                }
        
        # Check for forbidden operations
        forbidden = set(self.config.forbidden_operations)
        for step in workflow.steps:
            if step.operation in forbidden:
                issues.append({
                    "type": "forbidden_operation",
                    "step": step.step_number,
                    "operation": step.operation,
                })
        
        # Check type consistency
        type_issues = self._check_type_consistency(workflow)
        issues.extend(type_issues)
        
        # Check for unreachable code
        unreachable = self._detect_unreachable_steps(workflow)
        warnings.extend(unreachable)
        
        return {
            "status": "failure" if issues else "warning" if warnings else "passed",
            "issues": issues,
            "warnings": warnings,
            "complexity_score": self._compute_complexity_score(workflow),
        }
    
    async def _run_generated_tests(
        self,
        workflow: WorkflowSpec,
        test_inputs: List[Any],
        expected_behavior: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate and execute test cases."""
        
        if not test_inputs:
            return {"status": "skipped", "reason": "no_test_inputs"}
        
        results = []
        passed = 0
        failed = 0
        
        for i, test_input in enumerate(test_inputs[:self.config.max_validation_tests]):
            try:
                # Execute workflow in sandbox
                output = await self.sandbox.execute_workflow(
                    workflow=workflow,
                    input_data=test_input,
                    timeout_seconds=self.config.validation_timeout,
                )
                
                # Verify output matches expectations
                is_valid = self._verify_output(
                    output=output,
                    expected=expected_behavior,
                    test_input=test_input,
                )
                
                if is_valid:
                    passed += 1
                    results.append({"index": i, "passed": True, "output": output})
                else:
                    failed += 1
                    results.append({
                        "index": i,
                        "passed": False,
                        "error": "Output validation failed",
                        "expected": expected_behavior,
                        "actual": output,
                    })
                    
            except TimeoutError:
                failed += 1
                results.append({"index": i, "passed": False, "error": "timeout"})
            except Exception as e:
                failed += 1
                results.append({
                    "index": i,
                    "passed": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
        
        pass_rate = passed / len(test_inputs) if test_inputs else 0
        
        return {
            "status": "passed" if pass_rate >= self.config.min_test_pass_rate else "failed",
            "pass_rate": pass_rate,
            "total_tests": len(test_inputs),
            "passed": passed,
            "failed": failed,
            "results": results,
        }
    
    async def _safety_audit(self, workflow: WorkflowSpec) -> Dict[str, Any]:
        """Audit workflow for safety concerns."""
        
        concerns = []
        
        # Check for infinite loop potential
        if self._detect_infinite_loop_risk(workflow):
            concerns.append({
                "type": "infinite_loop_risk",
                "severity": "high",
                "description": "Workflow contains unbounded loop without exit condition",
            })
        
        # Check for excessive resource usage
        resource_estimate = self._estimate_resource_usage(workflow)
        if resource_estimate["estimated_memory_mb"] > self.config.max_memory_usage_mb:
            concerns.append({
                "type": "memory_risk",
                "severity": "medium",
                "estimated_mb": resource_estimate["estimated_memory_mb"],
            })
        
        if resource_estimate["estimated_time_seconds"] > self.config.max_workflow_execution_time_seconds:
            concerns.append({
                "type": "timeout_risk",
                "severity": "medium",
                "estimated_seconds": resource_estimate["estimated_time_seconds"],
            })
        
        # Check for dangerous side effects
        side_effects = self._detect_side_effects(workflow)
        if side_effects:
            concerns.append({
                "type": "side_effects",
                "severity": "low",
                "effects": side_effects,
            })
        
        return {
            "status": "warning" if concerns else "passed",
            "concerns": concerns,
            "resource_estimate": resource_estimate,
        }
    
    def _aggregate_validation_results(
        self,
        stage_results: List[tuple],
        workflow: WorkflowSpec,
    ) -> ValidationResult:
        """Combine validation stage results into final decision."""
        
        all_failures = []
        all_warnings = []
        test_results = []
        
        for stage, result in stage_results:
            if result.get("status") == "failure" or result.get("status") == "critical_failure":
                all_failures.append({"stage": stage, **result})
            elif result.get("status") == "warning":
                all_warnings.extend(result.get("warnings", []))
                all_warnings.extend(result.get("concerns", []))
            
            if stage == "tests":
                test_results = result.get("results", [])
        
        # Compute overall confidence
        confidence = self._compute_validation_confidence(
            failures=len(all_failures),
            warnings=len(all_warnings),
            test_pass_rate=result.get("pass_rate", 1.0) if stage_results else 1.0,
        )
        
        # Determine final status
        if all_failures:
            status = ValidationStatus.FAILED
            passed = False
        elif all_warnings and confidence < self.config.min_validation_confidence:
            status = ValidationStatus.WARNING
            passed = False
        else:
            status = ValidationStatus.PASSED
            passed = confidence >= self.config.min_validation_confidence
        
        return ValidationResult(
            status=status,
            passed=passed,
            confidence=confidence,
            failures=all_failures,
            warnings=all_warnings,
            test_results=test_results,
            static_analysis=next((r for s, r in stage_results if s == "static"), {}),
        )
    
    def _compute_validation_confidence(
        self,
        failures: int,
        warnings: int,
        test_pass_rate: float,
    ) -> float:
        """Compute overall validation confidence score."""
        
        # Start with test pass rate as base
        confidence = test_pass_rate
        
        # Penalize for failures
        confidence *= (1.0 - min(1.0, failures * 0.5))
        
        # Penalize for warnings
        confidence *= (1.0 - min(0.3, warnings * 0.05))
        
        return max(0.0, min(1.0, confidence))
    
    # Helper methods (simplified implementations)
    def _check_type_consistency(self, workflow: WorkflowSpec) -> List[Dict]:
        return []
    
    def _detect_unreachable_steps(self, workflow: WorkflowSpec) -> List[str]:
        return []
    
    def _compute_complexity_score(self, workflow: WorkflowSpec) -> float:
        return len(workflow.steps) * 1.5
    
    def _verify_output(self, output: Any, expected: Dict, test_input: Any) -> bool:
        # Would implement schema validation
        return True
    
    def _detect_infinite_loop_risk(self, workflow: WorkflowSpec) -> bool:
        # Would analyze control flow for unbounded loops
        return False
    
    def _estimate_resource_usage(self, workflow: WorkflowSpec) -> Dict[str, float]:
        return {
            "estimated_memory_mb": 50.0,
            "estimated_time_seconds": 2.0,
        }
    
    def _detect_side_effects(self, workflow: WorkflowSpec) -> List[str]:
        # Would check for file I/O, network calls, etc.
        return []
