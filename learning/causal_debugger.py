"""Causal Debugger for root-cause analysis of workflow errors.

This module implements causal inference using Bayesian networks and do-calculus
to identify the root cause of workflow failures and generate targeted fixes.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time


@dataclass
class WorkflowStep:
    """A step in the workflow with causal properties."""
    id: str
    operation: str
    parents: List[str] = field(default_factory=list)
    error_probabilities: Dict[str, float] = field(default_factory=dict)


class CausalGraph:
    """Bayesian network of workflow steps and error types."""
    
    def __init__(self, workflow):
        """Initialize causal graph from workflow specification."""
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.error_model: Dict[str, Dict[str, List[float]]] = {}
        
        # Build nodes from workflow steps
        if hasattr(workflow, 'steps'):
            for step in workflow.steps:
                step_id = step.get('id', f"step_{step.get('step_number', 0)}") if isinstance(step, dict) else getattr(step, 'id', f"step_{getattr(step, 'step_number', 0)}")
                operation = step.get('operation', 'unknown') if isinstance(step, dict) else getattr(step, 'operation', 'unknown')
                
                self.nodes[step_id] = {
                    "type": operation,
                    "parents": [],
                }
        
        # Build parent relationships (simplified: sequential dependency)
        step_ids = list(self.nodes.keys())
        for i in range(1, len(step_ids)):
            self.nodes[step_ids[i]]["parents"].append(step_ids[i-1])
    
    def learn_from_interventions(self, intervention_data: List[Dict[str, Any]]):
        """Update causal model from synthetic error injections."""
        for record in intervention_data:
            step_id = record.get("step")
            error_type = record.get("error", "unknown")
            
            if step_id not in self.error_model:
                self.error_model[step_id] = {}
            
            if error_type not in self.error_model[step_id]:
                self.error_model[step_id][error_type] = []
            
            # Record success indicator
            self.error_model[step_id][error_type].append(1.0 if record.get("success", False) else 0.0)
    
    def infer_root_cause(
        self, 
        observed_error: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Use do-calculus to identify most likely causative step."""
        scores: Dict[str, float] = {}
        
        for step_id in self.nodes:
            # P(error | do(fix_step)) ≈ P(error | step fixed)
            prob_if_fixed = self._counterfactual_prob(step_id, observed_error, context)
            # Higher score = more likely to be the cause
            scores[step_id] = 1.0 - prob_if_fixed
        
        # Return top 3 most likely causes
        sorted_causes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cause for cause, score in sorted_causes[:3] if score > 0.1]
    
    def _counterfactual_prob(
        self, 
        step_id: str, 
        error_type: str, 
        context: Dict[str, Any]
    ) -> float:
        """Estimate P(error | do(step is fixed)) via propensity matching."""
        # Simplified: use historical success rate of step when context matches
        history = self._get_step_history(step_id, context)
        
        if history["total_count"] == 0:
            return 0.5  # Prior probability when no data
        
        success_rate = history["success_count"] / history["total_count"]
        return 1.0 - success_rate  # Probability of error
    
    def _get_step_history(
        self, 
        step_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, int]:
        """Get historical performance data for a step."""
        # Check if we have error model data for this step
        if step_id in self.error_model:
            total = 0
            success = 0
            
            for error_type, outcomes in self.error_model[step_id].items():
                for outcome in outcomes:
                    total += 1
                    success += outcome
            
            return {"success_count": success, "total_count": total}
        
        return {"success_count": 0, "total_count": 0}
    
    def update_step_success(
        self, 
        step_id: str, 
        error_type: str, 
        success: bool
    ):
        """Record outcome for a step execution."""
        if step_id not in self.error_model:
            self.error_model[step_id] = {}
        
        if error_type not in self.error_model[step_id]:
            self.error_model[step_id][error_type] = []
        
        self.error_model[step_id][error_type].append(1.0 if success else 0.0)


class CausalDebugger:
    """Integrate causal inference into workflow refinement."""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def diagnose_and_fix(
        self, 
        workflow, 
        error: Exception, 
        context: Dict[str, Any]
    ):
        """Diagnose root cause and generate targeted fix."""
        # Build causal graph from workflow
        causal_graph = CausalGraph(workflow)
        
        # Infer root cause
        likely_causes = causal_graph.infer_root_cause(str(error), context)
        
        if not likely_causes:
            # No specific cause identified, return workflow as-is
            return workflow
        
        # Generate targeted fix prompt
        fix_prompt = f"""
Workflow error: {error}
Likely causative step(s): {likely_causes}
Context: {context}

Provide a minimal fix for the causative step only.
Return modified step spec as JSON.
"""
        
        # Apply fix if LLM is available
        if self.llm:
            try:
                fix_spec = await self.llm.generate(fix_prompt, response_format="json")
                workflow = self._apply_fix(workflow, likely_causes[0], fix_spec)
            except Exception:
                # LLM failed, apply heuristic fix
                workflow = self._apply_heuristic_fix(workflow, likely_causes[0], error)
        else:
            # No LLM, apply heuristic fix
            workflow = self._apply_heuristic_fix(workflow, likely_causes[0], error)
        
        return workflow
    
    def _apply_fix(self, workflow, step_id: str, fix_spec: Dict[str, Any]):
        """Apply LLM-generated fix to workflow."""
        # Convert workflow to mutable form if needed
        if hasattr(workflow, 'steps'):
            steps = workflow.steps
            for i, step in enumerate(steps):
                current_id = step.get('id', f"step_{i}") if isinstance(step, dict) else getattr(step, 'id', f"step_{i}")
                if current_id == step_id:
                    # Apply fix to this step
                    if isinstance(step, dict):
                        step.update(fix_spec)
                    else:
                        for key, value in fix_spec.items():
                            setattr(step, key, value)
                    break
        
        return workflow
    
    def _apply_heuristic_fix(self, workflow, step_id: str, error: Exception):
        """Apply heuristic-based fix when LLM is unavailable."""
        error_str = str(error).lower()
        
        # Common error patterns and fixes
        if "timeout" in error_str:
            # Add retry logic or increase timeout
            return self._add_retry_mechanism(workflow, step_id)
        elif "validation" in error_str or "format" in error_str:
            # Add input sanitization
            return self._add_input_sanitization(workflow, step_id)
        elif "null" in error_str or "none" in error_str:
            # Add null checks
            return self._add_null_checks(workflow, step_id)
        
        # Generic fix: wrap in try-except
        return self._add_error_handling(workflow, step_id)
    
    def _add_retry_mechanism(self, workflow, step_id: str):
        """Add retry logic to a step."""
        if hasattr(workflow, 'steps'):
            for step in workflow.steps:
                current_id = step.get('id', '') if isinstance(step, dict) else getattr(step, 'id', '')
                if current_id == step_id:
                    if isinstance(step, dict):
                        step['retry_count'] = step.get('retry_count', 0) + 3
                        step['retry_delay_ms'] = step.get('retry_delay_ms', 1000)
                    else:
                        setattr(step, 'retry_count', getattr(step, 'retry_count', 0) + 3)
                        setattr(step, 'retry_delay_ms', getattr(step, 'retry_delay_ms', 1000))
                    break
        return workflow
    
    def _add_input_sanitization(self, workflow, step_id: str):
        """Add input validation to a step."""
        if hasattr(workflow, 'steps'):
            for step in workflow.steps:
                current_id = step.get('id', '') if isinstance(step, dict) else getattr(step, 'id', '')
                if current_id == step_id:
                    if isinstance(step, dict):
                        step['validate_input'] = True
                        step['input_schema'] = step.get('input_schema', {'type': 'object'})
                    else:
                        setattr(step, 'validate_input', True)
                        setattr(step, 'input_schema', getattr(step, 'input_schema', {'type': 'object'}))
                    break
        return workflow
    
    def _add_null_checks(self, workflow, step_id: str):
        """Add null checking to a step."""
        if hasattr(workflow, 'steps'):
            for step in workflow.steps:
                current_id = step.get('id', '') if isinstance(step, dict) else getattr(step, 'id', '')
                if current_id == step_id:
                    if isinstance(step, dict):
                        step['handle_null'] = True
                        step['default_value'] = step.get('default_value', None)
                    else:
                        setattr(step, 'handle_null', True)
                        setattr(step, 'default_value', getattr(step, 'default_value', None))
                    break
        return workflow
    
    def _add_error_handling(self, workflow, step_id: str):
        """Add generic error handling to a step."""
        if hasattr(workflow, 'steps'):
            for step in workflow.steps:
                current_id = step.get('id', '') if isinstance(step, dict) else getattr(step, 'id', '')
                if current_id == step_id:
                    if isinstance(step, dict):
                        step['error_handling'] = 'catch_and_log'
                    else:
                        setattr(step, 'error_handling', 'catch_and_log')
                    break
        return workflow
