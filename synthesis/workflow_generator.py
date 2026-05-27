"""Synthesize executable workflows from task examples using LLM."""

from __future__ import annotations
import re
import ast
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..config.settings import AgentConfig
from ..core.memory_types import WorkflowSpec, WorkflowStep, ToolSpec

class WorkflowSynthesizer:
    """
    Converts task examples into executable workflow specifications.
    
    Process:
    1. Analyze input/output patterns from examples
    2. Generate pseudo-code workflow with LLM
    3. Parse pseudo-code into structured WorkflowSpec
    4. Extract required tools and operations
    5. Annotate with type hints and constraints
    """
    
    def __init__(self, llm_client, config: AgentConfig):
        self.llm = llm_client
        self.config = config
    
    async def generate_workflow(
        self,
        task_description: str,
        examples: List[Dict[str, Any]],
        input_sample: Any,
    ) -> Optional[WorkflowSpec]:
        """Generate a workflow specification from examples."""
        
        if not examples:
            return None
        
        # Step 1: Analyze examples to extract patterns
        pattern_analysis = await self._analyze_task_patterns(examples)
        
        # Step 2: Generate pseudo-code workflow via LLM
        pseudo_code = await self._generate_pseudo_code(
            task_description=task_description,
            pattern_analysis=pattern_analysis,
            input_sample=input_sample,
        )
        
        if not pseudo_code:
            return None
        
        # Step 3: Parse pseudo-code into structured spec
        workflow_spec = self._parse_pseudo_code_to_spec(pseudo_code, pattern_analysis)
        
        if not workflow_spec:
            return None
        
        # Step 4: Validate spec structure
        if not self._validate_spec_structure(workflow_spec):
            return None
        
        # Step 5: Annotate with metadata
        workflow_spec = self._annotate_workflow_spec(
            workflow_spec, task_description, pattern_analysis
        )
        
        return workflow_spec
    
    async def _analyze_task_patterns(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common patterns from successful task executions."""
        
        # Analyze input structures
        input_patterns = self._cluster_input_structures([e.get("input") for e in examples])
        
        # Analyze output structures
        output_patterns = self._cluster_output_structures([e.get("output") for e in examples])
        
        # Extract common operations
        operations = self._extract_common_operations(examples)
        
        # Identify decision points
        decision_points = self._identify_decision_points(examples)
        
        return {
            "input_schema": input_patterns,
            "output_schema": output_patterns,
            "common_operations": operations,
            "decision_points": decision_points,
            "success_criteria": self._infer_success_criteria(examples),
        }
    
    async def _generate_pseudo_code(
        self,
        task_description: str,
        pattern_analysis: Dict[str, Any],
        input_sample: Any,
    ) -> Optional[str]:
        """Use LLM to generate workflow pseudo-code."""
        
        prompt = f"""
        You are a workflow synthesis engine. Generate a step-by-step pseudo-code workflow 
        to accomplish this task.
        
        TASK: {task_description}
        
        EXAMPLES ANALYSIS:
        - Input pattern: {pattern_analysis['input_schema']}
        - Output pattern: {pattern_analysis['output_schema']}
        - Common operations: {pattern_analysis['common_operations']}
        - Decision points: {pattern_analysis['decision_points']}
        
        SAMPLE INPUT: {input_sample}
        
        Generate pseudo-code with this format:
        
        ```pseudo
        # Workflow: [name]
        # Input: [type description]
        # Output: [type description]
        
        STEP 1: [operation] [parameters]
        STEP 2: IF [condition] THEN
          STEP 2a: [operation]
        ELSE
          STEP 2b: [operation]
        ENDIF
        STEP 3: [operation] -> [output_variable]
        ...
        RETURN [output_variable]
        ```
        
        Rules:
        - Use only operations from: {self.config.allowed_operations}
        - Include type annotations for all variables
        - Handle errors gracefully with TRY/CATCH
        - Optimize for clarity over cleverness
        """
        
        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,  # Low temperature for deterministic output
        )
        
        # Extract pseudo-code block
        match = re.search(r'```pseudo\s*(.*?)\s*```', response.content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _parse_pseudo_code_to_spec(
        self,
        pseudo_code: str,
        pattern_analysis: Dict[str, Any],
    ) -> Optional[WorkflowSpec]:
        """Parse pseudo-code string into structured WorkflowSpec."""
        
        try:
            lines = [l.strip() for l in pseudo_code.split('\n') if l.strip()]
            
            # Parse metadata
            metadata = self._parse_workflow_metadata(lines)
            
            # Parse steps
            steps = []
            current_step = None
            indent_level = 0
            
            for line in lines:
                if line.startswith('#') or line.startswith('```'):
                    continue
                
                if line.startswith('STEP'):
                    step = self._parse_step_line(line, indent_level)
                    if step:
                        steps.append(step)
                        current_step = step
                elif line.startswith('IF') and current_step:
                    current_step.condition = self._parse_condition(line)
                elif line.startswith('RETURN') and current_step:
                    current_step.is_terminal = True
                    current_step.return_value = self._parse_return(line)
            
            if not steps:
                return None
            
            # Build spec
            spec = WorkflowSpec(
                name=metadata.get('name', 'auto_generated'),
                description=metadata.get('description', ''),
                input_schema=pattern_analysis['input_schema'],
                output_schema=pattern_analysis['output_schema'],
                steps=steps,
                required_tools=self._extract_required_tools(steps),
                error_handling=metadata.get('error_handling', 'fallback'),
            )
            
            return spec
            
        except Exception as e:
            print(f"Failed to parse pseudo-code: {e}")
            return None
    
    def _parse_workflow_metadata(self, lines: List[str]) -> Dict[str, Any]:
        """Extract workflow metadata from comment lines."""
        metadata = {}
        
        for line in lines:
            if line.startswith('# Workflow:'):
                metadata['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Input:'):
                metadata['input_desc'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Output:'):
                metadata['output_desc'] = line.split(':', 1)[1].strip()
        
        return metadata
    
    def _parse_step_line(self, line: str, indent_level: int) -> Optional[WorkflowStep]:
        """Parse a STEP line into a WorkflowStep object."""
        
        # Extract step number and content
        match = re.match(r'STEP\s+(\d+)([a-z]*)?:\s*(.+)', line, re.IGNORECASE)
        if not match:
            return None
        
        step_num = int(match.group(1))
        sub_step = match.group(2) or ""
        content = match.group(3).strip()
        
        # Parse operation and parameters
        parts = content.split(' ', 1)
        operation = parts[0].upper()
        parameters = parts[1] if len(parts) > 1 else ""
        
        return WorkflowStep(
            step_number=step_num,
            sub_step=sub_step,
            operation=operation,
            parameters=self._parse_parameters(parameters),
            indent_level=indent_level,
        )
    
    def _parse_parameters(self, param_str: str) -> Dict[str, Any]:
        """Parse parameter string into key-value dict."""
        # Simple parser: key=value, key2=value2
        params = {}
        for part in param_str.split(','):
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip()] = self._parse_value(value.strip())
        return params
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a parameter value string into appropriate Python type."""
        value_str = value_str.strip()
        
        # Try to parse as Python literal
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass
        
        # Check for variable reference
        if value_str.startswith('$'):
            return {"variable_ref": value_str[1:]}
        
        # Default to string
        return value_str
    
    def _extract_required_tools(self, steps: List[WorkflowStep]) -> List[str]:
        """Extract tool names referenced in workflow steps."""
        tools = set()
        
        for step in steps:
            # Map operation names to tool names
            tool_mapping = self.config.operation_to_tool_map
            if step.operation in tool_mapping:
                tools.add(tool_mapping[step.operation])
            
            # Check parameters for tool references
            for param_value in step.parameters.values():
                if isinstance(param_value, dict) and param_value.get('tool'):
                    tools.add(param_value['tool'])
        
        return list(tools)
    
    def _validate_spec_structure(self, spec: WorkflowSpec) -> bool:
        """Validate that workflow spec has required structure."""
        
        # Must have at least one step
        if not spec.steps:
            return False
        
        # Must have terminal step with return
        if not any(s.is_terminal for s in spec.steps):
            return False
        
        # All operations must be allowed
        allowed_ops = set(self.config.allowed_operations)
        for step in spec.steps:
            if step.operation not in allowed_ops:
                return False
        
        # Tool dependencies must be satisfiable
        # (Would check against available tools)
        
        return True
    
    def _annotate_workflow_spec(
        self,
        spec: WorkflowSpec,
        task_description: str,
        pattern_analysis: Dict[str, Any],
    ) -> WorkflowSpec:
        """Add metadata and optimization hints to workflow spec."""
        
        # Estimate complexity
        spec.complexity = self._estimate_workflow_complexity(spec)
        
        # Add optimization hints
        spec.optimization_hints = self._generate_optimization_hints(spec, pattern_analysis)
        
        # Add versioning
        spec.version = "1.0.0-draft"
        
        # Add provenance
        spec.metadata = {
            "generated_from": task_description,
            "example_count": pattern_analysis.get('example_count', 0),
            "generation_timestamp": time.time(),
        }
        
        return spec
    
    def _estimate_workflow_complexity(self, spec: WorkflowSpec) -> str:
        """Estimate workflow complexity: simple, moderate, complex."""
        step_count = len(spec.steps)
        branch_count = sum(1 for s in spec.steps if s.condition)
        tool_count = len(spec.required_tools)
        
        score = step_count + (branch_count * 2) + (tool_count * 3)
        
        if score < 10:
            return "simple"
        elif score < 25:
            return "moderate"
        else:
            return "complex"
    
    def _generate_optimization_hints(
        self,
        spec: WorkflowSpec,
        pattern_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate hints for workflow optimizer."""
        hints = []
        
        # Look for loop opportunities
        if self._detect_repetitive_pattern(spec.steps):
            hints.append("consider_loop_vectorization")
        
        # Look for caching opportunities
        if self._detect_repeated_computation(spec.steps):
            hints.append("consider_result_caching")
        
        # Look for parallelization opportunities
        if self._detect_independent_branches(spec.steps):
            hints.append("consider_parallel_execution")
        
        # Look for early termination opportunities
        if self._detect_early_exit_conditions(spec.steps):
            hints.append("consider_early_termination")
        
        return hints
    
    def _detect_repetitive_pattern(self, steps: List[WorkflowStep]) -> bool:
        """Detect if workflow has repetitive operations suitable for looping."""
        # Would analyze step sequence for repetition
        return False  # Placeholder
    
    def _detect_repeated_computation(self, steps: List[WorkflowStep]) -> bool:
        """Detect if workflow recomputes same values."""
        return False  # Placeholder
    
    def _detect_independent_branches(self, steps: List[WorkflowStep]) -> bool:
        """Detect if workflow has branches that can run in parallel."""
        return False  # Placeholder
    
    def _detect_early_exit_conditions(self, steps: List[WorkflowStep]) -> bool:
        """Detect if workflow has conditions that allow early return."""
        return any(s.condition and s.is_terminal for s in steps)
    
    # Helper methods for pattern analysis (simplified)
    def _cluster_input_structures(self, inputs: List[Any]) -> Dict[str, Any]:
        return {"type": "inferred", "structure": "object"}
    
    def _cluster_output_structures(self, outputs: List[Any]) -> Dict[str, Any]:
        return {"type": "inferred", "structure": "object"}
    
    def _extract_common_operations(self, examples: List[Dict[str, Any]]) -> List[str]:
        return ["TRANSFORM", "FILTER", "RETURN"]
    
    def _identify_decision_points(self, examples: List[Dict[str, Any]]) -> List[str]:
        return []
    
    def _infer_success_criteria(self, examples: List[Dict[str, Any]]) -> str:
        return "output_matches_expected_schema"
    
    def _parse_condition(self, line: str) -> Optional[str]:
        """Parse IF condition from line."""
        match = re.search(r'IF\s+(.+?)\s+THEN', line, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _parse_return(self, line: str) -> Optional[str]:
        """Parse RETURN value from line."""
        match = re.search(r'RETURN\s+(.+)', line, re.IGNORECASE)
        return match.group(1).strip() if match else None
