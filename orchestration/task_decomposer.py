"""Breaks high-level goals into atomic, assignable steps."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TaskStep(BaseModel):
    """Represents an atomic step in a larger task."""
    
    id: str = Field(..., description="Unique identifier for this step")
    description: str = Field(..., description="Clear instruction for what to do")
    required_files: List[str] = Field(
        default_factory=list, 
        description="List of file paths from context needed"
    )
    expected_output_type: str = Field(
        ..., 
        description="Expected output type: 'code', 'text', 'json', etc."
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of step IDs that must finish first"
    )
    priority: int = Field(
        default=0,
        description="Priority level (higher = more urgent)"
    )
    estimated_complexity: str = Field(
        default="medium",
        description="Estimated complexity: 'low', 'medium', 'high'"
    )


class TaskDecomposer:
    """
    Decomposes complex goals into atomic, executable steps.
    
    Uses LLM to analyze the goal and workspace context, then generates
    a structured plan with dependencies and resource requirements.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def decompose(
        self, 
        goal: str, 
        workspace_context: str,
        max_steps: int = 20
    ) -> List[TaskStep]:
        """
        Use LLM to break goal into atomic steps.
        
        Args:
            goal: High-level goal description
            workspace_context: Current workspace state and structure
            max_steps: Maximum number of steps to generate
            
        Returns:
            List of TaskStep objects representing the execution plan
        """
        prompt = f"""
You are an expert Task Planner for a multi-agent software development system.
Your job is to break down complex goals into small, atomic, executable steps.

WORKSPACE CONTEXT:
{workspace_context}

GOAL: {goal}

INSTRUCTIONS:
1. Break the goal into the smallest possible atomic steps
2. Each step should be executable by a single specialist agent
3. Specify exact file paths needed for each step
4. Define clear dependencies between steps
5. Keep steps focused on ONE responsibility
6. Consider parallel execution opportunities (steps without dependencies)
7. Return ONLY valid JSON, no additional text

Return a JSON array of steps. Each step must have:
- id: unique string identifier (use simple numbers like "1", "2", "3")
- description: clear, actionable instruction
- required_files: array of file paths from the workspace context
- expected_output_type: one of 'code', 'text', 'json', 'analysis'
- dependencies: array of step IDs that must complete first (empty for independent steps)
- priority: integer (0=normal, 1=high, 2=critical)
- estimated_complexity: one of 'low', 'medium', 'high'

Example format:
[
  {{
    "id": "1",
    "description": "Create main application file",
    "required_files": [],
    "expected_output_type": "code",
    "dependencies": [],
    "priority": 1,
    "estimated_complexity": "medium"
  }},
  {{
    "id": "2",
    "description": "Write unit tests for main application",
    "required_files": ["app.py"],
    "expected_output_type": "code",
    "dependencies": ["1"],
    "priority": 0,
    "estimated_complexity": "low"
  }}
]

Generate {max_steps} or fewer steps to accomplish the goal.
"""
        
        try:
            response = await self.llm.generate(
                prompt=prompt, 
                response_format="json"
            )
            
            # Parse response
            if hasattr(response, 'data'):
                steps_data = response.data
            elif hasattr(response, 'content'):
                import json
                steps_data = json.loads(response.content)
            else:
                steps_data = response
            
            # Validate and create TaskStep objects
            steps = []
            for step_dict in steps_data:
                try:
                    step = TaskStep(**step_dict)
                    steps.append(step)
                except Exception as e:
                    print(f"[TaskDecomposer] Warning: Invalid step format: {e}")
                    continue
            
            return steps
            
        except Exception as e:
            print(f"[TaskDecomposer] Error decomposing task: {e}")
            # Fallback: create a single step for the entire goal
            return [
                TaskStep(
                    id="1",
                    description=goal,
                    required_files=[],
                    expected_output_type="code",
                    dependencies=[]
                )
            ]
    
    async def refine_plan(
        self, 
        steps: List[TaskStep], 
        feedback: str
    ) -> List[TaskStep]:
        """
        Refine an existing plan based on feedback.
        
        Args:
            steps: Current plan steps
            feedback: Feedback about what needs improvement
            
        Returns:
            Refined list of TaskStep objects
        """
        steps_json = [step.dict() for step in steps]
        
        prompt = f"""
You are refining a task plan based on feedback.

CURRENT PLAN:
{steps_json}

FEEDBACK:
{feedback}

Revise the plan to address the feedback. You can:
- Add new steps
- Remove unnecessary steps
- Modify step descriptions
- Adjust dependencies
- Change priorities

Return the complete revised plan as a JSON array.
"""
        
        try:
            response = await self.llm.generate(prompt=prompt, response_format="json")
            
            if hasattr(response, 'data'):
                refined_data = response.data
            elif hasattr(response, 'content'):
                import json
                refined_data = json.loads(response.content)
            else:
                refined_data = response
            
            return [TaskStep(**step_dict) for step_dict in refined_data]
            
        except Exception as e:
            print(f"[TaskDecomposer] Error refining plan: {e}")
            return steps  # Return original on error
    
    @staticmethod
    def topological_sort(steps: List[TaskStep]) -> List[TaskStep]:
        """
        Sort steps in topological order based on dependencies.
        
        Args:
            steps: List of TaskStep objects
            
        Returns:
            List of TaskStep objects in execution order
        """
        # Build adjacency list and in-degree count
        step_map = {step.id: step for step in steps}
        in_degree = {step.id: len(step.dependencies) for step in steps}
        
        # Find all steps with no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority (higher first)
            queue.sort(key=lambda x: step_map[x].priority, reverse=True)
            current_id = queue.pop(0)
            current_step = step_map[current_id]
            result.append(current_step)
            
            # Reduce in-degree for dependent steps
            for step in steps:
                if current_id in step.dependencies:
                    in_degree[step.id] -= 1
                    if in_degree[step.id] == 0:
                        queue.append(step.id)
        
        # Check for cycles
        if len(result) != len(steps):
            print("[TaskDecomposer] Warning: Circular dependency detected!")
            # Return original order if cycle detected
            return steps
        
        return result
