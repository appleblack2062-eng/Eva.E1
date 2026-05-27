"""Factory for creating ephemeral Worker Agents."""

from typing import Dict, Any
from ..agents.worker_instance import WorkerInstance


class WorkerFactory:
    """
    Factory for creating specialized, ephemeral worker agents.
    
    Each worker is created with a highly optimized prompt containing
    ONLY the necessary context for its specific task, reducing token
    costs and improving focus.
    """
    
    def __init__(self, llm_client, config):
        self.llm = llm_client
        self.config = config
    
    def create_worker(
        self, 
        worker_id: str, 
        role: str, 
        task_description: str, 
        context: str, 
        output_type: str,
        additional_instructions: str = ""
    ) -> WorkerInstance:
        """
        Instantiate a worker with a highly optimized prompt.
        
        Args:
            worker_id: Unique identifier for this worker
            role: Specialist role (e.g., 'Coder', 'Tester', 'Analyst')
            task_description: Specific task to perform
            context: Workspace context relevant to this task
            output_type: Expected output format
            additional_instructions: Optional extra instructions
            
        Returns:
            Configured WorkerInstance ready to execute
        """
        
        system_prompt = self._build_system_prompt(
            role=role,
            task_description=task_description,
            context=context,
            output_type=output_type,
            additional_instructions=additional_instructions
        )
        
        return WorkerInstance(
            worker_id=worker_id,
            llm=self.llm,
            system_prompt=system_prompt,
            config=self.config
        )
    
    def _build_system_prompt(
        self,
        role: str,
        task_description: str,
        context: str,
        output_type: str,
        additional_instructions: str = ""
    ) -> str:
        """
        Build an optimized system prompt for a worker.
        
        The prompt is structured to maximize focus and minimize distractions.
        """
        
        role_descriptions = {
            'Coder': """
You are an expert Software Engineer specializing in clean, efficient code.
- Write production-ready code that follows best practices
- Ensure code integrates seamlessly with existing structures
- Include appropriate error handling and edge cases
- Add concise, meaningful comments where needed
- Follow the project's existing style and conventions
""",
            'Tester': """
You are an expert QA Engineer specializing in comprehensive testing.
- Create thorough test suites covering normal and edge cases
- Follow testing best practices for the language/framework
- Ensure tests are independent, repeatable, and maintainable
- Include both positive and negative test cases
- Mock external dependencies appropriately
""",
            'Analyst': """
You are an expert Code Analyst specializing in understanding complex systems.
- Analyze code structure, patterns, and potential issues
- Identify dependencies and relationships between components
- Provide clear, actionable insights
- Consider performance, security, and maintainability
- Document findings systematically
""",
            'Architect': """
You are an expert Software Architect specializing in system design.
- Design scalable, maintainable system architectures
- Consider trade-offs between different approaches
- Ensure proper separation of concerns
- Plan for future extensibility
- Document architectural decisions clearly
""",
            'Documenter': """
You are an expert Technical Writer specializing in clear documentation.
- Write comprehensive, easy-to-understand documentation
- Include examples and usage scenarios
- Maintain consistent terminology and style
- Cover both high-level concepts and detailed APIs
- Keep documentation up-to-date with code changes
""",
            'Debugger': """
You are an expert Debugging Specialist specializing in finding and fixing issues.
- Systematically identify root causes of problems
- Consider edge cases and race conditions
- Propose minimal, targeted fixes
- Verify fixes don't introduce new issues
- Document the debugging process and solution
"""
        }
        
        base_role_desc = role_descriptions.get(role, f"""
You are an expert {role}.
- Perform your task with precision and expertise
- Follow best practices for your domain
- Deliver high-quality results
""")
        
        output_format_instructions = {
            'code': """
OUTPUT FORMAT:
- Return ONLY the code, no explanations unless requested
- Ensure code is complete and runnable
- Include necessary imports and dependencies
- Use proper formatting and indentation
""",
            'text': """
OUTPUT FORMAT:
- Provide clear, well-structured text
- Use appropriate formatting (headings, lists, etc.)
- Be concise but comprehensive
""",
            'json': """
OUTPUT FORMAT:
- Return valid JSON only
- Ensure proper escaping and formatting
- Include all required fields
- No additional text outside the JSON
""",
            'analysis': """
OUTPUT FORMAT:
- Structure your analysis clearly
- Use sections and bullet points
- Provide evidence for conclusions
- Include recommendations where applicable
"""
        }
        
        output_instr = output_format_instructions.get(
            output_type.lower(), 
            f"OUTPUT FORMAT: {output_type}"
        )
        
        prompt = f"""
{base_role_desc}
YOUR SPECIFIC TASK:
{task_description}

WORKSPACE CONTEXT:
{context}

{output_instr}

CRITICAL INSTRUCTIONS:
1. Do NOT deviate from the assigned task
2. Do NOT ask for clarification - make reasonable assumptions if needed
3. If writing code, ensure it fits the existing structure shown in CONTEXT
4. Be concise and focused - avoid unnecessary verbosity
5. Double-check your work before completing
6. If the task involves modifying files, preserve existing functionality
7. Follow all conventions visible in the provided context
"""
        
        if additional_instructions:
            prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{additional_instructions}"
        
        return prompt
    
    def create_parallel_workers(
        self,
        tasks: list[Dict[str, Any]],
        base_context: str
    ) -> Dict[str, WorkerInstance]:
        """
        Create multiple workers for parallel execution.
        
        Args:
            tasks: List of task dictionaries with keys:
                   - id: worker ID
                   - role: specialist role
                   - description: task description
                   - output_type: expected output
            base_context: Common workspace context
            
        Returns:
            Dictionary mapping worker IDs to WorkerInstance objects
        """
        workers = {}
        
        for task in tasks:
            worker = self.create_worker(
                worker_id=task['id'],
                role=task.get('role', 'Generalist'),
                task_description=task['description'],
                context=base_context,
                output_type=task.get('output_type', 'code'),
                additional_instructions=task.get('instructions', '')
            )
            workers[task['id']] = worker
        
        return workers
