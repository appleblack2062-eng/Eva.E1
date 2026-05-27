"""Generates optimized context strings for agents."""

from typing import List, Optional
from .filesystem_graph import FileSystemGraph


class SpatialContextGenerator:
    """
    Generates clean, focused context for agents based on task requirements.
    
    This component intelligently selects relevant files from the workspace
    graph and formats them into concise context strings, reducing token
    usage and improving agent accuracy by eliminating irrelevant information.
    """
    
    def __init__(self, fs_graph: FileSystemGraph):
        self.fs_graph = fs_graph
    
    def generate_clean_map(self, task_description: str) -> str:
        """
        Intelligently select relevant files based on task description
        and generate a clean map.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            Formatted string containing relevant workspace context
        """
        # 1. Keyword extraction from task (simplified)
        keywords = self._extract_keywords(task_description)
        
        # 2. Find candidate files
        candidates = self._find_relevant_files(keywords, task_description)
        
        # 3. If no direct matches, fall back to root structure
        if not candidates:
            candidates = ["root"]  # Show top-level structure
            
        # 4. Generate Graph Context
        return self.fs_graph.get_subgraph_context(candidates, depth=2)
    
    def _extract_keywords(self, task_description: str) -> List[str]:
        """
        Extract meaningful keywords from task description.
        
        Args:
            task_description: Natural language task description
            
        Returns:
            List of significant keywords
        """
        # Simple keyword extraction - can be enhanced with NLP
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
            'or', 'because', 'until', 'while', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
            'which', 'who', 'whom', 'create', 'make', 'build', 'write',
            'add', 'update', 'fix', 'implement', 'test', 'run'
        }
        
        words = task_description.lower().split()
        keywords = [
            word.strip('.,!?;:"\'') 
            for word in words 
            if len(word) > 3 and word not in stop_words
        ]
        
        return keywords
    
    def _find_relevant_files(self, keywords: List[str], task_description: str) -> List[str]:
        """
        Find files relevant to the task based on keywords.
        
        Args:
            keywords: Extracted keywords from task
            task_description: Original task description
            
        Returns:
            List of relevant file paths
        """
        candidates = []
        scored_files = {}
        
        all_files = self.fs_graph.get_all_files()
        
        for file_path in all_files:
            score = 0
            
            # Score based on filename matching keywords
            filename = file_path.lower().split('/')[-1]
            for keyword in keywords:
                if keyword in filename:
                    score += 3
                elif keyword.replace('_', '') in filename.replace('_', ''):
                    score += 2
            
            # Score based on file extension relevance
            if task_description.lower():
                if 'test' in task_description.lower() and 'test' in filename:
                    score += 5
                if 'api' in task_description.lower() and 'api' in filename:
                    score += 5
                if 'model' in task_description.lower() and 'model' in filename:
                    score += 5
            
            # Score based on dependencies
            file_info = self.fs_graph.get_file_info(file_path)
            if file_info:
                deps = file_info.get('dependencies', [])
                for dep in deps:
                    for keyword in keywords:
                        if keyword in dep.lower():
                            score += 1
            
            if score > 0:
                scored_files[file_path] = score
        
        # Sort by score and take top matches
        sorted_files = sorted(scored_files.items(), key=lambda x: x[1], reverse=True)
        candidates = [f[0] for f in sorted_files[:10]]  # Top 10 matches
        
        return candidates
    
    def generate_worker_context(
        self, 
        required_files: List[str], 
        task_type: str,
        include_dependencies: bool = True
    ) -> str:
        """
        Generate optimized context specifically for a worker agent.
        
        This creates a minimal, focused context containing only the files
        the worker needs to complete its assigned task.
        
        Args:
            required_files: List of file paths the worker needs
            task_type: Type of task (e.g., 'coder', 'tester', 'analyst')
            include_dependencies: Whether to include dependency files
            
        Returns:
            Formatted context string for the worker
        """
        context_parts = []
        
        # Add workspace overview
        context_parts.append("=== WORKSPACE STRUCTURE ===")
        context_parts.append(self.fs_graph.get_directory_structure(max_depth=2))
        context_parts.append("")
        
        # Add specific file context
        context_parts.append("=== RELEVANT FILES ===")
        context_parts.append(self.fs_graph.get_subgraph_context(required_files, depth=1 if include_dependencies else 0))
        
        # Add task-specific guidance
        context_parts.append("=== TASK TYPE ===")
        context_parts.append(f"Role: {task_type}")
        
        if task_type.lower() == 'coder':
            context_parts.append("Focus: Write clean, efficient code that integrates with existing structure.")
        elif task_type.lower() == 'tester':
            context_parts.append("Focus: Create comprehensive tests covering edge cases.")
        elif task_type.lower() == 'analyst':
            context_parts.append("Focus: Analyze code structure and identify patterns or issues.")
        
        return "\n".join(context_parts)
    
    def get_file_content_context(
        self, 
        file_paths: List[str], 
        max_lines_per_file: int = 100
    ) -> str:
        """
        Get actual file contents for context (with line limits).
        
        Args:
            file_paths: List of file paths to include
            max_lines_per_file: Maximum lines to include per file
            
        Returns:
            Formatted string with file contents
        """
        context_lines = ["=== FILE CONTENTS ==="]
        
        for file_path in file_paths:
            abs_path = self.fs_graph.root_path / file_path
            
            if not abs_path.exists():
                context_lines.append(f"[File not found: {file_path}]")
                continue
            
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                    
                    if len(lines) > max_lines_per_file:
                        content = '\n'.join(lines[:max_lines_per_file])
                        content += f"\n... ({len(lines) - max_lines_per_file} more lines)"
                    
                    context_lines.append(f"\n--- {file_path} ---")
                    context_lines.append(content)
            except (IOError, OSError, UnicodeDecodeError) as e:
                context_lines.append(f"[Error reading {file_path}: {str(e)}]")
        
        return "\n".join(context_lines)
