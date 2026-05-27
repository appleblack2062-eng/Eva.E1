"""Parses code files to extract import dependencies for the Graph."""

import ast
from typing import List, Set, Dict, Optional


class CodeAnalyzer:
    """
    Static code analyzer for extracting structural information from source files.
    
    Currently supports Python files. Can be extended for other languages.
    """
    
    @staticmethod
    def extract_imports(file_content: str) -> List[str]:
        """
        Extract import statements from Python code.
        
        Args:
            file_content: Source code content as string
            
        Returns:
            List of imported module names
        """
        try:
            tree = ast.parse(file_content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        
            return list(set(imports))  # Remove duplicates
        except SyntaxError:
            return []
    
    @staticmethod
    def extract_functions(file_content: str) -> List[Dict[str, any]]:
        """
        Extract function definitions from Python code.
        
        Args:
            file_content: Source code content as string
            
        Returns:
            List of dicts with function name, line number, and arguments
        """
        try:
            tree = ast.parse(file_content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [
                            ast.unparse(d) if hasattr(ast, 'unparse') else str(d)
                            for d in node.decorator_list
                        ]
                    }
                    functions.append(func_info)
                    
            return functions
        except SyntaxError:
            return []
    
    @staticmethod
    def extract_classes(file_content: str) -> List[Dict[str, any]]:
        """
        Extract class definitions from Python code.
        
        Args:
            file_content: Source code content as string
            
        Returns:
            List of dicts with class name, line number, base classes, and methods
        """
        try:
            tree = ast.parse(file_content)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'bases': [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) 
                                 for base in node.bases],
                        'methods': [
                            m.name for m in node.body 
                            if isinstance(m, ast.FunctionDef)
                        ]
                    }
                    classes.append(class_info)
                    
            return classes
        except SyntaxError:
            return []
    
    @staticmethod
    def extract_dependencies(file_content: str, file_path: str) -> Dict[str, List[str]]:
        """
        Extract all types of dependencies from a code file.
        
        Args:
            file_content: Source code content as string
            file_path: Path to the file (for context)
            
        Returns:
            Dictionary with categorized dependencies
        """
        return {
            'imports': CodeAnalyzer.extract_imports(file_content),
            'functions': [f['name'] for f in CodeAnalyzer.extract_functions(file_content)],
            'classes': [c['name'] for c in CodeAnalyzer.extract_classes(file_content)],
        }
    
    @staticmethod
    def get_complexity_metrics(file_content: str) -> Dict[str, int]:
        """
        Calculate basic complexity metrics for a code file.
        
        Args:
            file_content: Source code content as string
            
        Returns:
            Dictionary with complexity metrics
        """
        try:
            tree = ast.parse(file_content)
            
            num_functions = sum(1 for node in ast.walk(tree) 
                               if isinstance(node, ast.FunctionDef))
            num_classes = sum(1 for node in ast.walk(tree) 
                             if isinstance(node, ast.ClassDef))
            num_lines = len(file_content.splitlines())
            
            # Count nesting depth (simplified)
            max_depth = 0
            for node in ast.walk(tree):
                depth = 0
                current = node
                while hasattr(current, 'parent'):
                    depth += 1
                    current = current.parent
                max_depth = max(max_depth, depth)
            
            return {
                'lines': num_lines,
                'functions': num_functions,
                'classes': num_classes,
                'estimated_complexity': num_functions + (num_classes * 2)
            }
        except SyntaxError:
            return {'lines': len(file_content.splitlines()), 'functions': 0, 'classes': 0}
