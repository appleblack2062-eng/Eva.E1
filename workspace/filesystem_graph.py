"""Real-time graph representation of the agent's isolated workspace."""

from __future__ import annotations
import networkx as nx
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class FileNode:
    """Represents a file or directory node in the workspace graph."""
    path: str
    type: str  # 'file', 'directory'
    size: int
    hash: str
    last_modified: float
    dependencies: List[str] = field(default_factory=list)  # Imports/Requires


class FileSystemGraph:
    """
    Maintains a directed graph of the workspace.
    
    Nodes = Files/Dirs. Edges = 'contains' or 'imports'.
    Updates in real-time via file system watchers or manual sync.
    
    This provides spatial awareness for agents, allowing them to navigate
    the workspace structure rather than searching blindly for files.
    """
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.graph = nx.DiGraph()
        self._file_hashes: Dict[str, str] = {}
        self._build_initial_graph()
    
    def _build_initial_graph(self):
        """Scan directory and build initial graph."""
        if not self.root_path.exists():
            self.root_path.mkdir(parents=True, exist_ok=True)
        
        for root, dirs, files in os.walk(self.root_path):
            dir_path = Path(root)
            rel_dir = dir_path.relative_to(self.root_path)
            dir_node = str(rel_dir) if str(rel_dir) != "." else "root"
            
            # Add Directory Node
            if not self.graph.has_node(dir_node):
                self.graph.add_node(
                    dir_node, 
                    type='directory', 
                    path=str(dir_path)
                )
            
            # Link to parent
            if rel_dir.parent != rel_dir:
                parent_node = str(rel_dir.parent) if str(rel_dir.parent) != "." else "root"
                if not self.graph.has_node(parent_node):
                    self.graph.add_node(parent_node, type='directory')
                self.graph.add_edge(parent_node, dir_node, relation='contains')

            for file in files:
                file_path = dir_path / file
                rel_file = str(file_path.relative_to(self.root_path))
                
                # Compute hash for change detection
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                except (IOError, OSError):
                    continue
                
                self.graph.add_node(
                    rel_file, 
                    type='file', 
                    path=str(file_path), 
                    hash=file_hash
                )
                self.graph.add_edge(dir_node, rel_file, relation='contains')
                self._file_hashes[rel_file] = file_hash

    def update_node(self, file_path: str):
        """
        Update graph node if file has changed.
        
        Args:
            file_path: Relative path to the file from root
        """
        abs_path = self.root_path / file_path
        
        if not abs_path.exists():
            if self.graph.has_node(file_path):
                self.graph.remove_node(file_path)
                if file_path in self._file_hashes:
                    del self._file_hashes[file_path]
            return

        try:
            with open(abs_path, 'rb') as f:
                new_hash = hashlib.md5(f.read()).hexdigest()
        except (IOError, OSError):
            return
        
        if self._file_hashes.get(file_path) != new_hash:
            # File changed: Update hash and re-analyze dependencies
            if self.graph.has_node(file_path):
                self.graph.nodes[file_path]['hash'] = new_hash
            else:
                # New file, add it to the graph
                rel_dir = str(abs_path.parent.relative_to(self.root_path))
                dir_node = rel_dir if rel_dir != "." else "root"
                
                if not self.graph.has_node(dir_node):
                    self.graph.add_node(dir_node, type='directory')
                
                self.graph.add_node(
                    file_path, 
                    type='file', 
                    path=str(abs_path), 
                    hash=new_hash
                )
                self.graph.add_edge(dir_node, file_path, relation='contains')
            
            self._file_hashes[file_path] = new_hash
            
            # Trigger dependency re-analysis
            self._update_dependencies(file_path, abs_path)

    def _update_dependencies(self, rel_path: str, abs_path: Path):
        """Update import dependencies for a file."""
        from .code_analyzer import CodeAnalyzer
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, OSError, UnicodeDecodeError):
            return
        
        imports = CodeAnalyzer.extract_imports(content)
        
        # Store dependencies in node
        if self.graph.has_node(rel_path):
            self.graph.nodes[rel_path]['dependencies'] = imports
            
        # Create edges to imported modules (if they exist in workspace)
        for imp in imports:
            # Try to find matching file in workspace
            potential_matches = self.find_files_by_pattern(f"*{imp}*")
            for match in potential_matches:
                if match != rel_path:
                    self.graph.add_edge(rel_path, match, relation='imports')

    def get_subgraph_context(self, focus_paths: List[str], depth: int = 1) -> str:
        """
        Generate a clean, text-based graph context for an agent.
        
        Only includes relevant files and their immediate neighbors,
        providing focused context to reduce token usage and improve accuracy.
        
        Args:
            focus_paths: List of file paths to center the context around
            depth: How many hops to include (default 1)
            
        Returns:
            Formatted string representation of the subgraph
        """
        context_lines = ["--- WORKSPACE CONTEXT ---"]
        visited = set()
        
        for path in focus_paths:
            if path not in self.graph:
                continue
            
            # Get ego-network (node + neighbors)
            neighbors = list(self.graph.neighbors(path))
            predecessors = list(self.graph.predecessors(path))
            related = set(neighbors + predecessors)
            
            context_lines.append(f"NODE: {path}")
            context_lines.append(f"  TYPE: {self.graph.nodes[path].get('type', 'unknown')}")
            
            if self.graph.nodes[path].get('type') == 'file':
                deps = self.graph.nodes[path].get('dependencies', [])
                if deps:
                    context_lines.append(f"  DEPENDENCIES: {', '.join(deps)}")
            
            if related:
                context_lines.append("  RELATED:")
                for r in sorted(related):
                    if r not in visited:
                        rel_type = self.graph.get_edge_data(path, r, {}).get(
                            'relation', 
                            self.graph.get_edge_data(r, path, {}).get('relation', 'linked')
                        )
                        context_lines.append(f"    - {r} [{rel_type}]")
                        visited.add(r)
            context_lines.append("")
            
        return "\n".join(context_lines)

    def find_files_by_pattern(self, pattern: str) -> List[str]:
        """
        Simple glob-like search on graph nodes.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            List of matching file paths
        """
        import fnmatch
        return [
            n for n in self.graph.nodes 
            if fnmatch.fnmatch(n, pattern) and self.graph.nodes[n].get('type') == 'file'
        ]

    def get_directory_structure(self, dir_path: str = "root", max_depth: int = 3) -> str:
        """
        Get a tree-like representation of directory structure.
        
        Args:
            dir_path: Starting directory node
            max_depth: Maximum depth to traverse
            
        Returns:
            Tree-formatted string of directory structure
        """
        lines = []
        
        def _traverse(node: str, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            children = [
                n for n in self.graph.successors(node)
                if self.graph.get_edge_data(node, n, {}).get('relation') == 'contains'
            ]
            
            for i, child in enumerate(sorted(children)):
                is_last = (i == len(children) - 1)
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{child}")
                
                if self.graph.nodes[child].get('type') == 'directory':
                    extension = "    " if is_last else "│   "
                    _traverse(child, prefix + extension, depth + 1)
        
        if dir_path in self.graph:
            lines.append(dir_path)
            _traverse(dir_path)
        
        return "\n".join(lines)

    def get_all_files(self) -> List[str]:
        """Return list of all file nodes in the graph."""
        return [
            n for n in self.graph.nodes 
            if self.graph.nodes[n].get('type') == 'file'
        ]

    def get_file_info(self, file_path: str) -> Optional[Dict[str, any]]:
        """Get detailed information about a specific file."""
        if file_path not in self.graph:
            return None
        
        node_data = dict(self.graph.nodes[file_path])
        return node_data
