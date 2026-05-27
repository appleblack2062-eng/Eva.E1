"""Workspace API for querying and managing the file system graph."""

from typing import List, Dict, Any, Optional
from pathlib import Path
from ..workspace.filesystem_graph import FileSystemGraph


class WorkspaceAPI:
    """
    RESTful-style API for workspace graph operations.
    
    Provides endpoints for:
    - Querying workspace structure
    - Finding files by pattern or content
    - Getting file relationships and dependencies
    - Updating workspace state
    """
    
    def __init__(self, fs_graph: FileSystemGraph):
        """
        Initialize workspace API.
        
        Args:
            fs_graph: File system graph instance
        """
        self.fs_graph = fs_graph
    
    def get_structure(self, path: str = "root", depth: int = 3) -> Dict[str, Any]:
        """
        Get directory structure as a tree.
        
        Args:
            path: Starting path (default: root)
            depth: Maximum depth to traverse
            
        Returns:
            Dictionary representing directory tree
        """
        if path not in self.fs_graph.graph:
            return {"error": f"Path not found: {path}"}
        
        node_type = self.fs_graph.graph.nodes[path].get('type', 'unknown')
        
        result = {
            "path": path,
            "type": node_type,
            "children": []
        }
        
        if node_type == 'directory':
            children = [
                n for n in self.fs_graph.graph.successors(path)
                if self.fs_graph.graph.get_edge_data(path, n, {}).get('relation') == 'contains'
            ]
            
            for child in sorted(children)[:50]:  # Limit children
                child_info = self.fs_graph.graph.nodes[child]
                result["children"].append({
                    "name": child,
                    "type": child_info.get('type', 'unknown'),
                    "hash": child_info.get('hash', '')[:8] if child_info.get('type') == 'file' else None
                })
        
        return result
    
    def find_files(
        self, 
        pattern: Optional[str] = None,
        file_type: Optional[str] = None,
        contains_import: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find files matching criteria.
        
        Args:
            pattern: Glob-like pattern (e.g., "*.py")
            file_type: Filter by type ('file' or 'directory')
            contains_import: Find files importing specific module
            
        Returns:
            List of matching file info dictionaries
        """
        results = []
        
        for node in self.fs_graph.graph.nodes:
            node_info = self.fs_graph.graph.nodes[node]
            
            # Filter by type
            if file_type and node_info.get('type') != file_type:
                continue
            
            # Filter by pattern
            if pattern:
                import fnmatch
                if not fnmatch.fnmatch(node, pattern):
                    continue
            
            # Filter by import
            if contains_import:
                deps = node_info.get('dependencies', [])
                if not any(contains_import in dep for dep in deps):
                    continue
            
            results.append({
                "path": node,
                "type": node_info.get('type'),
                "hash": node_info.get('hash', ''),
                "dependencies": node_info.get('dependencies', [])
            })
        
        return results
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information dictionary or None
        """
        if file_path not in self.fs_graph.graph:
            return None
        
        node_info = dict(self.fs_graph.graph.nodes[file_path])
        
        # Add relationship info
        incoming = list(self.fs_graph.graph.predecessors(file_path))
        outgoing = list(self.fs_graph.graph.successors(file_path))
        
        node_info['imported_by'] = [
            p for p in incoming 
            if self.fs_graph.graph.get_edge_data(p, file_path, {}).get('relation') == 'imports'
        ]
        node_info['imports'] = [
            s for s in outgoing 
            if self.fs_graph.graph.get_edge_data(file_path, s, {}).get('relation') == 'imports'
        ]
        
        return node_info
    
    def get_dependencies(self, file_path: str, recursive: bool = False) -> List[str]:
        """
        Get file dependencies.
        
        Args:
            file_path: Path to the file
            recursive: Whether to get transitive dependencies
            
        Returns:
            List of dependency file paths
        """
        if file_path not in self.fs_graph.graph:
            return []
        
        if recursive:
            # BFS to find all transitive dependencies
            visited = set()
            queue = [file_path]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                # Get imports
                deps = self.fs_graph.graph.nodes[current].get('dependencies', [])
                for dep in deps:
                    # Try to find matching file in workspace
                    matches = self.fs_graph.find_files_by_pattern(f"*{dep}*")
                    queue.extend(matches)
            
            visited.discard(file_path)
            return list(visited)
        else:
            return self.fs_graph.graph.nodes[file_path].get('dependencies', [])
    
    def get_context(
        self, 
        focus_paths: List[str], 
        depth: int = 1,
        include_content: bool = False
    ) -> Dict[str, Any]:
        """
        Get workspace context for agents.
        
        Args:
            focus_paths: Files to center context around
            depth: How many hops to include
            include_content: Whether to include file contents
            
        Returns:
            Context dictionary with graph and optional content
        """
        context_str = self.fs_graph.get_subgraph_context(focus_paths, depth)
        
        result = {
            "context_text": context_str,
            "focus_paths": focus_paths,
            "depth": depth,
            "files": {}
        }
        
        if include_content:
            for path in focus_paths:
                abs_path = self.fs_graph.root_path / path
                if abs_path.exists():
                    try:
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            result["files"][path] = f.read()
                    except (IOError, OSError, UnicodeDecodeError):
                        result["files"][path] = "[Error reading file]"
        
        return result
    
    def update_file(self, file_path: str) -> Dict[str, Any]:
        """
        Manually trigger update for a file.
        
        Args:
            file_path: Path to the file to update
            
        Returns:
            Update result dictionary
        """
        old_hash = self.fs_graph._file_hashes.get(file_path, '')
        
        self.fs_graph.update_node(file_path)
        
        new_hash = self.fs_graph._file_hashes.get(file_path, '')
        
        return {
            "path": file_path,
            "updated": old_hash != new_hash,
            "old_hash": old_hash[:8] if old_hash else None,
            "new_hash": new_hash[:8] if new_hash else None
        }
    
    def refresh(self) -> Dict[str, Any]:
        """
        Refresh the entire workspace graph.
        
        Returns:
            Refresh result dictionary
        """
        old_count = len(self.fs_graph.graph.nodes)
        
        # Rebuild graph
        self.fs_graph.graph.clear()
        self.fs_graph._file_hashes.clear()
        self.fs_graph._build_initial_graph()
        
        new_count = len(self.fs_graph.graph.nodes)
        
        return {
            "status": "refreshed",
            "old_node_count": old_count,
            "new_node_count": new_count,
            "change": new_count - old_count
        }
    
    def stats(self) -> Dict[str, Any]:
        """
        Get workspace statistics.
        
        Returns:
            Statistics dictionary
        """
        files = [n for n in self.fs_graph.graph.nodes 
                if self.fs_graph.graph.nodes[n].get('type') == 'file']
        dirs = [n for n in self.fs_graph.graph.nodes 
               if self.fs_graph.graph.nodes[n].get('type') == 'directory']
        
        # Count edges by type
        contains_edges = sum(
            1 for u, v, data in self.fs_graph.graph.edges(data=True)
            if data.get('relation') == 'contains'
        )
        import_edges = sum(
            1 for u, v, data in self.fs_graph.graph.edges(data=True)
            if data.get('relation') == 'imports'
        )
        
        return {
            "total_nodes": len(self.fs_graph.graph.nodes),
            "files": len(files),
            "directories": len(dirs),
            "total_edges": len(self.fs_graph.graph.edges),
            "contains_edges": contains_edges,
            "import_edges": import_edges,
            "root_path": str(self.fs_graph.root_path)
        }
