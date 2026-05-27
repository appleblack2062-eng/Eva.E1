"""Dynamic registry for available tools."""

from __future__ import annotations
from typing import Dict, Callable, Any, List

class ToolRegistry:
    """Manages available tools for agent execution."""
    
    def __init__(self, config, sandbox, tool_memory):
        self.config = config
        self.sandbox = sandbox
        self.tool_memory = tool_memory
        self._tools: Dict[str, Callable] = {}
        self._register_builtins()
    
    def _register_builtins(self):
        """Register default safe tools."""
        self._tools["filter_data"] = lambda data, **kw: [d for d in data if all(d.get(k)==v for k,v in kw.items())]
        self._tools["transform_data"] = lambda data, **kw: data  # Placeholder
        self._tools["sort_data"] = lambda data, key: sorted(data, key=lambda x: x.get(key, 0))
    
    def register(self, tool_spec):
        """Register a new dynamically generated tool."""
        # In real impl, compile tool_spec.code and add to dict
        self._tools[tool_spec.name] = None  # Placeholder
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        func = self._tools[tool_name]
        if func is None:
            # Load from memory/sandbox if lazy loaded
            pass
            
        return func(**arguments)
    
    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
    
    def count_registered(self) -> int:
        return len(self._tools)
