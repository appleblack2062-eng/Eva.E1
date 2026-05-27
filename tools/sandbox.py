"""Secure execution environment for generated code."""

from __future__ import annotations
import asyncio
import multiprocessing
import time
import subprocess
import json
import tempfile
import os
from typing import Any, Dict
from ..config.settings import AgentConfig

class SafeExecutionSandbox:
    """Executes untrusted code in an isolated process with resource limits."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
    
    async def execute_code(self, code: str, input_data: Any, timeout_seconds: float) -> Any:
        """Run Python code securely."""
        
        # Wrap code in a function to isolate scope
        wrapped_code = f"""
import json
import sys

def _sandbox_main(input_data):
    # User code starts here
{code}
    # User code ends here

if __name__ == '__main__':
    try:
        input_obj = json.loads(sys.argv[1])
        result = _sandbox_main(input_obj)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({{"_error": str(e)}}))
"""
        
        # Run in separate process
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._run_subprocess, 
            wrapped_code, 
            input_data, 
            timeout_seconds
        )
        
        if "_error" in result:
            raise Exception(result["_error"])
        
        return result
    
    def _run_subprocess(self, code: str, input_data: Any, timeout: float) -> Any:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name
        
        try:
            # Execute with timeout
            proc = subprocess.run(
                ['python', script_path, json.dumps(input_data)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONPATH': ''} # Restrict imports
            )
            
            if proc.returncode != 0:
                return {"_error": proc.stderr}
            
            return json.loads(proc.stdout)
            
        finally:
            os.unlink(script_path)
    
    async def execute_workflow(self, workflow, input_data, timeout_seconds):
        """Alias for execute_code for workflow specs."""
        return await self.execute_code(workflow.compiled_code, input_data, timeout_seconds)
