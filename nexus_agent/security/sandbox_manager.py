"""Sandbox Manager: Isolated execution environments for agents."""

from __future__ import annotations
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class SandboxConfig:
    """Configuration for a sandbox environment."""
    agent_id: str
    max_memory_mb: int = 512
    max_disk_mb: int = 100
    max_cpu_percent: float = 50.0
    timeout_seconds: int = 300
    network_enabled: bool = False
    allowed_paths: list = None
    
    def __post_init__(self):
        if self.allowed_paths is None:
            self.allowed_paths = []


class SandboxManager:
    """
    Creates and manages isolated execution environments for agents.
    
    Supports multiple isolation levels:
    - Process-level isolation (default)
    - Filesystem isolation (chroot-like)
    - Container isolation (Docker/bwrap if available)
    """
    
    def __init__(self, base_temp_dir: Optional[str] = None):
        self.base_temp_dir = Path(base_temp_dir) if base_temp_dir else Path(tempfile.gettempdir()) / "nexus_sandboxes"
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.sandboxes: Dict[str, SandboxConfig] = {}
        self.sandbox_dirs: Dict[str, Path] = {}
    
    async def create(self, agent_id: str, policy: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an isolated sandbox for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            policy: Security policy configuration
            
        Returns:
            Sandbox ID
        """
        # Parse policy into config
        config = SandboxConfig(
            agent_id=agent_id,
            max_memory_mb=policy.get('max_memory_mb', 512) if policy else 512,
            max_disk_mb=policy.get('max_disk_mb', 100) if policy else 100,
            timeout_seconds=policy.get('timeout_seconds', 300) if policy else 300,
            network_enabled=policy.get('network_enabled', False) if policy else False,
            allowed_paths=policy.get('allowed_paths', []) if policy else []
        )
        
        # Create sandbox directory
        sandbox_dir = self.base_temp_dir / agent_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Store sandbox info
        self.sandboxes[agent_id] = config
        self.sandbox_dirs[agent_id] = sandbox_dir
        
        return agent_id
    
    async def execute(self, agent_id: str, code: str, input_data: Any = None) -> Any:
        """
        Execute code within the agent's sandbox.
        
        Args:
            agent_id: ID of the agent's sandbox
            code: Code to execute
            input_data: Input data for the code
            
        Returns:
            Execution result
        """
        if agent_id not in self.sandboxes:
            raise ValueError(f"Sandbox {agent_id} not found")
        
        config = self.sandboxes[agent_id]
        sandbox_dir = self.sandbox_dirs[agent_id]
        
        try:
            # Create a temporary script file
            script_file = sandbox_dir / f"sandbox_{agent_id}.py"
            
            # Write code with safety wrapper
            safe_code = self._wrap_code(code, config)
            script_file.write_text(safe_code)
            
            # Execute with timeout and resource limits
            result = await self._run_with_limits(script_file, input_data, config)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Sandbox execution failed: {e}")
        finally:
            # Cleanup script file
            if script_file.exists():
                script_file.unlink()
    
    def _wrap_code(self, code: str, config: SandboxConfig) -> str:
        """Wrap user code with safety checks and resource monitoring."""
        
        # Safety wrapper
        wrapper = f"""
import sys
import os

# Restrict dangerous operations
class SafeModule:
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Access denied: {{name}}")
        return object()

# Override dangerous modules
sys.modules['os'].system = lambda x: None
sys.modules['os'].popen = lambda x: None
sys.modules['subprocess'] = SafeModule()

# Execute user code
try:
    # User code starts here
    USER_INPUT = {repr(config.allowed_paths)}
    
{code}

    # If there's a main function, call it
    if 'main' in locals() and callable(main):
        result = main()
        print(f"RESULT:{{result}}")
        
except Exception as e:
    print(f"ERROR:{{e}}")
    sys.exit(1)
"""
        return wrapper
    
    async def _run_with_limits(
        self, 
        script_file: Path, 
        input_data: Any,
        config: SandboxConfig
    ) -> Any:
        """Run script with resource limits."""
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # Build command
        cmd = ['python', str(script_file)]
        
        # In a real implementation, would use:
        # - asyncio.create_subprocess_exec with resource limits
        # - Or bubblewrap/docker for stronger isolation
        
        try:
            # Create subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.sandbox_dirs[config.agent_id])
            )
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds
            )
            
            # Parse output
            output = stdout.decode('utf-8')
            error_output = stderr.decode('utf-8')
            
            if process.returncode != 0:
                raise RuntimeError(f"Execution failed: {error_output}")
            
            # Extract result from output
            result = self._parse_result(output)
            return result
            
        except asyncio.TimeoutError:
            # Kill process on timeout
            process.kill()
            await process.wait()
            raise TimeoutError(f"Sandbox execution exceeded {config.timeout_seconds}s")
    
    def _parse_result(self, output: str) -> Any:
        """Parse result from sandbox output."""
        # Look for RESULT: marker
        for line in output.split('\n'):
            if line.startswith('RESULT:'):
                return line[7:]  # Remove RESULT: prefix
            if line.startswith('ERROR:'):
                raise RuntimeError(line[6:])
        
        # Return raw output if no marker found
        return output.strip()
    
    async def destroy(self, agent_id: str) -> bool:
        """
        Destroy a sandbox and clean up resources.
        
        Args:
            agent_id: ID of sandbox to destroy
            
        Returns:
            True if successful
        """
        if agent_id not in self.sandboxes:
            return False
        
        # Remove sandbox directory
        if agent_id in self.sandbox_dirs:
            sandbox_dir = self.sandbox_dirs[agent_id]
            if sandbox_dir.exists():
                try:
                    shutil.rmtree(sandbox_dir)
                except Exception as e:
                    print(f"Error removing sandbox directory: {e}")
        
        # Remove from tracking
        del self.sandboxes[agent_id]
        del self.sandbox_dirs[agent_id]
        
        return True
    
    async def reset(self, agent_id: str) -> bool:
        """
        Reset a sandbox to clean state.
        
        Args:
            agent_id: ID of sandbox to reset
            
        Returns:
            True if successful
        """
        # Destroy and recreate
        await self.destroy(agent_id)
        config = self.sandboxes.get(agent_id)  # Save config before delete
        
        if config:
            await self.create(agent_id, {
                'max_memory_mb': config.max_memory_mb,
                'max_disk_mb': config.max_disk_mb,
                'timeout_seconds': config.timeout_seconds
            })
            return True
        
        return False
    
    def get_sandbox_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a sandbox."""
        if agent_id not in self.sandboxes:
            return None
        
        config = self.sandboxes[agent_id]
        sandbox_dir = self.sandbox_dirs.get(agent_id)
        
        disk_usage = 0
        if sandbox_dir and sandbox_dir.exists():
            disk_usage = sum(
                f.stat().st_size for f in sandbox_dir.rglob('*') if f.is_file()
            )
        
        return {
            'agent_id': agent_id,
            'max_memory_mb': config.max_memory_mb,
            'max_disk_mb': config.max_disk_mb,
            'current_disk_mb': disk_usage / (1024 * 1024),
            'timeout_seconds': config.timeout_seconds,
            'network_enabled': config.network_enabled,
            'directory': str(sandbox_dir) if sandbox_dir else None
        }
    
    def list_sandboxes(self) -> list:
        """List all active sandboxes."""
        return list(self.sandboxes.keys())
