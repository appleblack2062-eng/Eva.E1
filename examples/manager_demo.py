"""Demonstrates the Manager/Worker architecture with Workspace Awareness."""

import asyncio
from pathlib import Path
from typing import Any

# Import from nexus_agent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import AgentConfig, GlobalConfig
from core.agent_brain import AgentBrain


class MockEmbeddingProvider:
    """Simple mock embedding provider for testing."""
    
    def __init__(self, model_name: str = "mock-embeddings"):
        self.model_name = model_name
    
    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


class MockLLM:
    """Mock LLM client for demonstration."""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> Any:
        self.call_count += 1
        
        # Simulate intelligent responses based on prompt content
        if "Decompose" in prompt or "Task Planner" in prompt:
            # Task decomposition response
            return type('obj', (object,), {
                'data': [
                    {
                        "id": "1",
                        "description": "Create a Python script 'hello.py' that prints hello world",
                        "required_files": [],
                        "expected_output_type": "code",
                        "dependencies": [],
                        "priority": 1,
                        "estimated_complexity": "low"
                    },
                    {
                        "id": "2", 
                        "description": "Create a test file 'test_hello.py' for hello.py",
                        "required_files": ["hello.py"],
                        "expected_output_type": "code",
                        "dependencies": ["1"],
                        "priority": 0,
                        "estimated_complexity": "low"
                    }
                ],
                'content': '',
                'token_count': 50
            })
        elif "expert Coder" in prompt:
            return type('obj', (object,), {
                'content': "def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()",
                'token_count': 30
            })
        elif "expert Tester" in prompt:
            return type('obj', (object,), {
                'content': "import unittest\n\nclass TestHello(unittest.TestCase):\n    def test_print(self):\n        # Test that hello.py runs without error\n        import hello\n        self.assertTrue(hasattr(hello, 'main'))\n\nif __name__ == '__main__':\n    unittest.main()",
                'token_count': 40
            })
        else:
            return type('obj', (object,), {
                'content': "Task completed successfully.",
                'token_count': 10
            })


async def main():
    """Run the Manager/Worker demo."""
    print("=" * 70)
    print("🚀 NexusAgent Pro v2.0 - Manager/Worker Architecture Demo")
    print("=" * 70)
    print()
    
    # Initialize configuration
    config = AgentConfig(agent_id="demo_bot")
    global_config = GlobalConfig()
    
    # Create mock dependencies
    llm = MockLLM()
    embedder = MockEmbeddingProvider()
    
    print("📦 Initializing AgentBrain with workspace awareness...")
    
    # Initialize Brain (this creates the FileSystemGraph and Manager)
    brain = AgentBrain("demo_bot", config, global_config, llm, embedder)
    
    # Create some dummy files in workspace for the graph to pick up
    ws_path = Path(config.base_storage_path) / "demo_bot" / "workspace"
    ws_path.mkdir(parents=True, exist_ok=True)
    
    # Add existing module
    (ws_path / "existing_module.py").write_text("""
def old_function():
    '''An existing function in the codebase.'''
    return "I was here before"

class ExistingClass:
    def __init__(self):
        self.value = 42
""")
    
    # Add a utility file
    (ws_path / "utils.py").write_text("""
def helper_function(x):
    return x * 2

CONSTANT = 100
""")
    
    print(f"✅ Created sample workspace at: {ws_path}")
    print()
    
    # Show workspace stats
    stats = brain.workspace_api.stats()
    print("📊 Initial Workspace Statistics:")
    print(f"   - Total nodes: {stats['total_nodes']}")
    print(f"   - Files: {stats['files']}")
    print(f"   - Directories: {stats['directories']}")
    print(f"   - Edges: {stats['total_edges']}")
    print()
    
    # Show directory structure
    print("📁 Workspace Structure:")
    structure = brain.fs_graph.get_directory_structure(max_depth=2)
    for line in structure.split('\n'):
        print(f"   {line}")
    print()
    
    # Execute a complex goal
    goal = "Create a hello world script and a unit test for it."
    
    print(f"🎯 Goal: {goal}")
    print()
    print("⚙️  Executing with Manager/Worker architecture...")
    print("-" * 70)
    
    try:
        result = await brain.execute_complex_task(goal)
        
        print("-" * 70)
        print()
        print("📈 Execution Results:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Total Steps: {result.get('total_steps', 0)}")
        print(f"   Completed: {result.get('completed_steps', 0)}")
        print(f"   Failed: {result.get('failed_steps', 0)}")
        print()
        
        if result.get('results'):
            print("📝 Step Results:")
            for step_id, output in result['results'].items():
                if isinstance(output, dict) and 'output' in output:
                    content = output['output']
                    if isinstance(content, str) and len(content) > 100:
                        content = content[:100] + "..."
                    print(f"   Step {step_id}: {content}")
                else:
                    print(f"   Step {step_id}: {output}")
        
        print()
        
        # Show updated workspace stats
        final_stats = brain.workspace_api.stats()
        print("📊 Final Workspace Statistics:")
        print(f"   - Total nodes: {final_stats['total_nodes']}")
        print(f"   - Files: {final_stats['files']}")
        print(f"   - Directories: {final_stats['directories']}")
        print()
        
        print("✅ Demo completed successfully!")
        print()
        print("🎯 Key Features Demonstrated:")
        print("   ✓ Workspace Graph: Real-time file system representation")
        print("   ✓ Manager Agent: Task decomposition and orchestration")
        print("   ✓ Worker Agents: Specialized, context-optimized execution")
        print("   ✓ Dynamic Sync: Automatic graph updates on file changes")
        print("   ✓ Parallel Execution: Independent steps run concurrently")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    await brain.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
