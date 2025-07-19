#!/usr/bin/env python3
"""
Artifact ATP System Entry Point
Comprehensive system initialization and startup with automatic dependency management
"""
import asyncio
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm.llm_service import (
    OllamaLLMService, 
    LLMConfig, 
    setup_ollama_environment,
    check_ollama_availability
)


class SystemStartup:
    """Manages complete system startup and initialization"""
    
    def __init__(self):
        self.llm_service = None
        self.startup_time = time.time()
        
    async def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        print("🔍 Checking system prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check if we're in the right directory
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found. Please run from project root.")
            return False
        print("✅ Project structure detected")
        
        return True
    
    async def setup_llm_service(self) -> bool:
        """Setup and initialize LLM service"""
        print("\n🤖 Setting up LLM service...")
        
        try:
            # Configure LLM service
            config = LLMConfig(
                provider="ollama",
                model="tinyllama",
                base_url="http://localhost:11434",
                timeout=30.0,
                temperature=0.7,
                max_tokens=1000,
                fallback_to_mock=True
            )
            
            # Initialize service (this will auto-setup Ollama if needed)
            self.llm_service = OllamaLLMService(config)
            success = await self.llm_service.initialize()
            
            if success:
                print("✅ LLM service initialized successfully")
                
                # Test the service
                print("🧪 Testing LLM service...")
                response = await self.llm_service.generate(
                    "Hello, are you working?", 
                    max_tokens=50
                )
                
                if response.is_valid:
                    print(f"✅ LLM test successful: {response.content[:50]}...")
                    return True
                else:
                    print(f"⚠️ LLM test failed: {response.error_message}")
                    return False
            else:
                print("❌ LLM service initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ LLM service setup error: {e}")
            return False
    
    async def test_core_components(self) -> bool:
        """Test core ATP system components"""
        print("\n🧠 Testing core ATP components...")
        
        try:
            # Test symbolic regressor
            print("Testing symbolic regressor...")
            from core.symbolic_regressor import run_symbolic_regression
            result = run_symbolic_regression([1, 2, 3], [1, 4, 9])
            print(f"✅ Symbolic regression: {result}")
            
            # Test multi-agent system
            print("Testing multi-agent system...")
            from core.multi_agent_system import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator()
            
            input_data = {
                "data": [1, 4, 9, 16],
                "context": "Square numbers test"
            }
            
            result = await orchestrator.orchestrate_reasoning(input_data)
            print(f"✅ Multi-agent system: {len(result)} components")
            
            # Test enhanced orchestrator
            print("Testing enhanced reasoning orchestrator...")
            from core.reasoning_orchestrator import EnhancedReasoningOrchestrator
            enhanced = EnhancedReasoningOrchestrator()
            
            data = [1, 4, 9, 16, 25]
            result = await enhanced.orchestrate_reasoning(data)
            print(f"✅ Enhanced orchestrator: {len(result)} results")
            
            return True
            
        except Exception as e:
            print(f"❌ Core components test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_sample_atp_task(self) -> bool:
        """Run a sample ATP task to demonstrate system functionality"""
        print("\n🎯 Running sample ATP task...")
        
        try:
            from core.reasoning_orchestrator import EnhancedReasoningOrchestrator
            
            # Initialize orchestrator
            orchestrator = EnhancedReasoningOrchestrator()
            
            # Sample mathematical sequence (Fibonacci-like)
            data = [1, 1, 2, 3, 5, 8, 13, 21]
            
            print(f"Input data: {data}")
            print("Running complete ATP pipeline...")
            
            # Run the complete reasoning pipeline
            result = await orchestrator.orchestrate_reasoning(data)
            
            print("\n📊 ATP Results Summary:")
            print(f"• Pipeline components: {len(result)}")
            
            for key, value in result.items():
                if isinstance(value, dict):
                    print(f"• {key}: {len(value)} sub-components")
                else:
                    print(f"• {key}: {type(value).__name__}")
            
            print("\n✅ Sample ATP task completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Sample ATP task failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def start_system(self) -> bool:
        """Complete system startup sequence"""
        print("=" * 60)
        print("🚀 ARTIFACT ATP SYSTEM STARTUP")
        print("=" * 60)
        
        # Step 1: Check prerequisites
        if not await self.check_prerequisites():
            return False
        
        # Step 2: Setup LLM service
        if not await self.setup_llm_service():
            return False
        
        # Step 3: Test core components
        if not await self.test_core_components():
            return False
        
        # Step 4: Run sample task
        if not await self.run_sample_atp_task():
            return False
        
        # Success!
        elapsed = time.time() - self.startup_time
        print("\n" + "=" * 60)
        print(f"🎉 SYSTEM STARTUP COMPLETE ({elapsed:.2f}s)")
        print("=" * 60)
        print("\n📋 System Status:")
        print("• LLM Service: ✅ Online")
        print("• Multi-Agent System: ✅ Ready")
        print("• Symbolic Regression: ✅ Ready")
        print("• Proof System: ✅ Ready (Simulation Mode)")
        print("• ATP Pipeline: ✅ Operational")
        
        print("\n🌐 Next Steps:")
        print("• API Server: python -m uvicorn main:app --reload")
        print("• Frontend: streamlit run frontend/app.py")
        print("• Test Suite: python -m pytest tests/")
        
        return True
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.llm_service:
            await self.llm_service.close()


async def main():
    """Main entry point"""
    startup = SystemStartup()
    
    try:
        success = await startup.start_system()
        if success:
            print("\n🎯 System ready for ATP tasks!")
            return 0
        else:
            print("\n❌ System startup failed!")
            return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        await startup.cleanup()


if __name__ == "__main__":
    # Run the system startup
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
