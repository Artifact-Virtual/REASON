"""
LLM Service Abstraction
Provides a centralized LLM service that can be configured, loaded, and managed independently
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import yaml
import httpx

from llm.gpt_wrapper import OllamaClient, LLMResponse


def run_command(command: str, shell: bool = True) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


async def check_ollama_availability() -> bool:
    """Check if Ollama is available and responding"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            return response.status_code == 200
    except:
        return False


async def install_ollama() -> bool:
    """Install Ollama if not available"""
    print("🔧 Ollama not found. Installing...")
    
    # Check if ollama command exists
    exit_code, stdout, stderr = run_command("which ollama")
    if exit_code == 0:
        print("✅ Ollama binary found, attempting to start service...")
        return await start_ollama_service()
    
    print("📥 Downloading and installing Ollama...")
    exit_code, stdout, stderr = run_command(
        "curl -fsSL https://ollama.com/install.sh | sh"
    )
    
    if exit_code != 0:
        print(f"❌ Failed to install Ollama: {stderr}")
        return False
    
    print("✅ Ollama installed successfully")
    return await start_ollama_service()


async def start_ollama_service() -> bool:
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    
    # Check if already running
    if await check_ollama_availability():
        print("✅ Ollama service already running")
        return True
    
    # Start ollama serve in background
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for service to start
        for i in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            if await check_ollama_availability():
                print("✅ Ollama service started successfully")
                return True
            if i == 10:
                print("⏳ Still waiting for Ollama service to start...")
        
        print("❌ Ollama service failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start Ollama service: {e}")
        return False


async def ensure_model_available(model_name: str = "tinyllama") -> bool:
    """Ensure the specified model is available in Ollama"""
    if not await check_ollama_availability():
        return False
    
    print(f"🔍 Checking if model '{model_name}' is available...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"].split(":")[0] for model in data.get("models", [])]
                
                if model_name in models:
                    print(f"✅ Model '{model_name}' is already available")
                    return True
                
                print(f"📥 Pulling model '{model_name}'...")
                # Pull the model
                exit_code, stdout, stderr = run_command(f"ollama pull {model_name}")
                
                if exit_code == 0:
                    print(f"✅ Model '{model_name}' pulled successfully")
                    return True
                else:
                    print(f"❌ Failed to pull model '{model_name}': {stderr}")
                    return False
            
    except Exception as e:
        print(f"❌ Error checking/pulling model: {e}")
        return False
    
    return False


async def setup_ollama_environment(model_name: str = "tinyllama") -> bool:
    """Complete Ollama setup: install, start service, and ensure model availability"""
    print("🎯 Setting up Ollama environment...")
    
    # Step 1: Check if Ollama is available
    if await check_ollama_availability():
        print("✅ Ollama service is already running")
    else:
        # Step 2: Install and start Ollama if needed
        if not await install_ollama():
            return False
    
    # Step 3: Ensure model is available
    if not await ensure_model_available(model_name):
        print(f"⚠️ Model '{model_name}' not available, will use mock fallback")
        return False
    
    print("🎉 Ollama environment setup complete!")
    return True


@dataclass
class LLMConfig:
    """LLM Configuration"""
    provider: str = "ollama"  # ollama, openai, mock
    model: str = "tinyllama"
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0
    temperature: float = 0.7
    max_tokens: int = 1000
    fallback_to_mock: bool = True


class LLMServiceProtocol(Protocol):
    """Protocol for LLM service implementations"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt"""
        ...
    
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        ...
    
    async def close(self) -> None:
        """Close service connections"""
        ...


class MockLLMService:
    """Mock LLM service for testing and fallback"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response based on prompt patterns"""
        
        # Pattern-based mock responses
        if "hypothesis" in prompt.lower() or "generate" in prompt.lower():
            content = '''
            {
                "hypotheses": [
                    {
                        "description": "Linear relationship: y = ax + b",
                        "mathematical_form": "y = 2*x + 1",
                        "confidence": 0.8,
                        "evidence": ["monotonic increase", "constant difference"],
                        "weaknesses": ["assumes linearity"]
                    },
                    {
                        "description": "Quadratic relationship: y = ax² + bx + c", 
                        "mathematical_form": "y = x^2",
                        "confidence": 0.9,
                        "evidence": ["accelerating growth", "perfect squares"],
                        "weaknesses": ["limited to polynomials"]
                    }
                ]
            }
            '''
        elif "validate" in prompt.lower():
            content = '''
            {
                "validation_results": [
                    {
                        "hypothesis": "y = x^2",
                        "is_valid": true,
                        "confidence": 0.95,
                        "reasoning": "Perfect fit with quadratic pattern"
                    }
                ]
            }
            '''
        elif "hallucination" in prompt.lower():
            content = '''
            {
                "analysis": {
                    "hallucination_detected": false,
                    "confidence": 0.8,
                    "reasoning": "Responses are consistent with mathematical patterns"
                }
            }
            '''
        elif "consensus" in prompt.lower():
            content = '''
            {
                "consensus": {
                    "agreed_hypothesis": "y = x^2",
                    "confidence": 0.9,
                    "supporting_agents": 3,
                    "reasoning": "All agents agree on quadratic relationship"
                }
            }
            '''
        else:
            content = "This is a mock response for the prompt: " + prompt[:100] + "..."
        
        return LLMResponse(
            content=content,
            confidence=0.8,
            reasoning_steps=["Mock reasoning step 1", "Mock reasoning step 2"],
            metadata={"provider": "mock", "model": "mock"},
            is_valid=True
        )
    
    async def health_check(self) -> bool:
        return True
    
    async def close(self) -> None:
        pass


class OllamaLLMService:
    """Ollama LLM service implementation"""
    
            
    # Create Ollama service
    def __init__(self, config):
        self.config = config
        self.fallback_service = None
        self.primary_service = None
        self._initialized = False
        if self.config and getattr(self.config, 'fallback_to_mock', False):
            self.fallback_service = MockLLMService(self.config)

    async def initialize(self):
        # Test primary service health asynchronously
        if self.primary_service:
            try:
                is_healthy = await self.primary_service.health_check()
                if not is_healthy and self.fallback_service:
                    print("⚠️ Primary LLM service unhealthy, will use mock fallback")
            except Exception:
                if not self.fallback_service:
                    raise
                print("⚠️ Primary LLM service failed, will use mock fallback")
        self._initialized = True
        return True
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with fallback logic"""
        if not self._initialized:
            await self.initialize()
        
        # Try primary service first
        try:
            if self.primary_service:
                is_healthy = await self.primary_service.health_check()
                if is_healthy:
                    return await self.primary_service.generate(prompt, **kwargs)
        except Exception as e:
            print(f"Primary LLM service failed: {e}")
        
        # Fall back to mock service
        if self.fallback_service:
            print("Using fallback LLM service")
            return await self.fallback_service.generate(prompt, **kwargs)
        
        # Last resort - return error response
        return LLMResponse(
            content="",
            confidence=0.0,
            reasoning_steps=[],
            metadata={},
            is_valid=False,
            error_message="No available LLM service"
        )
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all services"""
        results = {}
        
        if self.primary_service:
            try:
                results["primary"] = await self.primary_service.health_check()
            except Exception:
                results["primary"] = False
        
        if self.fallback_service:
            try:
                results["fallback"] = await self.fallback_service.health_check()
            except Exception:
                results["fallback"] = False
        
        return results
    
    async def close(self) -> None:
        """Close all services"""
        if self.primary_service:
            await self.primary_service.close()
        if self.fallback_service:
            await self.fallback_service.close()


# Global service instance
_llm_service: Optional[OllamaLLMService] = None


async def get_llm_service() -> OllamaLLMService:
    """Get or create global LLM service"""
    global _llm_service
    if _llm_service is None:
        # Always use Ollama with a valid config and preferred model
        config = LLMConfig(provider="ollama", model="tinyllama", base_url="http://localhost:11434", timeout=30.0, temperature=0.7, max_tokens=1000, fallback_to_mock=True)
        _llm_service = OllamaLLMService(config)
        await _llm_service.initialize()
    return _llm_service


async def generate_response(prompt: str, **kwargs) -> LLMResponse:
    """High-level function for generating LLM responses"""
    service = await get_llm_service()
    return await service.generate(prompt, **kwargs)


def configure_llm(config: LLMConfig) -> None:
    """Configure LLM service globally"""
    global _llm_service
    _llm_service = LLMService(config)


def use_mock_llm() -> None:
    """Switch to mock LLM for testing"""
    config = LLMConfig(provider="ollama", fallback_to_mock=False)
    configure_llm(config)


async def warmup_llm() -> bool:
    """Warm up the LLM service"""
    try:
        service = await get_llm_service()
        health = await service.health_check()
        print(f"LLM Service Health: {health}")
        return any(health.values())
    except Exception as e:
        print(f"LLM warmup failed: {e}")
        return False


if __name__ == "__main__":
    """CLI for LLM service management"""
    import sys
    
    async def main():
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "warmup":
                success = await warmup_llm()
                sys.exit(0 if success else 1)
            
            elif command == "health":
                service = await get_llm_service()
                health = await service.health_check()
                print(json.dumps(health, indent=2))
                sys.exit(0 if any(health.values()) else 1)
            
            elif command == "test":
                service = await get_llm_service()
                response = await service.generate("What is 2 + 2?")
                print(f"Response: {response.content}")
                print(f"Valid: {response.is_valid}")
                sys.exit(0 if response.is_valid else 1)
        
        else:
            print("Usage: python llm_service.py [warmup|health|test]")
            sys.exit(1)
    
    asyncio.run(main())
