"""
Real LLM inference using Ollama
Provides structured, validated responses for reasoning tasks
"""
import httpx
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structured LLM response with validation"""
    content: str
    confidence: float
    reasoning_steps: List[str]
    metadata: Dict[str, Any]
    is_valid: bool = True
    error_message: Optional[str] = None

class OllamaClient:
    """Enhanced Ollama client with error handling and validation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "codellama"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=10.0)
        
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        """Generate response with comprehensive error handling"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "").strip()
            
            # Basic validation
            if not content:
                return LLMResponse(
                    content="", confidence=0.0, reasoning_steps=[],
                    metadata={}, is_valid=False, 
                    error_message="Empty response from LLM"
                )
            
            # Extract reasoning steps if present
            reasoning_steps = self._extract_reasoning_steps(content)
            confidence = self._calculate_confidence(content, result)
            
            return LLMResponse(
                content=content,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                metadata={
                    "model": self.model,
                    "tokens_used": result.get("eval_count", 0),
                    "response_time": result.get("total_duration", 0) / 1e9
                }
            )
            
        except httpx.TimeoutException:
            logger.error(f"Timeout when calling Ollama API (timeout set to 10s)")
            return LLMResponse(
                content="", confidence=0.0, reasoning_steps=[],
                metadata={"timeout": 10.0, "model": self.model}, is_valid=False,
                error_message="LLM request timeout after 10 seconds"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return LLMResponse(
                content="", confidence=0.0, reasoning_steps=[],
                metadata={}, is_valid=False,
                error_message=f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in LLM generation: {str(e)}")
            return LLMResponse(
                content="", confidence=0.0, reasoning_steps=[],
                metadata={}, is_valid=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _extract_reasoning_steps(self, content: str) -> List[str]:
        """Extract reasoning steps from LLM response"""
        steps = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '-', '*', 'Step']):
                steps.append(line)
        return steps
    
    def _calculate_confidence(self, content: str, result: Dict) -> float:
        """Calculate confidence score based on response characteristics"""
        base_confidence = 0.5
        
        # Increase confidence for longer, more detailed responses
        if len(content) > 100:
            base_confidence += 0.2
        
        # Increase confidence if reasoning steps are present
        if any(word in content.lower() for word in ['because', 'therefore', 'since', 'thus']):
            base_confidence += 0.1
            
        # Decrease confidence for uncertainty markers
        if any(word in content.lower() for word in ['maybe', 'perhaps', 'might', 'could be']):
            base_confidence -= 0.2
            
        return max(0.0, min(1.0, base_confidence))
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Global client instance
_ollama_client = None

async def get_ollama_client() -> OllamaClient:
    """Get or create global Ollama client"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client

async def generate_response(prompt: str, **kwargs) -> LLMResponse:
    """High-level function for generating LLM responses - DEPRECATED"""
    client = await get_ollama_client()
    return await client.generate(prompt, **kwargs)
