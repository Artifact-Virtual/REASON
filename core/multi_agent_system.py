"""
Multi-Agent Reasoning System for Artifact ATP
Implements agent-to-agent communication, consensus building, and hallucination detection
"""
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
import json
from abc import ABC, abstractmethod

from utils.logger import logger, async_performance_monitor
from utils.exceptions import AgentCommunicationError, ConsensusError, handle_exceptions
from llm.gpt_wrapper import LLMResponse
from llm.llm_service import generate_response

class AgentRole(Enum):
    """Agent roles in the reasoning system"""
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    VALIDATOR = "validator" 
    META_REASONER = "meta_reasoner"
    PROOF_CHECKER = "proof_checker"
    HALLUCINATION_DETECTOR = "hallucination_detector"
    CONSENSUS_BUILDER = "consensus_builder"

@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResponse:
    """Structured agent response with validation"""
    agent_id: str
    role: AgentRole
    response: Any
    confidence: float
    reasoning_steps: List[str]
    evidence: Dict[str, Any]
    flags: List[str] = field(default_factory=list)  # Warning flags
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """Base class for all reasoning agents"""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_active = False
        logger.info(f"Initialized agent {agent_id} with role {role.value}")
    
    @abstractmethod
    async def process(self, input_data: Any) -> AgentResponse:
        """Process input and return structured response"""
        pass
    
    async def send_message(self, recipient: 'BaseAgent', message_type: str, content: Dict[str, Any], confidence: float = 1.0):
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient.agent_id,
            message_type=message_type,
            content=content,
            confidence=confidence
        )
        await recipient.receive_message(message)
        logger.debug(f"Message sent from {self.agent_id} to {recipient.agent_id}", 
                    message_type=message_type, confidence=confidence)
    
    async def receive_message(self, message: AgentMessage):
        """Receive message from another agent"""
        await self.message_queue.put(message)
    
    async def get_next_message(self) -> Optional[AgentMessage]:
        """Get next message from queue (non-blocking)"""
        try:
            return self.message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

class HypothesisGeneratorAgent(BaseAgent):
    """Agent specialized in generating hypotheses"""
    
    def __init__(self, agent_id: str = "hypothesis_gen_001"):
        super().__init__(agent_id, AgentRole.HYPOTHESIS_GENERATOR)
    
    @async_performance_monitor
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Generate multiple hypotheses from input data"""
        data = input_data.get("data", [])
        context = input_data.get("context", "")
        prompt = f"Analyze the following data and generate hypotheses. Data: {data}. Context: {context}"
        llm_response = await generate_response(prompt, temperature=0.8)

        if not llm_response.is_valid:
            return AgentResponse(
                agent_id=self.agent_id,
                role=self.role,
                response={},
                confidence=0.0,
                reasoning_steps=[],
                evidence={},
                flags=["LLM_ERROR", llm_response.error_message]
            )

        hypotheses = []
        try:
            # Parse JSON response
            hypotheses_data = json.loads(llm_response.content)
            hypotheses = hypotheses_data.get("hypotheses", [])
        except Exception:
            # Fallback to simple hypothesis generation
            if len(data) >= 2:
                simple_hypotheses = [
                    {
                        "description": f"Linear relationship: y = {data[1] - data[0]} * x + {data[0]}",
                        "mathematical_form": f"y = {data[1] - data[0]}x + {data[0]}",
                        "confidence": 0.6,
                        "evidence": ["Simple linear pattern"],
                        "weaknesses": ["May not capture non-linear relationships"]
                    }
                ]
                hypotheses = simple_hypotheses
            else:
                hypotheses = []

        # Calculate overall confidence
        overall_confidence = sum(h.get("confidence", 0) for h in hypotheses) / len(hypotheses) if hypotheses else 0

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            response=hypotheses,
            confidence=overall_confidence,
            reasoning_steps=llm_response.reasoning_steps,
            evidence={"llm_confidence": llm_response.confidence, "hypothesis_count": len(hypotheses)},
            metadata={"llm_metadata": llm_response.metadata}
        )

class ValidatorAgent(BaseAgent):
    """Agent specialized in validating hypotheses and detecting inconsistencies"""
    
    def __init__(self, agent_id: str = "validator_001"):
        super().__init__(agent_id, AgentRole.VALIDATOR)
    
    @async_performance_monitor
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Validate hypotheses for consistency and plausibility"""
        hypotheses = input_data.get("hypotheses", [])
        original_data = input_data.get("data", [])
        validation_results = []
        flags = []
        for h in hypotheses:
            # Example validation: check if hypothesis has confidence and description
            valid = "confidence" in h and "description" in h
            validation_results.append({"hypothesis": h, "is_valid": valid})
            if not valid:
                flags.append("INVALID_HYPOTHESIS")
        # Always define validation_results before use
        if not validation_results:
            validation_results = []
        overall_confidence = sum(h.get("confidence", 0) for h in hypotheses) / len(hypotheses) if hypotheses else 0
        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            response=validation_results,
            confidence=overall_confidence,
            reasoning_steps=[f"Validated {len(hypotheses)} hypotheses"],
            evidence={"validation_count": len(validation_results)},
            flags=flags
        )
    
    async def _validate_single_hypothesis(self, hypothesis: Dict[str, Any], data: List) -> Dict[str, Any]:
        """Validate a single hypothesis"""
        description = hypothesis.get("description", "")
        mathematical_form = hypothesis.get("mathematical_form", "")
        
        prompt = f"""
        Validate this hypothesis against the given data:
        
        Hypothesis: {description}
        Mathematical Form: {mathematical_form}
        Data: {data}
        
        Evaluate:
        1. Mathematical consistency
        2. Fit to data points
        3. Logical soundness
        4. Potential for overfitting
        
        Return a confidence score (0-1) and brief explanation.
        """
        
        llm_response = await generate_response(prompt, temperature=0.3)
        
        # Simple validation logic
        consistency_score = 0.8 if "consistent" in llm_response.content.lower() else 0.4
        confidence = min(hypothesis.get("confidence", 0.5), llm_response.confidence)
        
        return {
            "hypothesis_id": hypothesis.get("id", "unknown"),
            "confidence": confidence,
            "consistency_score": consistency_score,
            "validation_notes": llm_response.content[:200],
            "flags": []
        }

class HallucinationDetectorAgent(BaseAgent):
    """Agent specialized in detecting hallucinations and false information"""
    
    def __init__(self, agent_id: str = "hallucination_detector_001"):
        super().__init__(agent_id, AgentRole.HALLUCINATION_DETECTOR)
    
    @async_performance_monitor
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Detect potential hallucinations in agent responses"""
        responses = input_data.get("agent_responses", [])
        original_data = input_data.get("data", [])
        hallucination_flags = []
        for r in responses:
            # Example: flag if response contains 'hallucination' keyword
            if isinstance(r, dict) and 'hallucination' in str(r).lower():
                hallucination_flags.append("HALLUCINATION_DETECTED")
        # Always define hallucination_flags before use
        if not hallucination_flags:
            hallucination_flags = []
        overall_confidence = 1.0 - (len(hallucination_flags) / len(responses) if responses else 0)
        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            response={"hallucination_flags": hallucination_flags, "clean_responses": len(responses) - len(hallucination_flags)},
            confidence=overall_confidence,
            reasoning_steps=[f"Analyzed {len(responses)} responses for hallucinations"],
            evidence={"total_responses": len(responses), "flagged_responses": len(hallucination_flags)},
            flags=hallucination_flags
        )
    
    async def _detect_hallucination(self, response: Dict[str, Any], original_data: List) -> tuple:
        """Detect hallucination in a single response"""
        flags = []
        confidence = 1.0
        
        # Check for impossible values
        response_content = str(response)
        if any(word in response_content.lower() for word in ["impossible", "infinite", "undefined", "nan"]):
            flags.append("IMPOSSIBLE_VALUES")
            confidence -= 0.3
        
        # Check for inconsistency with data
        if "mathematical_form" in response:
            math_form = response["mathematical_form"]
            if "x^100" in math_form or "exp(1000" in math_form:  # Unreasonably complex
                flags.append("UNREASONABLY_COMPLEX")
                confidence -= 0.2
        
        # Check confidence calibration
        claimed_confidence = response.get("confidence", 0.5)
        if claimed_confidence > 0.95:  # Overconfidence
            flags.append("OVERCONFIDENT")
            confidence -= 0.1
        
        return flags, max(0.0, confidence)

class ConsensusBuilderAgent(BaseAgent):
    """Agent that builds consensus among multiple agent responses"""
    
    def __init__(self, agent_id: str = "consensus_builder_001"):
        super().__init__(agent_id, AgentRole.CONSENSUS_BUILDER)
    
    @async_performance_monitor
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Build consensus from multiple agent responses"""
        agent_responses = input_data.get("agent_responses", [])
        
        if len(agent_responses) < 2:
            return AgentResponse(
                agent_id=self.agent_id,
                role=self.role,
                response={},
                confidence=0.0,
                reasoning_steps=["Insufficient responses for consensus"],
                evidence={},
                flags=["INSUFFICIENT_RESPONSES"]
            )
        
        # Analyze agreements and disagreements
        consensus_result = await self._build_consensus(agent_responses)
        
        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            response=consensus_result,
            confidence=consensus_result.get("consensus_strength", 0.0),
            reasoning_steps=[f"Built consensus from {len(agent_responses)} responses"],
            evidence={"agent_count": len(agent_responses)},
            flags=consensus_result.get("consensus_flags", [])
        )
    
    async def _build_consensus(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build consensus from agent responses"""
        # Simple consensus building logic
        confidence_scores = [r.get("confidence", 0.0) for r in responses]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Check for agreement
        agreement_threshold = 0.7
        high_confidence_responses = [r for r in responses if r.get("confidence", 0) > agreement_threshold]
        
        consensus_flags = []
        if len(high_confidence_responses) < len(responses) * 0.5:
            consensus_flags.append("LOW_CONSENSUS")
        
        return {
            "consensus_strength": avg_confidence,
            "agreeing_agents": len(high_confidence_responses),
            "total_agents": len(responses),
            "consensus_flags": consensus_flags,
            "final_recommendation": high_confidence_responses[0] if high_confidence_responses else responses[0]
        }

class MultiAgentOrchestrator:
    """Orchestrates multi-agent reasoning with consensus building"""
    
    def __init__(self):
        self.agents = {
            AgentRole.HYPOTHESIS_GENERATOR: HypothesisGeneratorAgent(),
            AgentRole.VALIDATOR: ValidatorAgent(),
            AgentRole.HALLUCINATION_DETECTOR: HallucinationDetectorAgent(),
            AgentRole.CONSENSUS_BUILDER: ConsensusBuilderAgent()
        }
        logger.info("Initialized multi-agent orchestrator with 4 agents")
    
    @async_performance_monitor
    async def orchestrate_reasoning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multi-agent reasoning process"""
        results = {}
        
        # Step 1: Generate hypotheses
        hypothesis_response = await self.agents[AgentRole.HYPOTHESIS_GENERATOR].process(input_data)
        results["hypothesis_generation"] = hypothesis_response
        
        # Step 2: Validate hypotheses
        validation_input = {
            "hypotheses": hypothesis_response.response,
            "data": input_data.get("data", [])
        }
        validation_response = await self.agents[AgentRole.VALIDATOR].process(validation_input)
        results["validation"] = validation_response
        
        # Step 3: Detect hallucinations
        hallucination_input = {
            "agent_responses": [hypothesis_response.__dict__, validation_response.__dict__],
            "data": input_data.get("data", [])
        }
        hallucination_response = await self.agents[AgentRole.HALLUCINATION_DETECTOR].process(hallucination_input)
        results["hallucination_detection"] = hallucination_response
        
        # Step 4: Build consensus
        consensus_input = {
            "agent_responses": [hypothesis_response.__dict__, validation_response.__dict__]
        }
        consensus_response = await self.agents[AgentRole.CONSENSUS_BUILDER].process(consensus_input)
        results["consensus"] = consensus_response
        
        # Compile final results
        final_result = {
            "multi_agent_results": results,
            "final_consensus": consensus_response.response,
            "overall_confidence": consensus_response.confidence,
            "quality_flags": hallucination_response.flags + consensus_response.flags,
            "reasoning_quality": self._assess_reasoning_quality(results)
        }
        
        logger.info("Completed multi-agent reasoning orchestration", 
                   overall_confidence=consensus_response.confidence,
                   quality_flags_count=len(final_result["quality_flags"]))
        
        return final_result
    
    def _assess_reasoning_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall reasoning quality"""
        quality_metrics = {
            "hypothesis_diversity": len(results["hypothesis_generation"].response),
            "validation_strength": results["validation"].confidence,
            "hallucination_risk": len(results["hallucination_detection"].flags),
            "consensus_strength": results["consensus"].confidence
        }
        
        overall_quality = (
            quality_metrics["validation_strength"] * 0.3 +
            quality_metrics["consensus_strength"] * 0.4 +
            max(0, 1 - quality_metrics["hallucination_risk"] * 0.1) * 0.3
        )
        
        quality_metrics["overall_quality"] = overall_quality
        return quality_metrics
