"""
Custom exceptions for Artifact ATP system
Provides structured error handling across all components
"""

class ArtifactATPError(Exception):
    """Base exception for all Artifact Reason errors"""
    
    def __init__(self, message: str, error_code: str = None, metadata: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.metadata = metadata or {}
        self.timestamp = __import__('datetime').datetime.utcnow().isoformat()

class LLMError(ArtifactATPError):
    """Errors related to LLM operations"""
    pass

class LLMTimeoutError(LLMError):
    """LLM request timeout"""
    def __init__(self, timeout_duration: float):
        super().__init__(
            f"LLM request timed out after {timeout_duration} seconds",
            "LLM_TIMEOUT",
            {"timeout_duration": timeout_duration}
        )

class LLMValidationError(LLMError):
    """LLM response validation failed"""
    def __init__(self, validation_issues: list):
        super().__init__(
            f"LLM response validation failed: {', '.join(validation_issues)}",
            "LLM_VALIDATION_FAILED",
            {"validation_issues": validation_issues}
        )

class ProofError(ArtifactATPError):
    """Errors related to proof system"""
    pass

class ProofValidationError(ProofError):
    """Proof validation failed"""
    def __init__(self, proof_expression: str, error_details: str):
        super().__init__(
            f"Proof validation failed for expression: {proof_expression}",
            "PROOF_VALIDATION_FAILED",
            {"proof_expression": proof_expression, "error_details": error_details}
        )

class SymbolicRegressionError(ArtifactATPError):
    """Errors related to symbolic regression"""
    pass

class InsufficientDataError(SymbolicRegressionError):
    """Not enough data for symbolic regression"""
    def __init__(self, data_points: int, minimum_required: int):
        super().__init__(
            f"Insufficient data: {data_points} points, minimum {minimum_required} required",
            "INSUFFICIENT_DATA",
            {"data_points": data_points, "minimum_required": minimum_required}
        )

class ReasoningError(ArtifactATPError):
    """Errors in reasoning pipeline"""
    pass

class HypothesisGenerationError(ReasoningError):
    """Failed to generate valid hypotheses"""
    def __init__(self, stage: str, reason: str):
        super().__init__(
            f"Hypothesis generation failed at {stage}: {reason}",
            "HYPOTHESIS_GENERATION_FAILED",
            {"stage": stage, "reason": reason}
        )

class AgentCommunicationError(ArtifactATPError):
    """Errors in multi-agent communication"""
    pass

class ConsensusError(AgentCommunicationError):
    """Agents failed to reach consensus"""
    def __init__(self, conflicting_agents: list, disagreement_points: list):
        super().__init__(
            f"Consensus failed between agents: {', '.join(conflicting_agents)}",
            "CONSENSUS_FAILED",
            {"conflicting_agents": conflicting_agents, "disagreement_points": disagreement_points}
        )

class ValidationError(ArtifactATPError):
    """Input validation errors"""
    pass

class InvalidInputError(ValidationError):
    """Invalid input data"""
    def __init__(self, field: str, value: any, expected_type: str):
        super().__init__(
            f"Invalid input for field '{field}': expected {expected_type}, got {type(value).__name__}",
            "INVALID_INPUT",
            {"field": field, "value": str(value), "expected_type": expected_type}
        )

def handle_exceptions(func):
    """Decorator for consistent exception handling"""
    import functools
    from utils.logger import logger
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ArtifactATPError as e:
            logger.error(f"Artifact ATP Error in {func.__name__}", 
                        error_code=e.error_code, 
                        error_message=e.message,
                        metadata=e.metadata)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}", 
                        error_type=type(e).__name__, 
                        error_message=str(e))
            raise ArtifactATPError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "UNEXPECTED_ERROR",
                {"original_error": str(e), "function": func.__name__}
            )
    
    return wrapper

async def async_handle_exceptions(func):
    """Decorator for consistent async exception handling"""
    import functools
    from utils.logger import logger
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ArtifactATPError as e:
            logger.error(f"Artifact ATP Error in {func.__name__}", 
                        error_code=e.error_code, 
                        error_message=e.message,
                        metadata=e.metadata)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}", 
                        error_type=type(e).__name__, 
                        error_message=str(e))
            raise ArtifactATPError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "UNEXPECTED_ERROR",
                {"original_error": str(e), "function": func.__name__}
            )
    
    return wrapper
