
"""
Enhanced FastAPI server for Artifact ATP reasoning system
Provides robust API endpoints with comprehensive error handling and validation
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import json
import time
from pathlib import Path

from core.reasoning_orchestrator import get_orchestrator
from utils.logger import logger
from utils.exceptions import ArtifactATPError, ValidationError, InvalidInputError

# Initialize FastAPI app
app = FastAPI(
    title="Artifact ATP - Advanced Reasoning System",
    description="Full-stack automated theorem proving and symbolic reasoning system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class ReasoningRequest(BaseModel):
    """Request model for reasoning endpoint"""
    data: List[float] = Field(..., min_items=1, max_items=1000, description="Numerical data points for analysis")
    known_theorems: Optional[List[Dict[str, Any]]] = Field(default=None, description="Known mathematical theorems")
    analysis_depth: Optional[str] = Field(default="standard", pattern="^(basic|standard|deep)$", description="Analysis depth level")
    enable_multi_agent: Optional[bool] = Field(default=True, description="Enable multi-agent reasoning")
    timeout_seconds: Optional[float] = Field(default=60.0, ge=1.0, le=300.0, description="Maximum processing time")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All data points must be numbers")
        if any(abs(x) > 1e10 for x in v):
            raise ValueError("Data points must be within reasonable range")
        return v
    
    @validator('known_theorems')
    def validate_theorems(cls, v):
        if v is not None:
            for theorem in v:
                if not isinstance(theorem, dict):
                    raise ValueError("Each theorem must be a dictionary")
                if 'name' not in theorem or 'structure' not in theorem:
                    raise ValueError("Each theorem must have 'name' and 'structure' fields")
        return v

class ReasoningResponse(BaseModel):
    """Response model for reasoning endpoint"""
    success: bool
    execution_time: float
    analysis_depth: str
    results: Dict[str, Any]
    quality_score: float
    warnings: List[str]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    error_code: str
    details: Dict[str, Any]
    timestamp: str

# Global state
reasoning_jobs = {}  # Track background jobs

# Dependency for loading known theorems
def load_known_theorems() -> List[Dict[str, Any]]:
    """Load known theorems from file"""
    try:
        theorems_file = Path("data/known_theorems.json")
        if theorems_file.exists():
            with open(theorems_file) as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        logger.warning(f"Failed to load known theorems: {str(e)}")
        return []

# Error handlers
@app.exception_handler(ArtifactATPError)
async def artifact_error_handler(request, exc: ArtifactATPError):
    """Handle Artifact ATP specific errors"""
    logger.error(f"Artifact ATP error: {exc.message}", error_code=exc.error_code, metadata=exc.metadata)
    
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.message,
            error_code=exc.error_code,
            details=exc.metadata,
            timestamp=str(exc.timestamp)
        ).dict()
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {str(exc)}")
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Input validation failed",
            error_code="VALIDATION_ERROR",
            details={"validation_error": str(exc)},
            timestamp=str(time.time())
        ).dict()
    )

@app.exception_handler(Exception)
async def general_error_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"error_type": type(exc).__name__, "error_message": str(exc)},
            timestamp=str(time.time())
        ).dict()
    )

# Main endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
    "message": "Artifact Reason - Advanced Reasoning System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {}
    
    # Check orchestrator
    try:
        orchestrator = get_orchestrator()
        components["orchestrator"] = "healthy"
    except Exception:
        components["orchestrator"] = "unhealthy"
    
    # Check file system
    try:
        Path("data").exists()
        components["filesystem"] = "healthy"
    except Exception:
        components["filesystem"] = "unhealthy"
    
    # Overall status
    status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        components=components
    )

@app.post("/reason", response_model=ReasoningResponse)
async def reason(
    request: ReasoningRequest,
    background_tasks: BackgroundTasks,
    default_theorems: List[Dict[str, Any]] = Depends(load_known_theorems)
):
    """
    Main reasoning endpoint - performs complete ATP analysis
    
    This endpoint runs the full Artifact ATP pipeline including:
    - Multi-agent hypothesis generation and validation
    - Symbolic regression with advanced scoring
    - Formal proof verification
    - Cross-validation and consensus building
    """
    start_time = time.time()
    job_id = f"job_{int(start_time * 1000)}"
    
    try:
        logger.info(f"Starting reasoning job {job_id}", 
                   data_points=len(request.data), 
                   analysis_depth=request.analysis_depth,
                   multi_agent_enabled=request.enable_multi_agent)
        
        # Prepare input data
        known_theorems = request.known_theorems or default_theorems
        
        # Get orchestrator and run reasoning
        orchestrator = get_orchestrator()
        
        if request.enable_multi_agent and request.analysis_depth in ["standard", "deep"]:
            # Use enhanced multi-agent pipeline
            results = await asyncio.wait_for(
                orchestrator.orchestrate_reasoning(request.data, known_theorems),
                timeout=request.timeout_seconds
            )
        else:
            # Use traditional pipeline (call async method and await)
            results = await asyncio.wait_for(
                orchestrator.orchestrate_reasoning(request.data, known_theorems),
                timeout=request.timeout_seconds
            )
        
        execution_time = time.time() - start_time
        
        # Extract quality metrics
        quality_score = 0.5  # Default
        warnings = []
        
        if isinstance(results, dict):
            if "quality_assessment" in results:
                quality_score = results["quality_assessment"].get("overall_score", 0.5)
            
            if "final_recommendation" in results:
                warnings = results["final_recommendation"].get("warnings", [])
            
            # Add any consensus flags
            if "cross_validation" in results:
                warnings.extend(results["cross_validation"].get("consensus_flags", []))
        
        # Prepare response
        response = ReasoningResponse(
            success=True,
            execution_time=execution_time,
            analysis_depth=request.analysis_depth,
            results=results,
            quality_score=quality_score,
            warnings=warnings,
            metadata={
                "job_id": job_id,
                "data_points": len(request.data),
                "theorems_used": len(known_theorems),
                "multi_agent_enabled": request.enable_multi_agent,
                "timeout_seconds": request.timeout_seconds
            }
        )
        
        logger.info(f"Completed reasoning job {job_id}", 
                   execution_time=execution_time,
                   quality_score=quality_score,
                   warnings_count=len(warnings))
        # --- Begin: Automated dual-output file writing ---
        import os
        output_dir = "proofs"
        os.makedirs(output_dir, exist_ok=True)
        # Write JSON output
        json_path = os.path.join(output_dir, "prime_analysis.json")
        with open(json_path, "w") as f_json:
            f_json.write(response.json())
        # --- End: Automated dual-output file writing ---
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"Reasoning job {job_id} timed out after {request.timeout_seconds} seconds")
        raise HTTPException(
            status_code=408,
            detail=f"Reasoning timed out after {request.timeout_seconds} seconds"
        )
    
    except Exception as e:
        logger.error(f"Reasoning job {job_id} failed: {str(e)}", exc_info=True)
        raise

@app.post("/analyze/symbolic", response_model=Dict[str, Any])
async def analyze_symbolic(data: List[float], strategy: str = "mixed"):
    """
    Symbolic regression analysis endpoint
    
    Performs symbolic regression using specified strategy:
    - linear: Linear expressions only
    - polynomial: Polynomial expressions
    - transcendental: Exponential, trigonometric, logarithmic
    - mixed: All strategies combined
    """
    try:
        if not data or len(data) < 2:
            raise InvalidInputError("data", data, "list with at least 2 numbers")
        
        from search.candidate_space import CandidateGenerator
        from search.scoring_engine import ScoringEngine
        
        # Generate candidates
        generator = CandidateGenerator(population_size=50)
        candidates = generator.generate_candidates(data, strategy=strategy)
        
        # Score candidates
        if candidates:
            engine = ScoringEngine()
            x_data = list(range(len(data)))
            scores = engine.score_equations(candidates, x_data, data)
            top_candidates = engine.get_top_candidates(scores, top_k=5)
        else:
            top_candidates = []
        
        return {
            "strategy": strategy,
            "candidates_generated": len(candidates),
            "top_candidates": [
                {
                    "expression": expr,
                    "score": metrics.final_score,
                    "complexity": metrics.complexity_penalty,
                    "r_squared": metrics.r_squared
                }
                for expr, metrics in top_candidates
            ],
            "data_points": len(data)
        }
        
    except Exception as e:
        logger.error(f"Symbolic analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prove", response_model=Dict[str, Any])
async def verify_proof(formal_expression: str):
    """
    Proof verification endpoint
    
    Attempts to verify a formal mathematical expression using Lean 4
    """
    try:
        from proofs.lean_interface import prove_async
        
        result = await prove_async(formal_expression)
        
        return {
            "formal_expression": formal_expression,
            "is_valid": result.is_valid,
            "confidence": result.confidence,
            "proof_text": result.proof_text,
            "error_messages": result.error_messages,
            "tactics_used": result.tactics_used,
            "verification_time": result.verification_time,
            "complexity_score": result.complexity_score,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Proof verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str):
    """Get status of a background reasoning job"""
    if job_id in reasoning_jobs:
        return reasoning_jobs[job_id]
    else:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/theorems", response_model=List[Dict[str, Any]])
async def list_known_theorems(theorems: List[Dict[str, Any]] = Depends(load_known_theorems)):
    """List all known theorems"""
    return theorems

@app.post("/theorems", response_model=Dict[str, str])
async def add_theorem(name: str, structure: str, description: str = ""):
    """Add a new theorem to the knowledge base"""
    try:
        theorems_file = Path("data/known_theorems.json")
        
        # Load existing theorems
        if theorems_file.exists():
            with open(theorems_file) as f:
                theorems = json.load(f)
        else:
            theorems = []
        
        # Add new theorem
        new_theorem = {
            "name": name,
            "structure": structure,
            "description": description,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        theorems.append(new_theorem)
        
        # Save back to file
        theorems_file.parent.mkdir(exist_ok=True)
        with open(theorems_file, 'w') as f:
            json.dump(theorems, f, indent=2)
        
        logger.info(f"Added new theorem: {name}")
        return {"message": f"Theorem '{name}' added successfully"}
        
    except Exception as e:
        logger.error(f"Failed to add theorem: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting Artifact ATP server")
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize orchestrator
    try:
        orchestrator = get_orchestrator()
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down Artifact ATP server")
    
    # Cleanup proof checker
    try:
        from proofs.lean_interface import get_proof_checker
        checker = get_proof_checker()
        checker.cleanup()
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
