# Artifact Reason

**Version:** Final Production  
**LLM:** Ollama (TinyLlama) with Auto-Installation  
**Symbolic Regression:** PySR with Fallback  
**Proof System:** Lean Interface (Simulation Mode)  
**API:** FastAPI  
**Frontend:** Streamlit  
**Use Case:** Scientific Discovery, Symbolic Equation Discovery, Formalization and Proof

## Quick Start

Artifact Reason automatically handles all dependencies including Ollama installation:



# Artifact Reason

Artifact Reason is a technical, multi-agent reasoning system for scientific discovery, hypothesis generation, symbolic modeling, and formal proof verification. It integrates curated theorem knowledge for reflection and validation, supporting advanced research workflows.


## System Overview

- **Traditional Pipeline:**
  - Generates hypotheses and symbolic models.
  - Selects the best hypothesis and attempts formal proof.
- **Multi-Agent Analysis:**
  - Agents cross-validate hypotheses and models.
  - Consensus and validation are performed by the multi-agent system.
- **Symbolic Regression:**
  - Explores multiple modeling strategies.
  - Selects best candidate based on scoring and physical plausibility.
- **Cross-Validation:**
  - Divergent agent hypotheses are compared.
  - Consensus and confidence scores are computed.
- **Proof Verification:**
  - Formal proof attempted for main hypothesis (Lean 4 interface).
  - Results are flagged if proof fails or confidence is low.
- **Final Recommendation:**
  - Returns hypothesis, confidence, and agent consensus.
- **Quality Assessment:**
  - Overall score and recommendation provided for each analysis.


## Multi-Agent Validation

- The multi-agent validator performs consensus and validation using integrated agent logic and LLMs.
- Full functionality is available by default; consensus and confidence scores are computed for every analysis.


## Theorem Knowledge Base

- Curated, exhaustive list of mathematical and physical theorems in `data/known_theorems.json`.
- Theorems are loaded for every request and used for system reflection.
- Theorems are never re-proven; only used for knowledge integration and validation.


## Agents

- Abductive, analogical, meta-reasoner, symbolic regressor, multi-agent validator.
- All agents implement real logic (no stubs, samples, or mock code).

## Consensus and Fallback Behavior

- The system computes consensus and confidence scores for every analysis using its multi-agent architecture.
- If consensus among agents is low, or confidence in the result is insufficient, the system automatically falls back to a "low consensus" mode:
  - Results are flagged as having low consensus and/or low confidence.
  - The final recommendation will indicate the fallback status and suggest further validation or review.
  - This ensures transparency in cases where agent agreement is weak or the solution is not robust.
- Users are notified in the response when low consensus fallback occurs, and the output will include a quality assessment and recommendation for next steps.

## Project Structure

- `core/`: Reasoning pipeline, agent logic, orchestrator
- `llm/`: LLM service, prompt templates
- `proofs/`: Lean 4 interface
- `data/`: Input data, curated theorems
- `search/`: Candidate space, scoring engine
- `frontend/`: Streamlit app
- `utils/`: Config, logging, exceptions
- `outputs/`, `logs/`: Results and logs

## API Usage

- FastAPI endpoints defined in `main.py`.
- Submit problems via `/reason` endpoint; receive hypotheses, models, proofs, and quality assessment.

## Running the System

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the system:
   ```bash
   python main.py
   # or
   python run_system.py
   ```
3. Interact via API or frontend (see `frontend/app.py`).

## Testing

- See `tests/test_pipeline.py` for pipeline tests.
- All mock/sample/test code removed; only real agent logic is tested.

## Requirements

- Python 3.12+
- FastAPI
- Pydantic v2
- Streamlit
- PySR
- Lean 4
- Ollama
- Docker
- httpx
- pytest

## Roadmap

- Improve multi-agent LLM service for consensus/validation
- Expand theorem base and agent capabilities
- Enhance frontend and API

## License

- See `licence.md`
- Structured JSON response validation

⚡ **Production Ready**
- FastAPI backend with async support
- Streamlit frontend for interactive exploration
- Comprehensive test suite with pytest
- Docker support and cloud deployment ready

## System Architecture

```
Input Data → Multi-Agent Analysis → Symbolic Regression → Proof Generation → Results
    ↓              ↓                      ↓                   ↓            ↓
  Validation → Hypothesis Gen → Pattern Discovery → Formalization → Report
```

**Components:**
- **Reasoning Orchestrator**: Coordinates the complete Artifact Reason pipeline
- **Multi-Agent System**: Specialized agents for different reasoning tasks
- **LLM Service**: Abstracted language model interface with auto-setup
- **Symbolic Regressor**: Mathematical relationship discovery
- **Proof System**: Lean 4 interface with simulation fallback
- **Knowledge Base**: Graph-based storage and retrieval

## Usage Examples

### Command Line
```bash
# Run complete Artifact Reason analysis
python run_system.py

# Start API server
uvicorn main:app --reload

# Launch interactive frontend  
streamlit run frontend/app.py

# Run test suite
pytest tests/ -v
```

### API Usage
```python
import httpx

# Analyze mathematical sequence
response = httpx.post("http://localhost:8000/reason", json={
    "data": [1, 4, 9, 16, 25, 36],
    "context": "Perfect squares sequence"
})

results = response.json()
```

### Python Integration
```python
from core.reasoning_orchestrator import EnhancedReasoningOrchestrator
import asyncio

async def analyze_pattern():
    orchestrator = EnhancedReasoningOrchestrator()
    results = await orchestrator.orchestrate_reasoning([1, 1, 2, 3, 5, 8])
    return results

# Run analysis
results = asyncio.run(analyze_pattern())
```

## Author: Artifact Virtual — We Don’t Build Demos. We Build Dominance.

