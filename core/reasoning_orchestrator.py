
"""
Enhanced reasoning orchestrator with multi-agent system integration
Coordinates traditional reasoning pipeline with multi-agent validation
"""
import asyncio
from typing import Dict, List, Any, Optional
import time

from core.abductive_engine import generate_abductive_hypotheses
from core.analogical_engine import find_analogies
from core.symbolic_regressor import run_symbolic_regression
from core.meta_reasoner import evaluate_meta_reasoning
from core.multi_agent_system import MultiAgentOrchestrator
from llm.autoformalizer import autoformalize
from proofs.lean_interface import prove_async, get_proof_checker
from search.candidate_space import generate_candidates
from search.scoring_engine import ScoringEngine
from utils.logger import logger, async_performance_monitor
from utils.exceptions import handle_exceptions, ReasoningError
import numpy as np

class EnhancedReasoningOrchestrator:
    def _analyze_prediction_agreement(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze agreement between different prediction approaches"""
        hypotheses = [pred["hypothesis"] for pred in predictions.values() if pred["hypothesis"]]
        confidences = [pred["confidence"] for pred in predictions.values()]
        if len(hypotheses) >= 2:
            similarities = []
            for i in range(len(hypotheses)):
                for j in range(i+1, len(hypotheses)):
                    sim = self._calculate_string_similarity(hypotheses[i], hypotheses[j])
                    similarities.append(sim)
            avg_similarity = np.mean(similarities) if similarities else 0.0
        else:
            avg_similarity = 0.0
        if len(confidences) >= 2:
            confidence_std = np.std(confidences)
            confidence_agreement = max(0, 1.0 - confidence_std)
        else:
            confidence_agreement = 1.0
        overall_agreement = (avg_similarity + confidence_agreement) / 2
        return {
            "hypothesis_similarity": avg_similarity,
            "confidence_agreement": confidence_agreement,
            "overall_agreement": overall_agreement,
            "hypothesis_count": len(hypotheses),
            "approaches_used": len(predictions)
        }

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        str1 = str(str1) if str1 is not None else ""
        str2 = str(str2) if str2 is not None else ""
        if not str1 or not str2:
            return 0.0
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _generate_consensus_flags(self, agreement_analysis: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """Generate warning flags based on consensus analysis"""
        flags = []
        if agreement_analysis["overall_agreement"] < 0.3:
            flags.append("LOW_CONSENSUS")
        if agreement_analysis["hypothesis_similarity"] < 0.2:
            flags.append("DIVERGENT_HYPOTHESES")
        if agreement_analysis["confidence_agreement"] < 0.3:
            flags.append("CONFIDENCE_DISAGREEMENT")
        low_confidence = [k for k, v in predictions.items() if v["confidence"] < 0.3]
        if low_confidence:
            flags.append(f"LOW_CONFIDENCE_APPROACHES: {', '.join(low_confidence)}")
        return flags

    async def _verify_final_proofs(self, consensus_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify proofs for consensus hypotheses"""
        hypothesis = consensus_results.get("consensus_hypothesis", "")
        if not hypothesis:
            return {
                "proof_attempted": False,
                "error": "No hypothesis to prove"
            }
        try:
            formal_expression = autoformalize(hypothesis)
            proof_result = await prove_async(formal_expression)
            return {
                "proof_attempted": True,
                "formal_expression": formal_expression,
                "proof_result": proof_result.__dict__,
                "is_provable": proof_result.is_valid,
                "proof_confidence": proof_result.confidence,
                "verification_time": proof_result.verification_time
            }
        except Exception as e:
            logger.error(f"Proof verification failed: {str(e)}")
            return {
                "proof_attempted": True,
                "error": str(e),
                "is_provable": False,
                "proof_confidence": 0.0
            }

    def _generate_final_recommendation(self, consensus: Dict[str, Any], proof: Dict[str, Any], multi_agent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation based on all analyses"""
        recommendation_confidence = consensus.get("consensus_confidence", 0.0)
        if proof.get("is_provable", False):
            recommendation_confidence += 0.1
        ma_quality = multi_agent.get("reasoning_quality", {}).get("overall_quality", 0.0)
        recommendation_confidence = (recommendation_confidence + ma_quality) / 2
        if recommendation_confidence >= 0.8:
            strength = "HIGH"
        elif recommendation_confidence >= 0.6:
            strength = "MEDIUM"
        elif recommendation_confidence >= 0.4:
            strength = "LOW"
        else:
            strength = "VERY_LOW"
        return {
            "hypothesis": consensus.get("consensus_hypothesis", ""),
            "confidence": recommendation_confidence,
            "strength": strength,
            "supporting_evidence": {
                "consensus_score": consensus.get("consensus_confidence", 0.0),
                "proof_verified": proof.get("is_provable", False),
                "multi_agent_quality": ma_quality,
                "best_approach": consensus.get("best_approach", "unknown")
            },
            "warnings": consensus.get("consensus_flags", []) + multi_agent.get("quality_flags", [])
        }

    def _assess_overall_quality(self, traditional: Dict[str, Any], multi_agent: Dict[str, Any], symbolic: Dict[str, Any], proof: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of the reasoning process"""
        quality_factors = {
            "traditional_confidence": traditional.get("pipeline_confidence", 0.0),
            "multi_agent_quality": multi_agent.get("reasoning_quality", {}).get("overall_quality", 0.0),
            "symbolic_performance": symbolic.get("overall_confidence", 0.0),
            "proof_success": 1.0 if proof.get("is_provable", False) else 0.0
        }
        weights = {"traditional_confidence": 0.2, "multi_agent_quality": 0.4, "symbolic_performance": 0.3, "proof_success": 0.1}
        overall_score = sum(weights[factor] * score for factor, score in quality_factors.items())
        return {
            "individual_scores": quality_factors,
            "overall_score": overall_score,
            "quality_level": self._categorize_quality(overall_score),
            "recommendations": self._generate_quality_recommendations(quality_factors)
        }

    def _categorize_quality(self, score: float) -> str:
        """Categorize quality score"""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.7:
            return "GOOD"
        elif score >= 0.5:
            return "FAIR"
        elif score >= 0.3:
            return "POOR"
        else:
            return "VERY_POOR"

    def _generate_quality_recommendations(self, quality_factors: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving quality"""
        recommendations = []
        if quality_factors["traditional_confidence"] < 0.5:
            recommendations.append("Consider providing more or higher quality input data")
        if quality_factors["multi_agent_quality"] < 0.5:
            recommendations.append("Multi-agent validation identified potential issues")
        if quality_factors["symbolic_performance"] < 0.5:
            recommendations.append("Symbolic regression struggled to find good fits")
        if quality_factors["proof_success"] < 0.5:
            recommendations.append("Formal proof verification failed or was not attempted")
        if not recommendations:
            recommendations.append("Results appear to be of high quality")
        return recommendations
    """Enhanced orchestrator with multi-agent reasoning and validation"""
    
    def __init__(self):
        self.multi_agent_system = MultiAgentOrchestrator()
        self.scoring_engine = ScoringEngine()
        self.proof_checker = get_proof_checker()
        logger.info("Initialized enhanced reasoning orchestrator")

    @async_performance_monitor
    @handle_exceptions
    async def orchestrate_reasoning(self, data: List[float], known_theorems: List[Dict] = None) -> Dict[str, Any]:
        import json
        import os
        # Always load and reflect on the exhaustive theorem list before reasoning
        theorems_path = os.path.join(os.path.dirname(__file__), '../data/known_theorems.json')
        try:
            with open(theorems_path, 'r') as f:
                all_theorems = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load known_theorems.json: {e}")
            all_theorems = []
        # Merge user-provided theorems with the exhaustive set, avoiding duplicates
        user_theorems = known_theorems or []
        all_names = {t['name'] for t in all_theorems if 'name' in t}
        merged_theorems = all_theorems + [t for t in user_theorems if t.get('name') not in all_names]
        # The system should reflect on these theorems and use them for reasoning, but never attempt to re-prove them
        # You can pass merged_theorems to all downstream agents and logic as needed
        """
        Complete reasoning pipeline with multi-agent validation
        
        Args:
            data: Input numerical data
            known_theorems: List of known mathematical theorems
        
        Returns:
            Comprehensive reasoning results with validation
        """
        start_time = time.time()
        known_theorems = merged_theorems
        try:
            # Phase 1: Traditional reasoning pipeline
            logger.info("Starting traditional reasoning pipeline")
            traditional_results = await self._run_traditional_pipeline(data, known_theorems)
            # Phase 2: Multi-agent reasoning and validation
            logger.info("Starting multi-agent reasoning")
            multi_agent_input = {
                "data": data,
                "context": f"Traditional pipeline results: {traditional_results}",
                "known_theorems": known_theorems
            }
            multi_agent_results = await self.multi_agent_system.orchestrate_reasoning(multi_agent_input)
            # Phase 3: Advanced symbolic regression with candidate scoring
            logger.info("Running advanced symbolic regression")
            symbolic_results = await self._run_advanced_symbolic_regression(data)
            # Phase 4: Cross-validation and consensus building
            logger.info("Building cross-validation consensus")
            consensus_results = await self._build_cross_validation_consensus(
                traditional_results, multi_agent_results, symbolic_results
            )
            # Phase 5: Final proof verification
            logger.info("Performing final proof verification")
            proof_results = await self._verify_final_proofs(consensus_results)
            # Compile comprehensive results
            final_results = {
                "execution_metadata": {
                    "total_time": time.time() - start_time,
                    "pipeline_version": "enhanced_v1.0",
                    "agent_count": 4,
                    "validation_layers": 3
                },
                "traditional_pipeline": traditional_results,
                "multi_agent_analysis": multi_agent_results,
                "symbolic_regression": symbolic_results,
                "cross_validation": consensus_results,
                "proof_verification": proof_results,
                "final_recommendation": self._generate_final_recommendation(
                    consensus_results, proof_results, multi_agent_results
                ),
                "quality_assessment": self._assess_overall_quality(
                    traditional_results, multi_agent_results, symbolic_results, proof_results
                )
            }
            logger.info("Completed enhanced reasoning orchestration", 
                       total_time=final_results["execution_metadata"]["total_time"],
                       quality_score=final_results["quality_assessment"]["overall_score"])
            return final_results
        except Exception as e:
            logger.error(f"Reasoning orchestration failed: {str(e)}")
            raise ReasoningError("orchestration", str(e))

    async def _run_traditional_pipeline(self, data: List[float], known_theorems: List[Dict]) -> Dict[str, Any]:
        """Run the traditional reasoning pipeline"""
        hypotheses = generate_abductive_hypotheses(data)
        analogies = find_analogies(hypotheses[0] if hypotheses else "", known_theorems)
        try:
            symbolic = run_symbolic_regression(list(range(len(data))), data)
        except Exception as e:
            logger.warning(f"Symbolic regression failed: {str(e)}")
            symbolic = {"equation": "y = x", "score": 0.0}
        meta = evaluate_meta_reasoning(hypotheses)
        formal = autoformalize(hypotheses[0] if hypotheses else "y = x")
        return {
            "hypotheses": hypotheses,
            "analogies": analogies,
            "symbolic_model": symbolic,
            "meta_reasoning": meta,
            "formal_expression": formal,
            "pipeline_confidence": 0.7 if hypotheses else 0.3
        }

    async def _run_advanced_symbolic_regression(self, data: List[float]) -> Dict[str, Any]:
        """Run advanced symbolic regression with multiple strategies"""
        results = {}
        strategies = ["linear", "polynomial", "transcendental", "mixed"]
        for strategy in strategies:
            try:
                candidates = generate_candidates(data, strategy=strategy, population_size=50)
                if candidates:
                    x_data = list(range(len(data)))
                    scoring_results = self.scoring_engine.score_equations(candidates, x_data, data)
                    top_candidates = self.scoring_engine.get_top_candidates(scoring_results, top_k=3)
                    results[strategy] = {
                        "candidates_generated": len(candidates),
                        "top_candidates": top_candidates,
                        "best_score": top_candidates[0][1].final_score if top_candidates else 0.0
                    }
                else:
                    results[strategy] = {"error": "No candidates generated"}
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {str(e)}")
                results[strategy] = {"error": str(e)}
        best_candidate = None
        best_score = 0.0
        for strategy_results in results.values():
            if isinstance(strategy_results, dict) and "top_candidates" in strategy_results:
                candidates = strategy_results["top_candidates"]
                if candidates and candidates[0][1].final_score > best_score:
                    best_score = candidates[0][1].final_score
                    best_candidate = candidates[0]
        return {
            "strategy_results": results,
            "best_candidate": best_candidate,
            "overall_confidence": best_score
        }

    async def _build_cross_validation_consensus(self, traditional: Dict, multi_agent: Dict, symbolic: Dict) -> Dict[str, Any]:
        """Build consensus across different reasoning approaches"""
        predictions = {
            "traditional": {
                "hypothesis": traditional.get("hypotheses", [""])[0],
                "confidence": traditional.get("pipeline_confidence", 0.0)
            },
            "multi_agent": {
                "hypothesis": multi_agent.get("final_consensus", {}).get("final_recommendation", {}),
                "confidence": multi_agent.get("overall_confidence", 0.0)
            },
            "symbolic": {
                "hypothesis": symbolic.get("best_candidate", ["", None])[0] if symbolic.get("best_candidate") else "",
                "confidence": symbolic.get("overall_confidence", 0.0)
            }
        }
        agreement_analysis = self._analyze_prediction_agreement(predictions)
        weights = {
            "traditional": 0.3,
            "multi_agent": 0.4,
            "symbolic": 0.3
        }
        consensus_confidence = sum(
            weights[approach] * pred["confidence"] 
            for approach, pred in predictions.items()
        )
        best_approach = max(predictions.keys(), key=lambda k: predictions[k]["confidence"])
        consensus_hypothesis = predictions[best_approach]["hypothesis"]
        agreement_boost = agreement_analysis["overall_agreement"] * 0.2
        final_confidence = min(1.0, consensus_confidence + agreement_boost)
        return {
            "predictions": predictions,
            "agreement_analysis": agreement_analysis,
            "consensus_hypothesis": consensus_hypothesis,
            "consensus_confidence": final_confidence,
            "best_approach": best_approach,
            "consensus_flags": self._generate_consensus_flags(agreement_analysis, predictions)
        }
    
_orchestrator_instance = None

def get_orchestrator():
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = EnhancedReasoningOrchestrator()
    return _orchestrator_instance
    
    async def _run_traditional_pipeline(self, data: List[float], known_theorems: List[Dict]) -> Dict[str, Any]:
        """Run the traditional reasoning pipeline"""
        # Generate hypotheses
        hypotheses = generate_abductive_hypotheses(data)
        
        # Find analogies
        analogies = find_analogies(hypotheses[0] if hypotheses else "", known_theorems)
        
        # Run symbolic regression
        try:
            symbolic = run_symbolic_regression(list(range(len(data))), data)
        except Exception as e:
            logger.warning(f"Symbolic regression failed: {str(e)}")
            symbolic = {"equation": "y = x", "score": 0.0}
        
        # Meta reasoning
        meta = evaluate_meta_reasoning(hypotheses)
        
        # Auto-formalization
        formal = autoformalize(hypotheses[0] if hypotheses else "y = x")
        
        return {
            "hypotheses": hypotheses,
            "analogies": analogies,
            "symbolic_model": symbolic,
            "meta_reasoning": meta,
            "formal_expression": formal,
            "pipeline_confidence": 0.7 if hypotheses else 0.3
        }
    
    async def _run_advanced_symbolic_regression(self, data: List[float]) -> Dict[str, Any]:
        """Run advanced symbolic regression with multiple strategies"""
        results = {}
        
        # Generate candidates using different strategies
        strategies = ["linear", "polynomial", "transcendental", "mixed"]
        
        for strategy in strategies:
            try:
                candidates = generate_candidates(data, strategy=strategy, population_size=50)
                
                if candidates:
                    # Score candidates
                    x_data = list(range(len(data)))
                    scoring_results = self.scoring_engine.score_equations(candidates, x_data, data)
                    
                    # Get top candidates
                    top_candidates = self.scoring_engine.get_top_candidates(scoring_results, top_k=3)
                    
                    results[strategy] = {
                        "candidates_generated": len(candidates),
                        "top_candidates": top_candidates,
                        "best_score": top_candidates[0][1].final_score if top_candidates else 0.0
                    }
                else:
                    results[strategy] = {"error": "No candidates generated"}
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {str(e)}")
                results[strategy] = {"error": str(e)}
        
        # Find overall best candidate
        best_candidate = None
        best_score = 0.0
        
        for strategy_results in results.values():
            if isinstance(strategy_results, dict) and "top_candidates" in strategy_results:
                candidates = strategy_results["top_candidates"]
                if candidates and candidates[0][1].final_score > best_score:
                    best_score = candidates[0][1].final_score
                    best_candidate = candidates[0]
        
        return {
            "strategy_results": results,
            "best_candidate": best_candidate,
            "overall_confidence": best_score
        }
    
    async def _build_cross_validation_consensus(self, traditional: Dict, multi_agent: Dict, symbolic: Dict) -> Dict[str, Any]:
        """Build consensus across different reasoning approaches"""
        
        # Extract key predictions from each approach
        predictions = {
            "traditional": {
                "hypothesis": traditional.get("hypotheses", [""])[0],
                "confidence": traditional.get("pipeline_confidence", 0.0)
            },
            "multi_agent": {
                "hypothesis": multi_agent.get("final_consensus", {}).get("final_recommendation", {}),
                "confidence": multi_agent.get("overall_confidence", 0.0)
            },
            "symbolic": {
                "hypothesis": symbolic.get("best_candidate", ["", None])[0] if symbolic.get("best_candidate") else "",
                "confidence": symbolic.get("overall_confidence", 0.0)
            }
        }
        
        # Calculate agreement scores
        agreement_analysis = self._analyze_prediction_agreement(predictions)
        
        # Build weighted consensus
        weights = {
            "traditional": 0.3,
            "multi_agent": 0.4,  # Higher weight for multi-agent validation
            "symbolic": 0.3
        }
        
        consensus_confidence = sum(
            weights[approach] * pred["confidence"] 
            for approach, pred in predictions.items()
        )
        
        # Determine consensus hypothesis
        # Choose the highest confidence prediction, but adjust for agreement
        best_approach = max(predictions.keys(), key=lambda k: predictions[k]["confidence"])
        consensus_hypothesis = predictions[best_approach]["hypothesis"]
        
        # Adjust confidence based on agreement
        agreement_boost = agreement_analysis["overall_agreement"] * 0.2
        final_confidence = min(1.0, consensus_confidence + agreement_boost)
        
        return {
            "predictions": predictions,
            "agreement_analysis": agreement_analysis,
            "consensus_hypothesis": consensus_hypothesis,
            "consensus_confidence": final_confidence,
            "best_approach": best_approach,
            "consensus_flags": self._generate_consensus_flags(agreement_analysis, predictions)
        }
    
    def _analyze_prediction_agreement(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze agreement between different prediction approaches"""
        
        hypotheses = [pred["hypothesis"] for pred in predictions.values() if pred["hypothesis"]]
        confidences = [pred["confidence"] for pred in predictions.values()]
        
        # Simple string similarity for hypotheses
        if len(hypotheses) >= 2:
            similarities = []
            for i in range(len(hypotheses)):
                for j in range(i+1, len(hypotheses)):
                    sim = self._calculate_string_similarity(hypotheses[i], hypotheses[j])
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
        else:
            avg_similarity = 0.0
        
        # Confidence agreement (how similar are the confidence scores)
        if len(confidences) >= 2:
            confidence_std = np.std(confidences)
            confidence_agreement = max(0, 1.0 - confidence_std)  # Lower std = higher agreement
        else:
            confidence_agreement = 1.0
        
        overall_agreement = (avg_similarity + confidence_agreement) / 2
        
        return {
            "hypothesis_similarity": avg_similarity,
            "confidence_agreement": confidence_agreement,
            "overall_agreement": overall_agreement,
            "hypothesis_count": len(hypotheses),
            "approaches_used": len(predictions)
        }
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Convert to string if not already a string
        str1 = str(str1) if str1 is not None else ""
        str2 = str(str2) if str2 is not None else ""
        
        if not str1 or not str2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_consensus_flags(self, agreement_analysis: Dict, predictions: Dict) -> List[str]:
        """Generate warning flags based on consensus analysis"""
        flags = []
        
        if agreement_analysis["overall_agreement"] < 0.3:
            flags.append("LOW_CONSENSUS")
        
        if agreement_analysis["hypothesis_similarity"] < 0.2:
            flags.append("DIVERGENT_HYPOTHESES")
        
        if agreement_analysis["confidence_agreement"] < 0.3:
            flags.append("CONFIDENCE_DISAGREEMENT")
        
        # Check for very low confidence predictions
        low_confidence = [k for k, v in predictions.items() if v["confidence"] < 0.3]
        if low_confidence:
            flags.append(f"LOW_CONFIDENCE_APPROACHES: {', '.join(low_confidence)}")
        
        return flags
    
    async def _verify_final_proofs(self, consensus_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify proofs for consensus hypotheses"""
        
        hypothesis = consensus_results.get("consensus_hypothesis", "")
        
        if not hypothesis:
            return {
                "proof_attempted": False,
                "error": "No hypothesis to prove"
            }
        
        try:
            # Auto-formalize the hypothesis
            formal_expression = autoformalize(hypothesis)
            
            # Attempt proof verification
            proof_result = await prove_async(formal_expression)
            
            return {
                "proof_attempted": True,
                "formal_expression": formal_expression,
                "proof_result": proof_result.__dict__,
                "is_provable": proof_result.is_valid,
                "proof_confidence": proof_result.confidence,
                "verification_time": proof_result.verification_time
            }
            
        except Exception as e:
            logger.error(f"Proof verification failed: {str(e)}")
            return {
                "proof_attempted": True,
                "error": str(e),
                "is_provable": False,
                "proof_confidence": 0.0
            }
    
    def _generate_final_recommendation(self, consensus: Dict, proof: Dict, multi_agent: Dict) -> Dict[str, Any]:
        """Generate final recommendation based on all analyses"""
        
        recommendation_confidence = consensus.get("consensus_confidence", 0.0)
        
        # Boost confidence if proof succeeded
        if proof.get("is_provable", False):
            recommendation_confidence += 0.1
        
        # Consider multi-agent quality assessment
        ma_quality = multi_agent.get("reasoning_quality", {}).get("overall_quality", 0.0)
        recommendation_confidence = (recommendation_confidence + ma_quality) / 2
        
        # Determine recommendation strength
        if recommendation_confidence >= 0.8:
            strength = "HIGH"
        elif recommendation_confidence >= 0.6:
            strength = "MEDIUM"
        elif recommendation_confidence >= 0.4:
            strength = "LOW"
        else:
            strength = "VERY_LOW"
        
        return {
            "hypothesis": consensus.get("consensus_hypothesis", ""),
            "confidence": recommendation_confidence,
            "strength": strength,
            "supporting_evidence": {
                "consensus_score": consensus.get("consensus_confidence", 0.0),
                "proof_verified": proof.get("is_provable", False),
                "multi_agent_quality": ma_quality,
                "best_approach": consensus.get("best_approach", "unknown")
            },
            "warnings": consensus.get("consensus_flags", []) + multi_agent.get("quality_flags", [])
        }
    
    def _assess_overall_quality(self, traditional: Dict, multi_agent: Dict, 
                               symbolic: Dict, proof: Dict) -> Dict[str, Any]:
        """Assess overall quality of the reasoning process"""
        
        quality_factors = {
            "traditional_confidence": traditional.get("pipeline_confidence", 0.0),
            "multi_agent_quality": multi_agent.get("reasoning_quality", {}).get("overall_quality", 0.0),
            "symbolic_performance": symbolic.get("overall_confidence", 0.0),
            "proof_success": 1.0 if proof.get("is_provable", False) else 0.0
        }
        
        # Weighted overall score
        weights = {"traditional_confidence": 0.2, "multi_agent_quality": 0.4, 
                  "symbolic_performance": 0.3, "proof_success": 0.1}
        
        overall_score = sum(weights[factor] * score for factor, score in quality_factors.items())
        
        return {
            "individual_scores": quality_factors,
            "overall_score": overall_score,
            "quality_level": self._categorize_quality(overall_score),
            "recommendations": self._generate_quality_recommendations(quality_factors)
        }
    
    def _categorize_quality(self, score: float) -> str:
        """Categorize quality score"""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.7:
            return "GOOD" 
        elif score >= 0.5:
            return "FAIR"
        elif score >= 0.3:
            return "POOR"
        else:
            return "VERY_POOR"
    
    def _generate_quality_recommendations(self, quality_factors: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving quality"""
        recommendations = []
        
        if quality_factors["traditional_confidence"] < 0.5:
            recommendations.append("Consider providing more or higher quality input data")
        
        if quality_factors["multi_agent_quality"] < 0.5:
            recommendations.append("Multi-agent validation identified potential issues")
        
        if quality_factors["symbolic_performance"] < 0.5:
            recommendations.append("Symbolic regression struggled to find good fits")
        
        if quality_factors["proof_success"] < 0.5:
            recommendations.append("Formal proof verification failed or was not attempted")
        
        if not recommendations:
            recommendations.append("Results appear to be of high quality")
        
        return recommendations




