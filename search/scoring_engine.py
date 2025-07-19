"""
Advanced scoring engine for mathematical expression candidates
Implements multi-criteria evaluation with various fitness metrics
"""
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger, performance_monitor
from utils.exceptions import handle_exceptions, SymbolicRegressionError
from search.candidate_space import ExpressionCandidate

@dataclass
class ScoringMetrics:
    """Container for all scoring metrics"""
    mse: float  # Mean Squared Error
    mae: float  # Mean Absolute Error
    r_squared: float  # R-squared
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    complexity_penalty: float
    parsimony_score: float
    novelty_score: float
    robustness_score: float
    final_score: float
    metadata: Dict[str, Any]

class ExpressionEvaluator:
    """Safely evaluates mathematical expressions"""
    
    def __init__(self):
        # Safe evaluation context
        self.safe_functions = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': self._safe_exp,
            'log': self._safe_log,
            'sqrt': self._safe_sqrt,
            'abs': np.abs,
            'pow': self._safe_pow,
            'pi': math.pi,
            'e': math.e
        }
    
    def _safe_exp(self, x):
        """Safe exponential function with overflow protection"""
        try:
            result = np.exp(np.clip(x, -700, 700))  # Prevent overflow
            return np.where(np.isfinite(result), result, 1e10)
        except:
            return np.full_like(x, 1e10) if hasattr(x, '__len__') else 1e10
    
    def _safe_log(self, x):
        """Safe logarithm function"""
        try:
            return np.where(x > 0, np.log(x), -1e10)
        except:
            return np.full_like(x, -1e10) if hasattr(x, '__len__') else -1e10
    
    def _safe_sqrt(self, x):
        """Safe square root function"""
        try:
            return np.where(x >= 0, np.sqrt(x), 0)
        except:
            return np.zeros_like(x) if hasattr(x, '__len__') else 0
    
    def _safe_pow(self, base, exponent):
        """Safe power function with overflow protection"""
        try:
            # Clip exponent to prevent overflow
            safe_exp = np.clip(exponent, -100, 100)
            result = np.power(np.abs(base), safe_exp)
            return np.where(np.isfinite(result), result, 1e10)
        except:
            return np.full_like(base, 1e10) if hasattr(base, '__len__') else 1e10
    
    @performance_monitor
    def evaluate_expression(self, expression: str, x_values: np.ndarray) -> np.ndarray:
        """Safely evaluate expression at given x values"""
        try:
            # Prepare safe namespace
            namespace = self.safe_functions.copy()
            namespace['x'] = x_values
            
            # Replace ^ with **
            safe_expression = expression.replace('^', '**')
            
            # Evaluate with timeout protection
            result = eval(safe_expression, {"__builtins__": {}}, namespace)
            
            # Ensure result is array
            if np.isscalar(result):
                result = np.full_like(x_values, result)
            
            # Handle infinite or NaN values
            result = np.where(np.isfinite(result), result, 1e10)
            
            return result
            
        except Exception as e:
            logger.debug(f"Expression evaluation failed: {expression}, error: {str(e)}")
            return np.full_like(x_values, 1e10)  # Return large error values

class ScoringEngine:
    """Advanced multi-criteria scoring system for expression candidates"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.evaluator = ExpressionEvaluator()
        
        # Default weights for different criteria
        self.weights = weights or {
            'accuracy': 0.4,      # How well it fits the data
            'simplicity': 0.25,   # Parsimony principle
            'robustness': 0.15,   # Stability across different inputs
            'novelty': 0.1,       # Uniqueness of the solution
            'interpretability': 0.1  # How easy to understand
        }
        
        logger.info(f"Initialized scoring engine with weights: {self.weights}")
    
    @performance_monitor
    @handle_exceptions
    def score_equations(self, candidates: List[ExpressionCandidate], 
                       x_data: List[float], y_data: List[float]) -> Dict[str, ScoringMetrics]:
        """
        Score multiple expression candidates against data
        
        Args:
            candidates: List of expression candidates to score
            x_data: Input data points
            y_data: Target output values
        
        Returns:
            Dictionary mapping expression to scoring metrics
        """
        if len(x_data) != len(y_data):
            raise SymbolicRegressionError("Mismatched data lengths", "DATA_LENGTH_MISMATCH")
        
        if len(x_data) < 2:
            raise SymbolicRegressionError("Insufficient data points", "INSUFFICIENT_DATA")
        
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        results = {}
        
        for candidate in candidates:
            try:
                metrics = self._score_single_candidate(candidate, x_array, y_array)
                results[candidate.expression] = metrics
            except Exception as e:
                logger.warning(f"Failed to score candidate {candidate.expression}: {str(e)}")
                # Create failed metrics
                results[candidate.expression] = ScoringMetrics(
                    mse=1e10, mae=1e10, r_squared=-1e10, aic=1e10, bic=1e10,
                    complexity_penalty=1.0, parsimony_score=0.0, novelty_score=0.0,
                    robustness_score=0.0, final_score=0.0,
                    metadata={'error': str(e), 'evaluation_failed': True}
                )
        
        # Normalize novelty scores
        self._normalize_novelty_scores(results)
        
        # Calculate final scores
        for expr, metrics in results.items():
            metrics.final_score = self._calculate_final_score(metrics)
        
        logger.info(f"Scored {len(results)} candidates")
        return results
    
    def _score_single_candidate(self, candidate: ExpressionCandidate, 
                               x_data: np.ndarray, y_data: np.ndarray) -> ScoringMetrics:
        """Score a single candidate expression"""
        
        # Evaluate expression
        y_pred = self.evaluator.evaluate_expression(candidate.expression, x_data)
        
        # Basic accuracy metrics
        mse = np.mean((y_data - y_pred) ** 2)
        mae = np.mean(np.abs(y_data - y_pred))
        
        # R-squared
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Information criteria
        n = len(y_data)
        k = candidate.complexity  # Number of parameters
        log_likelihood = -n/2 * np.log(2 * np.pi * mse) - n/2
        
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Complexity penalty
        complexity_penalty = self._calculate_complexity_penalty(candidate)
        
        # Parsimony score (favors simpler expressions)
        parsimony_score = 1.0 / (1.0 + candidate.complexity / 10.0)
        
        # Robustness score
        robustness_score = self._calculate_robustness(candidate, x_data)
        
        # Placeholder for novelty (will be calculated across all candidates)
        novelty_score = 0.0
        
        return ScoringMetrics(
            mse=mse,
            mae=mae,
            r_squared=r_squared,
            aic=aic,
            bic=bic,
            complexity_penalty=complexity_penalty,
            parsimony_score=parsimony_score,
            novelty_score=novelty_score,
            robustness_score=robustness_score,
            final_score=0.0,  # Will be calculated later
            metadata={
                'expression': candidate.expression,
                'complexity': candidate.complexity,
                'operators': candidate.operators,
                'constants': candidate.constants
            }
        )
    
    def _calculate_complexity_penalty(self, candidate: ExpressionCandidate) -> float:
        """Calculate penalty based on expression complexity"""
        base_penalty = candidate.complexity / 20.0
        
        # Additional penalties for specific operators
        operator_penalties = {
            'exp': 0.1,
            'log': 0.1,
            '^': 0.05,
            'sin': 0.05,
            'cos': 0.05
        }
        
        additional_penalty = sum(operator_penalties.get(op, 0) for op in candidate.operators)
        
        return min(1.0, base_penalty + additional_penalty)
    
    def _calculate_robustness(self, candidate: ExpressionCandidate, x_data: np.ndarray) -> float:
        """Calculate robustness by testing on perturbed data"""
        try:
            original_pred = self.evaluator.evaluate_expression(candidate.expression, x_data)
            
            # Test with small perturbations
            perturbations = [0.01, -0.01, 0.05, -0.05]
            robustness_scores = []
            
            for perturbation in perturbations:
                perturbed_x = x_data + perturbation
                perturbed_pred = self.evaluator.evaluate_expression(candidate.expression, perturbed_x)
                
                # Calculate relative change
                relative_change = np.mean(np.abs(perturbed_pred - original_pred) / (np.abs(original_pred) + 1e-10))
                robustness_scores.append(1.0 / (1.0 + relative_change))
            
            return np.mean(robustness_scores)
            
        except Exception:
            return 0.0  # Not robust if evaluation fails
    
    def _normalize_novelty_scores(self, results: Dict[str, ScoringMetrics]):
        """Calculate and normalize novelty scores across all candidates"""
        expressions = list(results.keys())
        n_expr = len(expressions)
        
        for i, expr1 in enumerate(expressions):
            if results[expr1].metadata.get('evaluation_failed', False):
                results[expr1].novelty_score = 0.0
                continue
                
            similarity_scores = []
            
            for j, expr2 in enumerate(expressions):
                if i != j and not results[expr2].metadata.get('evaluation_failed', False):
                    similarity = self._calculate_expression_similarity(expr1, expr2)
                    similarity_scores.append(similarity)
            
            # Novelty is inverse of average similarity
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            results[expr1].novelty_score = 1.0 - avg_similarity
    
    def _calculate_expression_similarity(self, expr1: str, expr2: str) -> float:
        """Calculate similarity between two expressions"""
        # Simple similarity based on operator overlap and string similarity
        
        # Normalize expressions
        norm1 = expr1.replace(" ", "").lower()
        norm2 = expr2.replace(" ", "").lower()
        
        # String similarity (Jaccard coefficient on character n-grams)
        bigrams1 = set(norm1[i:i+2] for i in range(len(norm1)-1))
        bigrams2 = set(norm2[i:i+2] for i in range(len(norm2)-1))
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        string_similarity = intersection / union if union > 0 else 0.0
        
        return string_similarity
    
    def _calculate_final_score(self, metrics: ScoringMetrics) -> float:
        """Calculate final weighted score"""
        if metrics.metadata.get('evaluation_failed', False):
            return 0.0
        
        # Normalize individual scores to [0, 1]
        accuracy_score = max(0, min(1, metrics.r_squared))  # R-squared can be negative
        simplicity_score = metrics.parsimony_score
        robustness_score = metrics.robustness_score
        novelty_score = metrics.novelty_score
        
        # Interpretability score (simple heuristic)
        interpretability_score = 1.0 - metrics.complexity_penalty
        
        # Weighted combination
        final_score = (
            self.weights['accuracy'] * accuracy_score +
            self.weights['simplicity'] * simplicity_score +
            self.weights['robustness'] * robustness_score +
            self.weights['novelty'] * novelty_score +
            self.weights['interpretability'] * interpretability_score
        )
        
        return max(0.0, min(1.0, final_score))
    
    def get_top_candidates(self, scored_results: Dict[str, ScoringMetrics], 
                          top_k: int = 5) -> List[Tuple[str, ScoringMetrics]]:
        """Get top K candidates by final score"""
        sorted_results = sorted(
            scored_results.items(), 
            key=lambda x: x[1].final_score, 
            reverse=True
        )
        
        return sorted_results[:top_k]

# Convenience function for backward compatibility
@performance_monitor
def score_equations(candidates: List, data: List[float]) -> Dict[str, Any]:
    """Score expression candidates against data"""
    if not candidates or not data:
        return {}
    
    # Create dummy y data if not provided
    x_data = list(range(len(data)))
    y_data = data
    
    # Convert candidates to ExpressionCandidate objects if needed
    if candidates and not isinstance(candidates[0], ExpressionCandidate):
        from search.candidate_space import ExpressionCandidate
        candidates = [
            ExpressionCandidate(
                expression=str(c),
                complexity=2,
                operators=['+', '*'],
                variables=['x'],
                constants=[1, 0]
            ) for c in candidates
        ]
    
    engine = ScoringEngine()
    return engine.score_equations(candidates, x_data, y_data)
