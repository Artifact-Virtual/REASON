"""
Advanced candidate generation for symbolic expressions
Implements multiple strategies for generating mathematical expression candidates
"""
import itertools
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import random
from dataclasses import dataclass
import math

from utils.logger import logger, performance_monitor
from utils.exceptions import handle_exceptions, InsufficientDataError

@dataclass
class ExpressionCandidate:
    """Represents a candidate mathematical expression"""
    expression: str
    complexity: int
    operators: List[str]
    variables: List[str]
    constants: List[float]
    fitness_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class CandidateGenerator:
    """Advanced candidate generation system"""
    
    def __init__(self, max_complexity: int = 10, population_size: int = 100):
        self.max_complexity = max_complexity
        self.population_size = population_size
        
        # Define operator sets
        self.unary_operators = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs']
        self.binary_operators = ['+', '-', '*', '/', '^', 'pow']
        self.constants_pool = [0, 1, -1, 2, -2, 0.5, -0.5, math.pi, math.e]
        
        logger.info(f"Initialized candidate generator with complexity {max_complexity}, population {population_size}")
    
    @performance_monitor
    @handle_exceptions
    def generate_candidates(self, data: List[float], target: List[float] = None, 
                          strategy: str = "mixed") -> List[ExpressionCandidate]:
        """
        Generate candidates using specified strategy
        
        Args:
            data: Input data points
            target: Target values (if available)
            strategy: Generation strategy ('linear', 'polynomial', 'transcendental', 'mixed')
        """
        if len(data) < 2:
            raise InsufficientDataError(len(data), 2)
        
        candidates = []
        
        if strategy == "linear" or strategy == "mixed":
            candidates.extend(self._generate_linear_candidates(data, target))
        
        if strategy == "polynomial" or strategy == "mixed":
            candidates.extend(self._generate_polynomial_candidates(data, target))
        
        if strategy == "transcendental" or strategy == "mixed":
            candidates.extend(self._generate_transcendental_candidates(data, target))
        
        if strategy == "mixed":
            candidates.extend(self._generate_random_candidates(data))
        
        # Remove duplicates and sort by complexity
        unique_candidates = self._remove_duplicates(candidates)
        sorted_candidates = sorted(unique_candidates, key=lambda x: x.complexity)
        
        # Limit to population size
        final_candidates = sorted_candidates[:self.population_size]
        
        logger.info(f"Generated {len(final_candidates)} unique candidates using {strategy} strategy")
        return final_candidates
    
    def _generate_linear_candidates(self, data: List[float], target: List[float] = None) -> List[ExpressionCandidate]:
        """Generate linear expression candidates"""
        candidates = []
        
        # Simple linear: y = ax + b
        if target and len(data) == len(target):
            # Use least squares to estimate parameters
            X = np.array(data).reshape(-1, 1)
            X_with_intercept = np.column_stack([X, np.ones(len(X))])
            y = np.array(target)
            
            try:
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                a, b = coeffs[0], coeffs[1]
                
                expression = f"{a:.4f} * x + {b:.4f}"
                candidates.append(ExpressionCandidate(
                    expression=expression,
                    complexity=2,
                    operators=['+', '*'],
                    variables=['x'],
                    constants=[a, b],
                    metadata={'method': 'least_squares'}
                ))
            except np.linalg.LinAlgError:
                pass
        
        # Generate various linear forms
        for a in [-2, -1, -0.5, 0.5, 1, 2]:
            for b in [-2, -1, 0, 1, 2]:
                if a != 0:  # Avoid trivial constant
                    expression = f"{a} * x + {b}" if b != 0 else f"{a} * x"
                    candidates.append(ExpressionCandidate(
                        expression=expression,
                        complexity=2 if b != 0 else 1,
                        operators=['+', '*'] if b != 0 else ['*'],
                        variables=['x'],
                        constants=[a, b] if b != 0 else [a],
                        metadata={'method': 'systematic_linear'}
                    ))
        
        return candidates
    
    def _generate_polynomial_candidates(self, data: List[float], target: List[float] = None) -> List[ExpressionCandidate]:
        """Generate polynomial expression candidates"""
        candidates = []
        
        # Generate polynomials up to degree 4
        for degree in range(2, min(5, len(data))):
            # Simple polynomial forms
            for coeffs in itertools.product([-2, -1, 0, 1, 2], repeat=degree+1):
                if any(c != 0 for c in coeffs):  # At least one non-zero coefficient
                    terms = []
                    operators = []
                    constants = []
                    
                    for i, coeff in enumerate(coeffs):
                        if coeff != 0:
                            power = degree - i
                            if power == 0:
                                terms.append(f"{coeff}")
                                constants.append(coeff)
                            elif power == 1:
                                terms.append(f"{coeff} * x")
                                constants.append(coeff)
                                operators.extend(['*'])
                            else:
                                terms.append(f"{coeff} * x^{power}")
                                constants.append(coeff)
                                operators.extend(['*', '^'])
                    
                    if terms:
                        expression = " + ".join(terms).replace(" + -", " - ")
                        candidates.append(ExpressionCandidate(
                            expression=expression,
                            complexity=len(constants) + len([op for op in operators if op in ['^']]),
                            operators=list(set(operators + ['+'])),
                            variables=['x'],
                            constants=constants,
                            metadata={'method': 'polynomial', 'degree': degree}
                        ))
                        
                        if len(candidates) > 50:  # Limit polynomial candidates
                            break
            
            if len(candidates) > 50:
                break
        
        return candidates
    
    def _generate_transcendental_candidates(self, data: List[float], target: List[float] = None) -> List[ExpressionCandidate]:
        """Generate transcendental function candidates"""
        candidates = []
        
        # Exponential forms
        for a in [0.5, 1, 2, -0.5, -1, -2]:
            for b in [0, 1, -1, 2, -2]:
                expression = f"{a} * exp({b} * x)" if b != 0 else f"{a} * exp(x)"
                candidates.append(ExpressionCandidate(
                    expression=expression,
                    complexity=3 if b != 0 else 2,
                    operators=['*', 'exp'],
                    variables=['x'],
                    constants=[a, b] if b != 0 else [a],
                    metadata={'method': 'exponential'}
                ))
        
        # Trigonometric forms
        for func in ['sin', 'cos']:
            for a in [1, 2, 0.5]:
                for b in [1, 2, 0.5, math.pi]:
                    expression = f"{a} * {func}({b} * x)"
                    candidates.append(ExpressionCandidate(
                        expression=expression,
                        complexity=3,
                        operators=['*', func],
                        variables=['x'],
                        constants=[a, b],
                        metadata={'method': 'trigonometric'}
                    ))
        
        # Logarithmic forms
        for a in [1, 2, 0.5, -1]:
            for b in [1, 2, 0.5]:
                expression = f"{a} * log({b} * x)"
                candidates.append(ExpressionCandidate(
                    expression=expression,
                    complexity=3,
                    operators=['*', 'log'],
                    variables=['x'],
                    constants=[a, b],
                    metadata={'method': 'logarithmic'}
                ))
        
        return candidates
    
    def _generate_random_candidates(self, data: List[float]) -> List[ExpressionCandidate]:
        """Generate random expression candidates using genetic programming principles"""
        candidates = []
        
        for _ in range(20):  # Generate 20 random candidates
            complexity = random.randint(2, min(self.max_complexity, 6))
            expression, operators, constants = self._build_random_expression(complexity)
            
            candidates.append(ExpressionCandidate(
                expression=expression,
                complexity=complexity,
                operators=operators,
                variables=['x'],
                constants=constants,
                metadata={'method': 'random'}
            ))
        
        return candidates
    
    def _build_random_expression(self, target_complexity: int) -> Tuple[str, List[str], List[float]]:
        """Build a random expression with specified complexity"""
        operators_used = []
        constants_used = []
        
        # Start with a variable or constant
        if random.random() < 0.7:
            expression = "x"
        else:
            const = random.choice(self.constants_pool)
            constants_used.append(const)
            expression = str(const)
        
        complexity_used = 1
        
        while complexity_used < target_complexity:
            if random.random() < 0.6:  # Binary operation
                operator = random.choice(self.binary_operators)
                operators_used.append(operator)
                
                if random.random() < 0.5:
                    operand = "x"
                else:
                    const = random.choice(self.constants_pool)
                    constants_used.append(const)
                    operand = str(const)
                
                expression = f"({expression} {operator} {operand})"
                complexity_used += 1
            
            else:  # Unary operation
                operator = random.choice(self.unary_operators)
                operators_used.append(operator)
                expression = f"{operator}({expression})"
                complexity_used += 1
        
        return expression, operators_used, constants_used
    
    def _remove_duplicates(self, candidates: List[ExpressionCandidate]) -> List[ExpressionCandidate]:
        """Remove duplicate expressions"""
        seen_expressions = set()
        unique_candidates = []
        
        for candidate in candidates:
            # Normalize expression for comparison
            normalized = self._normalize_expression(candidate.expression)
            if normalized not in seen_expressions:
                seen_expressions.add(normalized)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _normalize_expression(self, expression: str) -> str:
        """Normalize expression for duplicate detection"""
        # Simple normalization - could be enhanced
        normalized = expression.replace(" ", "")
        # Additional normalization rules could be added here
        return normalized

# Convenience function for backward compatibility
@performance_monitor
def generate_candidates(data: List[float] = None, strategy: str = "mixed", **kwargs) -> List[ExpressionCandidate]:
    """Generate expression candidates using the specified strategy"""
    if data is None:
        data = [1, 2, 3, 4, 5]  # Default data
    
    generator = CandidateGenerator(**kwargs)
    return generator.generate_candidates(data, strategy=strategy)
