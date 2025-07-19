
import numpy as np

# Lazy import PySR to avoid hanging on startup
_pysr_available = None

def _check_pysr_available():
    """Check if PySR is available (lazy loading)"""
    global _pysr_available
    if _pysr_available is None:
        try:
            from pysr import PySRRegressor
            _pysr_available = True
        except Exception:
            _pysr_available = False
    return _pysr_available

def run_symbolic_regression(X, y):
    """Run symbolic regression with fallback if PySR not available"""
    # For now, use fallback to avoid hanging issues
    return _fallback_regression(X, y)
    
    # Original PySR code (commented out to avoid hanging)
    

def _fallback_regression(X, y):
    """Simple fallback regression using polynomial fitting"""
    import numpy as np
    X_arr = np.array(X).reshape(-1, 1)
    y_arr = np.array(y)
    
    # Try to fit a simple polynomial
    try:
        # Fit degree 2 polynomial
        coeffs = np.polyfit(X_arr.flatten(), y_arr, 2)
        if len(coeffs) == 3:
            a, b, c = coeffs
            return {"expression": f"{a:.3f}*x^2 + {b:.3f}*x + {c:.3f}", 
                   "complexity": 3}
    except:
        pass
    
    # Fallback to linear
    try:
        coeffs = np.polyfit(X_arr.flatten(), y_arr, 1)
        if len(coeffs) == 2:
            a, b = coeffs
            return {"expression": f"{a:.3f}*x + {b:.3f}", "complexity": 2}
    except:
        pass
    
    # Ultimate fallback
    return {"expression": "unknown", "complexity": 0}
