"""
Comprehensive logging system for Artifact ATP
Provides structured logging with performance metrics and error tracking
"""
import logging
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import functools

class StructuredLogger:
    """Structured logger with JSON output and performance tracking"""
    
    def __init__(self, name: str = "artifact_reason", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "artifact_reason.log")
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional metadata"""
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional metadata"""
        self._log(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional metadata"""
        self._log(logging.WARNING, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional metadata"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with metadata"""
        extra = {
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": kwargs
        }
        self.logger.log(level, message, extra=extra)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": getattr(record, 'timestamp', datetime.utcnow().isoformat()),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add metadata if present
        if hasattr(record, 'metadata'):
            log_entry["metadata"] = record.metadata
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = f"{func.__module__}.{func.__name__}"
        
        logger.debug(f"Starting {function_name}", 
                    function=function_name, args_count=len(args), kwargs_count=len(kwargs))
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(f"Completed {function_name}", 
                       function=function_name, duration=duration, success=True)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {function_name}", 
                        function=function_name, duration=duration, 
                        error=str(e), success=False)
            raise
    
    return wrapper

def async_performance_monitor(func):
    """Decorator to monitor async function performance"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = f"{func.__module__}.{func.__name__}"
        
        logger.debug(f"Starting async {function_name}", 
                    function=function_name, args_count=len(args), kwargs_count=len(kwargs))
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(f"Completed async {function_name}", 
                       function=function_name, duration=duration, success=True)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed async {function_name}", 
                        function=function_name, duration=duration, 
                        error=str(e), success=False)
            raise
    
    return wrapper


# Global logger instance
logger = StructuredLogger()

def log(msg: str, level: str = "INFO", **kwargs):
    """Legacy function for backward compatibility"""
    getattr(logger, level.lower())(msg, **kwargs)
