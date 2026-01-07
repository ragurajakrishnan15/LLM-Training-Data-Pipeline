"""
Logging utilities for the LLM Training Data Pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


class PipelineLogger:
    """Custom logger for the pipeline with rich formatting."""
    
    _loggers: dict = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (usually __name__)
            level: Logging level
            log_file: Optional file path for logging
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    return PipelineLogger.get_logger(name, log_file=log_file)


class ProgressTracker:
    """Track and display progress for pipeline stages."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()
        self.console = Console()
    
    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n
        
    def get_stats(self) -> dict:
        """Get current progress statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        return {
            "current": self.current,
            "total": self.total,
            "percentage": (self.current / self.total) * 100 if self.total > 0 else 0,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "eta_seconds": eta
        }
    
    def log_progress(self, logger: logging.Logger) -> None:
        """Log current progress."""
        stats = self.get_stats()
        logger.info(
            f"{self.description}: {stats['current']:,}/{stats['total']:,} "
            f"({stats['percentage']:.1f}%) - "
            f"{stats['rate_per_second']:.1f}/s - "
            f"ETA: {stats['eta_seconds']:.0f}s"
        )
