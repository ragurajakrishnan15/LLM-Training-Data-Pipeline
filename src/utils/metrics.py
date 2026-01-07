"""
Metrics tracking and reporting for the LLM Training Data Pipeline.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    
    stage_name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    error_count: int = 0
    bytes_processed: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        """Mark stage start time."""
        self.start_time = time.time()
    
    def end(self) -> None:
        """Mark stage end time."""
        self.end_time = time.time()
    
    @property
    def duration_seconds(self) -> float:
        """Get stage duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def throughput(self) -> float:
        """Get documents per second."""
        if self.duration_seconds == 0:
            return 0.0
        return self.output_count / self.duration_seconds
    
    @property
    def filter_rate(self) -> float:
        """Get percentage of documents filtered."""
        if self.input_count == 0:
            return 0.0
        return (self.filtered_count / self.input_count) * 100


class PipelineMetrics:
    """
    Comprehensive metrics tracking for the entire pipeline.
    
    Usage:
        metrics = PipelineMetrics()
        metrics.start_stage("ingestion")
        # ... do work ...
        metrics.end_stage("ingestion", input_count=1000, output_count=1000)
        metrics.print_report()
    """
    
    def __init__(self):
        self.stages: Dict[str, StageMetrics] = {}
        self.pipeline_start_time: Optional[float] = None
        self.pipeline_end_time: Optional[float] = None
        self.filter_reasons: Dict[str, int] = defaultdict(int)
        self.console = Console()
    
    def start_pipeline(self) -> None:
        """Mark pipeline start."""
        self.pipeline_start_time = time.time()
    
    def end_pipeline(self) -> None:
        """Mark pipeline end."""
        self.pipeline_end_time = time.time()
    
    def start_stage(self, stage_name: str) -> StageMetrics:
        """
        Start tracking a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            StageMetrics object for the stage
        """
        stage = StageMetrics(stage_name=stage_name)
        stage.start()
        self.stages[stage_name] = stage
        return stage
    
    def end_stage(
        self,
        stage_name: str,
        input_count: int = 0,
        output_count: int = 0,
        filtered_count: int = 0,
        error_count: int = 0,
        bytes_processed: int = 0,
        **custom_metrics
    ) -> None:
        """
        End tracking a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            input_count: Number of input documents
            output_count: Number of output documents
            filtered_count: Number of filtered documents
            error_count: Number of errors
            bytes_processed: Total bytes processed
            **custom_metrics: Additional custom metrics
        """
        if stage_name not in self.stages:
            raise ValueError(f"Stage '{stage_name}' was not started")
        
        stage = self.stages[stage_name]
        stage.end()
        stage.input_count = input_count
        stage.output_count = output_count
        stage.filtered_count = filtered_count
        stage.error_count = error_count
        stage.bytes_processed = bytes_processed
        stage.custom_metrics = custom_metrics
    
    def add_filter_reason(self, reason: str, count: int = 1) -> None:
        """Track why documents were filtered."""
        self.filter_reasons[reason] += count
    
    def get_stage(self, stage_name: str) -> Optional[StageMetrics]:
        """Get metrics for a specific stage."""
        return self.stages.get(stage_name)
    
    @property
    def total_duration(self) -> float:
        """Get total pipeline duration."""
        if self.pipeline_start_time is None or self.pipeline_end_time is None:
            return sum(s.duration_seconds for s in self.stages.values())
        return self.pipeline_end_time - self.pipeline_start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "pipeline_duration_seconds": self.total_duration,
            "stages": {
                name: asdict(stage) for name, stage in self.stages.items()
            },
            "filter_reasons": dict(self.filter_reasons),
            "timestamp": datetime.now().isoformat()
        }
    
    def save(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_report(self) -> None:
        """Print a formatted metrics report."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold blue]PIPELINE METRICS REPORT[/bold blue]",
            border_style="blue"
        ))
        
        # Summary table
        summary_table = Table(title="Stage Summary", show_header=True)
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Input", justify="right")
        summary_table.add_column("Output", justify="right")
        summary_table.add_column("Filtered", justify="right")
        summary_table.add_column("Filter %", justify="right")
        summary_table.add_column("Duration", justify="right")
        summary_table.add_column("Throughput", justify="right")
        
        for name, stage in self.stages.items():
            summary_table.add_row(
                name.upper(),
                f"{stage.input_count:,}",
                f"{stage.output_count:,}",
                f"{stage.filtered_count:,}",
                f"{stage.filter_rate:.1f}%",
                f"{stage.duration_seconds:.1f}s",
                f"{stage.throughput:.1f}/s"
            )
        
        self.console.print(summary_table)
        
        # Filter reasons
        if self.filter_reasons:
            self.console.print()
            filter_table = Table(title="Filter Reasons", show_header=True)
            filter_table.add_column("Reason", style="yellow")
            filter_table.add_column("Count", justify="right")
            
            for reason, count in sorted(
                self.filter_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                filter_table.add_row(reason, f"{count:,}")
            
            self.console.print(filter_table)
        
        # Total
        self.console.print()
        self.console.print(f"[bold]Total Pipeline Duration:[/bold] {self.total_duration:.1f}s")
        self.console.print()


# Global metrics instance
_metrics: Optional[PipelineMetrics] = None


def get_metrics() -> PipelineMetrics:
    """Get or create the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PipelineMetrics()
    return _metrics


def reset_metrics() -> None:
    """Reset the global metrics instance."""
    global _metrics
    _metrics = PipelineMetrics()
