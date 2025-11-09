"""
Logging utilities for training and inference.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path


class Logger:
    """Simple file-based logger."""
    
    def __init__(self, log_dir: str, name: str = "train"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"
        
        self.info(f"Logger initialized: {self.log_file}")
    
    def _write(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        print(log_line.rstrip())
    
    def info(self, message: str):
        self._write("INFO", message)
    
    def warning(self, message: str):
        self._write("WARNING", message)
    
    def error(self, message: str):
        self._write("ERROR", message)
    
    def debug(self, message: str):
        self._write("DEBUG", message)


class TensorBoardLogger:
    """TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: str, enabled: bool = True):
        self.log_dir = log_dir
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
                print(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                print("Warning: tensorboard not available, logging disabled")
                self.enabled = False
    
    def add_scalar(self, tag: str, value: float, step: int):
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def add_image(self, tag: str, img_tensor, step: int):
        if self.enabled and self.writer:
            self.writer.add_image(tag, img_tensor, step)
    
    def add_images(self, tag: str, img_tensor, step: int):
        if self.enabled and self.writer:
            self.writer.add_images(tag, img_tensor, step)
    
    def add_histogram(self, tag: str, values, step: int):
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def flush(self):
        if self.enabled and self.writer:
            self.writer.flush()
    
    def close(self):
        if self.enabled and self.writer:
            self.writer.close()


class MetricTracker:
    """Track and compute running averages of metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(value))
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        
        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def get_all_averages(self, last_n: Optional[int] = None) -> Dict[str, float]:
        return {key: self.get_average(key, last_n) for key in self.metrics.keys()}
    
    def reset(self, key: Optional[str] = None):
        if key is None:
            self.metrics = {}
        elif key in self.metrics:
            self.metrics[key] = []
    
    def __repr__(self) -> str:
        avgs = self.get_all_averages()
        return " | ".join([f"{k}: {v:.4f}" for k, v in avgs.items()])
