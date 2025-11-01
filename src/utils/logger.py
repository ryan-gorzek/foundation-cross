"""
Logging utilities for the pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class PipelineLogger:
    """Custom logger for the pipeline."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def section(self, title: str, char: str = "=", width: int = 89):
        """Log a section header."""
        self.logger.info("\n" + char * width)
        self.logger.info(title)
        self.logger.info(char * width)
    
    def subsection(self, title: str, char: str = "-", width: int = 89):
        """Log a subsection header."""
        self.logger.info(char * width)
        self.logger.info(title)
        self.logger.info(char * width)


def setup_logger(name: str, save_dir: Path, level: int = logging.INFO) -> PipelineLogger:
    """Setup logger with file and console output."""
    log_file = save_dir / "run.log"
    return PipelineLogger(name, log_file, level)