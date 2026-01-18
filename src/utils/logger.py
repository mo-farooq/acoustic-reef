"""
Advanced logging system for Acoustic Reef
Provides structured logging with different levels and handlers
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import traceback
import json

class AcousticReefLogger:
    """Custom logger for Acoustic Reef application"""
    
    def __init__(self, name: str = "acoustic_reef", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"acoustic_reef_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details"""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['traceback'] = traceback.format_exc()
        self.logger.error(self._format_message(message, **kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get the path to the current log file"""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)
        return None

    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with optional context"""
        if kwargs:
            context = json.dumps(kwargs, default=str, indent=2)
            return f"{message}\nContext: {context}"
        return message

# Global logger instance
logger = AcousticReefLogger()

def get_logger(name: Optional[str] = None) -> AcousticReefLogger:
    """Get logger instance"""
    if name:
        return AcousticReefLogger(name)
    return logger

