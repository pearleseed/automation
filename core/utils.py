"""Shared utility functions."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for module."""
    return logging.getLogger(name)


class StructuredLogger:
    """Enhanced logger with file output and structured formatting.

    This class provides structured logging with clear visual separation of execution
    stages, steps, and results. Supports both console and file output with different
    formatting for better readability.
    """

    def __init__(
        self, name: str, log_file: Optional[str] = None, level: int = logging.INFO
    ):
        """Initialize structured logger with console and optional file output.

        Args:
            name: Logger name for identification.
            log_file: Path to log file for persistent logging (None = console only).
            level: Logging level (default: logging.INFO).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent duplicate logs

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatters
        console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")

        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        self.log_file: Optional[str] = None
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                ensure_directory(log_dir)

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.log_file = log_file

    def section_header(self, title: str, char: str = "=", width: int = 70):
        """Log a section header for better visual separation."""
        separator = char * width
        self.logger.info(separator)
        self.logger.info(f" {title}")
        self.logger.info(separator)

    def subsection_header(self, title: str, char: str = "-", width: int = 70):
        """Log a subsection header."""
        separator = char * width
        self.logger.info(separator)
        self.logger.info(f" {title}")
        self.logger.info(separator)

    def step(self, step_num: int, step_name: str, status: str = "START"):
        """Log a step with clear formatting."""
        self.logger.info(f"[STEP {step_num:2d}] {step_name} - {status}")

    def step_success(self, step_num: int, step_name: str, details: str = ""):
        """Log successful step completion."""
        msg = f"[STEP {step_num:2d}] ✓ {step_name} - SUCCESS"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)

    def step_failed(self, step_num: int, step_name: str, error: str = ""):
        """Log failed step."""
        msg = f"[STEP {step_num:2d}] ✗ {step_name} - FAILED"
        if error:
            msg += f" | {error}"
        self.logger.error(msg)

    def step_retry(
        self, step_num: int, step_name: str, attempt: int, max_attempts: int
    ):
        """Log step retry."""
        self.logger.warning(
            f"[STEP {step_num:2d}] {step_name} - RETRY {attempt}/{max_attempts}"
        )

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def stage_start(self, stage_idx: int, stage_name: str, stage_info: str = ""):
        """Log stage start with prominent formatting."""
        self.section_header(f"STAGE {stage_idx}: {stage_name}")
        if stage_info:
            self.info(f"Stage Info: {stage_info}")
        self.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("")

    def stage_end(self, stage_idx: int, success: bool, duration: float = 0):
        """Log stage completion."""
        status = "✓ COMPLETED SUCCESSFULLY" if success else "✗ FAILED"
        self.info("")
        if duration > 0:
            self.info(f"Duration: {duration:.2f} seconds")
        self.section_header(f"STAGE {stage_idx}: {status}")
        self.info("")

    def automation_start(self, automation_name: str, config: Optional[dict] = None):
        """Log automation start."""
        self.section_header(f"{automation_name} - AUTOMATION START", "=", 80)
        self.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.log_file:
            self.info(f"Log File: {self.log_file}")
        if config:
            self.info("Configuration:")
            for key, value in config.items():
                self.info(f"  - {key}: {value}")
        self.info("")

    def automation_end(
        self, automation_name: str, success: bool, summary: Optional[dict] = None
    ):
        """Log automation end."""
        self.info("")
        status = "✓ COMPLETED" if success else "✗ FAILED"
        self.section_header(f"{automation_name} - {status}", "=", 80)
        if summary:
            self.info("Summary:")
            for key, value in summary.items():
                self.info(f"  - {key}: {value}")
        self.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.section_header("", "=", 80)
