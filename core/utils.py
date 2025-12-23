"""
Core Utilities Module

This module provides shared utility functions for file operations, logging,
input validation, and report generation used across the automation framework.
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Initialize module logger
logger = logging.getLogger(__name__)


# ==================== PATH VALIDATION ====================

# Pattern to detect path traversal attempts
_PATH_TRAVERSAL_PATTERN = re.compile(r"(^|[\\/])\.\.($|[\\/])")

# Dangerous path components that should be blocked
_DANGEROUS_COMPONENTS = frozenset(
    {
        "..",
        "...",
        "~",
        "$",
        "%",
        "CON",
        "PRN",
        "AUX",
        "NUL",  # Windows reserved names
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def validate_file_path(
    path: str,
    base_dir: Optional[str] = None,
    allow_absolute: bool = False,
    must_exist: bool = False,
) -> bool:
    """Validate a file path for security and correctness.

    Args:
        path: The file path to validate.
        base_dir: If provided, ensure path is within this directory.
        allow_absolute: Whether to allow absolute paths.
        must_exist: Whether the path must exist.

    Returns:
        bool: True if path is valid and safe, False otherwise.
    """
    if not path or not isinstance(path, str):
        return False

    # Check for null bytes (security)
    if "\x00" in path:
        logger.warning(f"Path contains null byte: {path!r}")
        return False

    # Check for path traversal patterns
    if _PATH_TRAVERSAL_PATTERN.search(path):
        logger.warning(f"Path traversal detected: {path}")
        return False

    # Check for dangerous path components
    path_parts = Path(path).parts
    for part in path_parts:
        part_upper = part.upper().split(".")[0]  # Handle "CON.txt" etc.
        if part_upper in _DANGEROUS_COMPONENTS or part.startswith("~"):
            logger.warning(f"Dangerous path component detected: {part}")
            return False

    # Check absolute path restriction
    if not allow_absolute and os.path.isabs(path):
        logger.warning(f"Absolute path not allowed: {path}")
        return False

    # Validate within base directory using resolved paths
    if base_dir:
        try:
            base_resolved = Path(base_dir).resolve(strict=False)
            path_resolved = Path(base_dir, path).resolve(strict=False)

            # Use os.path.commonpath for reliable containment check
            try:
                common = os.path.commonpath([str(base_resolved), str(path_resolved)])
                if common != str(base_resolved):
                    logger.warning(f"Path escapes base directory: {path}")
                    return False
            except ValueError:
                # Different drives on Windows
                logger.warning(f"Path on different drive than base: {path}")
                return False

        except (ValueError, OSError) as e:
            logger.warning(f"Path resolution failed: {path}, error: {e}")
            return False

    # Check existence if required
    if must_exist and not os.path.exists(path):
        return False

    return True


def validate_directory_path(
    path: str,
    allowed_bases: Optional[List[str]] = None,
    create_if_missing: bool = False,
) -> Optional[str]:
    """Validate and optionally create a directory path.

    Args:
        path: The directory path to validate.
        allowed_bases: List of allowed base directories. Path must be under one of these.
        create_if_missing: Create the directory if it doesn't exist.

    Returns:
        Optional[str]: Resolved absolute path if valid, None otherwise.
    """
    if not path or not isinstance(path, str):
        return None

    try:
        resolved = Path(path).resolve(strict=False)
        resolved_str = str(resolved)

        # Check against allowed bases if provided
        if allowed_bases:
            is_allowed = False
            for base in allowed_bases:
                try:
                    base_resolved = Path(base).resolve(strict=False)
                    common = os.path.commonpath([str(base_resolved), resolved_str])
                    if common == str(base_resolved):
                        is_allowed = True
                        break
                except (ValueError, OSError):
                    continue

            if not is_allowed:
                logger.warning(f"Directory not under allowed bases: {path}")
                return None

        # Create if requested and doesn't exist
        if create_if_missing and not resolved.exists():
            resolved.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {resolved_str}")

        return resolved_str

    except (ValueError, OSError) as e:
        logger.warning(f"Directory validation failed: {path}, error: {e}")
        return None


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename by removing/replacing unsafe characters.

    Args:
        filename: The filename to sanitize.
        max_length: Maximum allowed length for the filename.

    Returns:
        str: Sanitized filename safe for filesystem use.
    """
    if not filename:
        return "unnamed"

    # Remove path separators, null bytes, and other unsafe characters
    unsafe_chars = '<>:"/\\|?*\x00\n\r\t'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing dots, spaces, and underscores
    filename = filename.strip(". _")

    # Check for Windows reserved names
    name_upper = filename.upper().split(".")[0]
    if name_upper in _DANGEROUS_COMPONENTS:
        filename = f"_{filename}"

    # Collapse multiple underscores
    while "__" in filename:
        filename = filename.replace("__", "_")

    # Truncate if too long (preserve extension)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_len = max(1, max_length - len(ext))
        filename = name[:max_name_len] + ext

    return filename or "unnamed"


def safe_join_path(base_dir: str, *parts: str) -> Optional[str]:
    """Safely join path components, preventing path traversal.

    Args:
        base_dir: The base directory (must be absolute or will be resolved).
        *parts: Path components to join.

    Returns:
        Optional[str]: Safe joined path, or None if traversal detected.
    """
    if not base_dir:
        return None

    try:
        base_resolved = Path(base_dir).resolve(strict=False)

        # Sanitize each part
        safe_parts = [sanitize_filename(str(p)) for p in parts if p]
        if not safe_parts:
            return str(base_resolved)

        # Join and resolve
        joined = base_resolved.joinpath(*safe_parts)
        joined_resolved = joined.resolve(strict=False)

        # Verify still under base
        try:
            common = os.path.commonpath([str(base_resolved), str(joined_resolved)])
            if common != str(base_resolved):
                logger.warning(f"Path traversal attempt: {parts}")
                return None
        except ValueError:
            return None

        return str(joined_resolved)

    except (ValueError, OSError) as e:
        logger.warning(f"Safe path join failed: {e}")
        return None


# ==================== FILE UTILS ====================


def ensure_directory(path: str) -> bool:
    """Create directory if it doesn't exist.

    Args:
        path: The directory path to create.

    Returns:
        bool: True if directory exists or was created, False on error.
    """
    if not path:
        return False

    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


# ==================== LOGGING UTILS ====================


def get_logger(name: str) -> logging.Logger:
    """Get a standard logger instance for a module.

    Args:
        name: The name of the logger (usually __name__).

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)


class StructuredLogger:
    """Logger with file output and structured formatting.

    This class provides structured logging with clear visual separation of execution
    stages, steps, and results. It supports both console and file output with
    different formatting for better readability.
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

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        # Console: Simpler format for readability
        console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")

        # File: Detailed format with timestamps and module names
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

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8-sig")
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.log_file = log_file

    def section_header(self, title: str, char: str = "=", width: int = 70) -> None:
        """Log a major section header for visual separation.

        Args:
            title: The title text to display.
            char: The character to use for the separator line (default: "=").
            width: The width of the separator line (default: 70).
        """
        separator = char * width
        self.logger.info(separator)
        self.logger.info(f" {title}")
        self.logger.info(separator)

    def subsection_header(self, title: str, char: str = "-", width: int = 70) -> None:
        """Log a subsection header.

        Args:
            title: The title text to display.
            char: The character to use for the separator line (default: "-").
            width: The width of the separator line (default: 70).
        """
        separator = char * width
        self.logger.info(separator)
        self.logger.info(f" {title}")
        self.logger.info(separator)

    def step(self, step_num: int, step_name: str, status: str = "START") -> None:
        """Log a step start with clear formatting.

        Args:
            step_num: The step number.
            step_name: The name/description of the step.
            status: The status label (default: "START").
        """
        self.logger.info(f"[STEP {step_num:2d}] {step_name} - {status}")

    def step_success(self, step_num: int, step_name: str, details: str = "") -> None:
        """Log successful step completion.

        Args:
            step_num: The step number.
            step_name: The name/description of the step.
            details: Optional additional details to log.
        """
        msg = f"[STEP {step_num:2d}] ✓ {step_name} - SUCCESS"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)

    def step_failed(self, step_num: int, step_name: str, error: str = "") -> None:
        """Log failed step.

        Args:
            step_num: The step number.
            step_name: The name/description of the step.
            error: Optional error message or details.
        """
        msg = f"[STEP {step_num:2d}] ✗ {step_name} - FAILED"
        if error:
            msg += f" | {error}"
        self.logger.error(msg)

    def step_retry(
        self, step_num: int, step_name: str, attempt: int, max_attempts: int
    ) -> None:
        """Log step retry attempt.

        Args:
            step_num: The step number.
            step_name: The name/description of the step.
            attempt: The current attempt number.
            max_attempts: The maximum number of attempts allowed.
        """
        self.logger.warning(
            f"[STEP {step_num:2d}] {step_name} - RETRY {attempt}/{max_attempts}"
        )

    def info(self, msg: str) -> None:
        """Log an info message.

        Args:
            msg: The message to log.
        """
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message.

        Args:
            msg: The message to log.
        """
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message.

        Args:
            msg: The message to log.
        """
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        """Log a debug message.

        Args:
            msg: The message to log.
        """
        self.logger.debug(msg)

    def stage_start(
        self, stage_idx: int, stage_name: str, stage_info: str = ""
    ) -> None:
        """Log the start of a major execution stage.

        Args:
            stage_idx: The index/number of the stage.
            stage_name: The name of the stage.
            stage_info: Optional additional info about the stage.
        """
        self.section_header(f"STAGE {stage_idx}: {stage_name}")
        if stage_info:
            self.info(f"Stage Info: {stage_info}")
        self.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("")

    def stage_end(self, stage_idx: int, success: bool, duration: float = 0) -> None:
        """Log the completion of a major execution stage.

        Args:
            stage_idx: The index/number of the stage.
            success: True if the stage completed successfully, False otherwise.
            duration: The duration of the stage in seconds.
        """
        status = "✓ COMPLETED SUCCESSFULLY" if success else "✗ FAILED"
        self.info("")
        if duration > 0:
            self.info(f"Duration: {duration:.2f} seconds")
        self.section_header(f"STAGE {stage_idx}: {status}")
        self.info("")

    def automation_start(
        self, automation_name: str, config: Optional[dict] = None
    ) -> None:
        """Log the start of an entire automation process.

        Args:
            automation_name: The name of the automation.
            config: Optional configuration dictionary to log.
        """
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
    ) -> None:
        """Log the end of an entire automation process.

        Args:
            automation_name: The name of the automation.
            success: True if the automation completed successfully.
            summary: Optional summary dictionary to log.
        """
        self.info("")
        status = "✓ COMPLETED" if success else "✗ FAILED"
        self.section_header(f"{automation_name} - {status}", "=", 80)
        if summary:
            self.info("Summary:")
            for key, value in summary.items():
                self.info(f"  - {key}: {value}")
        self.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.section_header("", "=", 80)
