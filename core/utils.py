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


def generate_html_report_content(data: list) -> str:
    """Generate HTML report content from result data.

    Args:
        data: List of result dictionaries.

    Returns:
        String containing complete HTML report.
    """
    if not data:
        return "<html><body><h1>No Data</h1></body></html>"

    # Calculate summary
    total = len(data)
    ok_count = sum(1 for row in data if row.get("result") == "OK")
    ng_count = sum(1 for row in data if row.get("result") == "NG")
    skip_count = sum(1 for row in data if row.get("result") == "SKIP")
    error_count = sum(1 for row in data if row.get("result") == "ERROR")
    success_rate = (ok_count / total * 100) if total > 0 else 0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get headers from first row, prioritizing common fields
    common_headers = ["test_case_id", "timestamp", "result", "error_message"]
    all_keys = set().union(*(d.keys() for d in data))
    other_headers = sorted([k for k in all_keys if k not in common_headers])
    headers = [h for h in common_headers if h in all_keys] + other_headers

    # Generate rows
    rows_html = ""
    for row in data:
        result_class = row.get("result", "").lower()
        rows_html += f"<tr class='{result_class}'>"
        for header in headers:
            val = row.get(header, "")
            rows_html += f"<td>{val}</td>"
        rows_html += "</tr>"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Automation Report - {timestamp}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 6px; }}
        .summary-item {{ flex: 1; text-align: center; }}
        .summary-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .summary-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background-color: #f8f9fa; font-weight: 600; color: #2c3e50; position: sticky; top: 0; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .ok {{ border-left: 4px solid #2ecc71; }}
        .ng {{ border-left: 4px solid #e74c3c; background-color: #fff5f5; }}
        .skip {{ border-left: 4px solid #f1c40f; }}
        .error {{ border-left: 4px solid #e74c3c; background-color: #ffe6e6; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .badge-ok {{ background: #d4edda; color: #155724; }}
        .badge-ng {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Automation Report</h1>
        <div class="summary">
            <div class="summary-item">
                <div class="summary-value">{total}</div>
                <div class="summary-label">Total Tests</div>
            </div>
            <div class="summary-item">
                <div class="summary-value" style="color: #2ecc71">{ok_count}</div>
                <div class="summary-label">Passed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value" style="color: #e74c3c">{ng_count}</div>
                <div class="summary-label">Failed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{success_rate:.1f}%</div>
                <div class="summary-label">Success Rate</div>
            </div>
        </div>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        {''.join(f'<th>{h}</th>' for h in headers)}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
