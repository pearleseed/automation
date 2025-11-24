"""
Core Utilities Module

This module provides shared utility functions for file operations, logging,
and report generation used across the automation framework.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Initialize module logger
logger = logging.getLogger(__name__)


# ==================== FILE UTILS ====================


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist.

    Args:
        path: The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


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


# ==================== REPORTING UTILS ====================


def generate_html_report_content(data: List[Dict[str, Any]]) -> str:
    """Generate a styled HTML report from automation result data.

    Args:
        data: List of result dictionaries containing test execution data.

    Returns:
        str: Complete HTML document string with enhanced visualizations.
    """
    if not data:
        return "<html><body><h1>No Data</h1></body></html>"

    # Calculate summary statistics
    total = len(data)
    ok_count = sum(1 for row in data if row.get("result") == "OK")
    ng_count = sum(1 for row in data if row.get("result") == "NG")
    skip_count = sum(1 for row in data if row.get("result") == "SKIP")
    error_count = sum(1 for row in data if row.get("result") == "ERROR")
    success_rate = (ok_count / total * 100) if total > 0 else 0
    fail_rate = (ng_count / total * 100) if total > 0 else 0

    # Calculate percentages for progress bars
    ok_percent = (ok_count / total * 100) if total > 0 else 0
    ng_percent = (ng_count / total * 100) if total > 0 else 0
    skip_percent = (skip_count / total * 100) if total > 0 else 0
    error_percent = (error_count / total * 100) if total > 0 else 0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine table headers
    common_headers = ["test_case_id", "timestamp", "result", "error_message"]
    all_keys = set().union(*(d.keys() for d in data))
    other_headers = sorted([k for k in all_keys if k not in common_headers])
    headers = [h for h in common_headers if h in all_keys] + other_headers

    # Generate table rows with enhanced result badges
    rows_html = ""
    for idx, row in enumerate(data, 1):
        result = row.get("result", "").upper()
        result_class = result.lower()

        # Create badge for result column
        badge_class = {
            "OK": "badge-ok",
            "NG": "badge-ng",
            "SKIP": "badge-skip",
            "ERROR": "badge-error",
        }.get(result, "badge-default")

        rows_html += f"<tr class='{result_class}'>"
        rows_html += f"<td class='row-number'>{idx}</td>"

        for header in headers:
            val = row.get(header, "")
            if header == "result":
                rows_html += f"<td><span class='badge {badge_class}'>{val}</span></td>"
            else:
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
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        .container {{ 
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header {{
            background: #fff;
            border-bottom: 1px solid #e0e0e0;
            padding: 20px 30px;
        }}
        .header h1 {{ 
            font-size: 20px;
            font-weight: 600;
            color: #333;
            margin-bottom: 4px;
        }}
        .header .timestamp {{
            font-size: 12px;
            color: #666;
        }}
        .content {{ padding: 30px; }}
        
        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }}
        .summary-card {{
            background: #fafafa;
            border-radius: 4px;
            padding: 16px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        
        .summary-value {{
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 4px;
            line-height: 1;
            color: #333;
        }}
        
        .summary-label {{
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
        }}
        .summary-percent {{
            font-size: 11px;
            margin-top: 2px;
            color: #999;
        }}
        
        /* Progress Bar Section */
        .progress-section {{
            background: #fafafa;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid #e0e0e0;
        }}
        .progress-section h2 {{
            font-size: 13px;
            margin-bottom: 12px;
            color: #333;
            font-weight: 600;
        }}
        .progress-bar-container {{
            background: #e0e0e0;
            border-radius: 4px;
            height: 24px;
            overflow: hidden;
            display: flex;
        }}
        .progress-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 500;
            font-size: 11px;
        }}
        .progress-ok {{ background: #4caf50; }}
        .progress-ng {{ background: #f44336; }}
        .progress-skip {{ background: #ff9800; }}
        .progress-error {{ background: #ff5722; }}
        
        .progress-legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 12px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #666;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        
        /* Table Section */
        .table-section {{
            margin-top: 24px;
        }}
        .table-section h2 {{
            font-size: 13px;
            margin-bottom: 12px;
            color: #333;
            font-weight: 600;
        }}
        .table-controls {{
            margin-bottom: 12px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .search-box {{
            flex: 1;
            min-width: 200px;
            padding: 6px 10px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-size: 12px;
        }}
        .filter-btn {{
            padding: 6px 12px;
            border: 1px solid #e0e0e0;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
        }}
        .filter-btn:hover {{
            background: #fafafa;
        }}
        .filter-btn.active {{
            background: #333;
            color: white;
            border-color: #333;
        }}
        
        .table-wrapper {{
            overflow-x: auto;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #f0f0f0;
        }}
        th {{
            background: #fafafa;
            font-weight: 600;
            color: #333;
            position: sticky;
            top: 0;
            z-index: 10;
            font-size: 11px;
        }}
        tbody tr:hover {{
            background-color: #fafafa;
        }}
        .row-number {{
            font-weight: 500;
            color: #999;
            width: 50px;
        }}
        
        /* Result badges */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        .badge-ok {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-ng {{ background: #ffebee; color: #c62828; }}
        .badge-skip {{ background: #fff3e0; color: #ef6c00; }}
        .badge-error {{ background: #fce4ec; color: #c2185b; }}
        .badge-default {{ background: #f5f5f5; color: #616161; }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 16px;
            color: #999;
            font-size: 11px;
            border-top: 1px solid #e0e0e0;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .header h1 {{ font-size: 18px; }}
            .summary-value {{ font-size: 24px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Automation Test Report</h1>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>
        
        <div class="content">
            <!-- Summary Cards -->
            <div class="summary-grid">
                <div class="summary-card total">
                    <div class="summary-value">{total}</div>
                    <div class="summary-label">Total Tests</div>
                </div>
                <div class="summary-card passed">
                    <div class="summary-value">{ok_count}</div>
                    <div class="summary-label">Passed</div>
                    <div class="summary-percent">{ok_percent:.1f}%</div>
                </div>
                <div class="summary-card failed">
                    <div class="summary-value">{ng_count}</div>
                    <div class="summary-label">Failed</div>
                    <div class="summary-percent">{ng_percent:.1f}%</div>
                </div>
                <div class="summary-card skipped">
                    <div class="summary-value">{skip_count}</div>
                    <div class="summary-label">Skipped</div>
                    <div class="summary-percent">{skip_percent:.1f}%</div>
                </div>
                <div class="summary-card error">
                    <div class="summary-value">{error_count}</div>
                    <div class="summary-label">Errors</div>
                    <div class="summary-percent">{error_percent:.1f}%</div>
                </div>
            </div>
            
            <!-- Progress Bar -->
            <div class="progress-section">
                <h2>Test Results Distribution</h2>
                <div class="progress-bar-container">
                    {f'<div class="progress-segment progress-ok" style="width: {ok_percent}%">{ok_count if ok_count > 0 else ""}</div>' if ok_count > 0 else ''}
                    {f'<div class="progress-segment progress-ng" style="width: {ng_percent}%">{ng_count if ng_count > 0 else ""}</div>' if ng_count > 0 else ''}
                    {f'<div class="progress-segment progress-skip" style="width: {skip_percent}%">{skip_count if skip_count > 0 else ""}</div>' if skip_count > 0 else ''}
                    {f'<div class="progress-segment progress-error" style="width: {error_percent}%">{error_count if error_count > 0 else ""}</div>' if error_count > 0 else ''}
                </div>
                <div class="progress-legend">
                    <div class="legend-item">
                        <div class="legend-color progress-ok"></div>
                        <span>Passed ({ok_percent:.1f}%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color progress-ng"></div>
                        <span>Failed ({ng_percent:.1f}%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color progress-skip"></div>
                        <span>Skipped ({skip_percent:.1f}%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color progress-error"></div>
                        <span>Errors ({error_percent:.1f}%)</span>
                    </div>
                </div>
            </div>
            
            <!-- Table Section -->
            <div class="table-section">
                <h2>Detailed Test Results</h2>
                <div class="table-controls">
                    <input type="text" class="search-box" placeholder="Search test cases..." onkeyup="filterTable()">
                    <button class="filter-btn active" onclick="filterByStatus('all')">All</button>
                    <button class="filter-btn" onclick="filterByStatus('ok')">Passed</button>
                    <button class="filter-btn" onclick="filterByStatus('ng')">Failed</button>
                    <button class="filter-btn" onclick="filterByStatus('skip')">Skipped</button>
                    <button class="filter-btn" onclick="filterByStatus('error')">Errors</button>
                </div>
                <div class="table-wrapper">
                    <table id="resultsTable">
                        <thead>
                            <tr>
                                <th>#</th>
                                {''.join(f'<th>{h.replace("_", " ").title()}</th>' for h in headers)}
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Generated by Automation Framework | {timestamp}
        </div>
    </div>
    
    <script>
        function filterTable() {{
            const input = document.querySelector('.search-box');
            const filter = input.value.toUpperCase();
            const table = document.getElementById('resultsTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {{
                const row = rows[i];
                const text = row.textContent || row.innerText;
                row.style.display = text.toUpperCase().indexOf(filter) > -1 ? '' : 'none';
            }}
        }}
        
        function filterByStatus(status) {{
            const table = document.getElementById('resultsTable');
            const rows = table.getElementsByTagName('tr');
            const buttons = document.querySelectorAll('.filter-btn');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            for (let i = 1; i < rows.length; i++) {{
                const row = rows[i];
                if (status === 'all') {{
                    row.style.display = '';
                }} else {{
                    row.style.display = row.classList.contains(status) ? '' : 'none';
                }}
            }}
        }}
    </script>
</body>
</html>
"""
