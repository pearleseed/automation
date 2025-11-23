"""
Data Module - CSV and JSON data read/write management
"""

import csv
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import ensure_directory, get_logger

logger = get_logger(__name__)


# ==================== DATA LOADING ====================


def _validate_file(file_path: str) -> None:
    """Validate file exists, raise FileNotFoundError if not."""
    if not os.path.exists(file_path):
        logger.error(f" File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")


def load_csv(file_path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Read data from CSV file."""
    _validate_file(file_path)
    try:
        with open(file_path, encoding=encoding) as f:
            data = list(csv.DictReader(f))
        logger.info(f" Loaded {len(data)} rows from CSV: {file_path}")
        return data
    except Exception as e:
        logger.error(f" Error loading CSV: {e}")
        return []


def load_json(file_path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Read data from JSON file, normalize to list."""
    _validate_file(file_path)
    try:
        with open(file_path, encoding=encoding) as f:
            data = json.load(f)

        result = (
            data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        )
        if not isinstance(data, (list, dict)):
            logger.warning(f" Unexpected JSON type: {type(data)}, returning empty list")

        logger.info(f" Loaded {len(result)} items from JSON: {file_path}")
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f" Error loading JSON: {e}")
        return []


def load_data(file_path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Auto-detect and load data from CSV or JSON file.

    Args:
        file_path: Path to data file (.csv or .json).
        encoding: File encoding (default: 'utf-8').

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing loaded data.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is not supported.
    """
    _validate_file(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {".csv": load_csv, ".json": load_json}

    if ext not in loaders:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")
    return loaders[ext](file_path, encoding)


# ==================== DATA WRITING ====================


def write_csv(
    file_path: str, data: List[Dict[str, Any]], encoding: str = "utf-8", mode: str = "w"
) -> bool:
    """
    Write list of dictionaries to CSV file.

    Note: For incremental writes, always use mode='w' to rewrite entire file
    with all accumulated results. This ensures data consistency.
    """
    if not data:
        logger.warning(" No data to write")
        return False

    try:
        if directory := os.path.dirname(file_path):
            ensure_directory(directory)

        # Always write in 'w' mode to ensure complete file rewrite
        # This is safe because ResultWriter maintains all results in memory
        with open(file_path, "w", newline="", encoding=encoding) as f:
            # Collect all unique keys from all dictionaries
            all_keys = set().union(*(d.keys() for d in data))

            # Prioritize common fields for better readability
            common_fields = ["test_case_id", "timestamp", "result", "error_message"]
            ordered_fields = [f for f in common_fields if f in all_keys]
            remaining_fields = sorted([k for k in all_keys if k not in ordered_fields])
            fieldnames = ordered_fields + remaining_fields

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.debug(f" Wrote {len(data)} rows to CSV: {file_path}")
        return True
    except Exception as e:
        logger.error(f" Error writing CSV: {e}")
        return False


def write_json(
    file_path: str, data: List[Dict[str, Any]], encoding: str = "utf-8"
) -> bool:
    """Write list of dictionaries to JSON file."""
    if not data:
        logger.warning(" No data to write")
        return False

    try:
        if directory := os.path.dirname(file_path):
            ensure_directory(directory)

        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f" Wrote {len(data)} items to JSON: {file_path}")
        return True
    except Exception as e:
        logger.error(f" Error writing JSON: {e}")
        return False


def write_html(
    file_path: str, data: List[Dict[str, Any]], encoding: str = "utf-8"
) -> bool:
    """Write list of dictionaries to HTML report."""
    if not data:
        logger.warning(" No data to write")
        return False

    try:
        from core.utils import generate_html_report_content

        if directory := os.path.dirname(file_path):
            ensure_directory(directory)

        html_content = generate_html_report_content(data)
        with open(file_path, "w", encoding=encoding) as f:
            f.write(html_content)

        logger.debug(f" Wrote {len(data)} items to HTML: {file_path}")
        return True
    except Exception as e:
        logger.error(f" Error writing HTML: {e}")
        return False


# ==================== RESULT WRITER CLASS ====================


class ResultWriter:
    """Utility class for writing test/automation results to multiple formats with buffering.

    This class provides incremental result writing with automatic resume capability,
    allowing interrupted automation sessions to continue from where they left off.
    Supports CSV, JSON, and HTML formats.
    """

    RESULT_OK, RESULT_NG, RESULT_SKIP, RESULT_ERROR = "OK", "NG", "SKIP", "ERROR"

    def __init__(
        self,
        output_path: str,
        formats: List[str] = None,
        auto_write: bool = True,
        resume: bool = True,
    ):
        """Initialize ResultWriter with output path and formats.

        Args:
            output_path: Base path for output files (extension will be adjusted for formats).
            formats: List of formats to write (['csv', 'json', 'html']). Default: ['csv'].
            auto_write: Automatically write after each add_result (default: True).
            resume: Load existing results if file exists (default: True).
        """
        self.base_path = os.path.splitext(output_path)[0]
        self.formats = [f.lower() for f in (formats or ["csv"])]
        self.auto_write = auto_write
        self.results: List[Dict[str, Any]] = []
        self.completed_test_ids: set = set()

        # Ensure directory exists
        if directory := os.path.dirname(self.base_path):
            ensure_directory(directory)

        # Load existing results for resume support (only from CSV as it's the primary source)
        if resume:
            self._load_existing_results()

        logger.info(
            f"ResultWriter initialized: {self.base_path} "
            f"(formats={self.formats}, auto_write={auto_write}, resume={resume})"
        )

    def _load_existing_results(self) -> None:
        """Load existing results from CSV file for resume support."""
        csv_path = f"{self.base_path}.csv"
        if not os.path.exists(csv_path):
            logger.info("No existing results file found, starting fresh")
            return

        try:
            existing_data = load_csv(csv_path)
            if not existing_data:
                return

            self.results = existing_data

            # Track completed test IDs
            for row in existing_data:
                test_id = row.get("test_case_id")
                if test_id:
                    self.completed_test_ids.add(str(test_id))

            # Log detailed resume information
            completed_ids = sorted(
                [int(id) for id in self.completed_test_ids if id.isdigit()]
            )
            logger.info(f"✓ Resume: Loaded {len(self.results)} existing results")
            logger.info(
                f"✓ Resume: {len(self.completed_test_ids)} completed test cases"
            )

        except Exception as e:
            logger.warning(f"Could not load existing results: {e}, starting fresh")

    def is_completed(self, test_case: Dict[str, Any]) -> bool:
        """Check if a test case has already been completed."""
        test_id = test_case.get("test_case_id")
        return str(test_id) in self.completed_test_ids if test_id else False

    def add_result(
        self,
        test_case: Dict[str, Any],
        result: str,
        error_message: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a test case result with timestamp and optional error/extra fields.

        Args:
            test_case: Dictionary containing test case data.
            result: Result status (RESULT_OK, RESULT_NG, RESULT_SKIP, or RESULT_ERROR).
            error_message: Optional error message if test failed.
            extra_fields: Optional additional fields to include in the result.
        """
        row_data = {
            **test_case,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result,
        }
        if error_message:
            row_data["error_message"] = error_message
        if extra_fields:
            row_data.update(extra_fields)

        # Track completed test ID
        test_id = test_case.get("test_case_id")
        if test_id:
            self.completed_test_ids.add(str(test_id))

        self.results.append(row_data)

        # Incremental save when auto_write is enabled
        if self.auto_write:
            self.write()

    def write(self, clear_after_write: bool = False) -> bool:
        """Write all results to configured formats."""
        if not self.results:
            return False

        success = True
        
        # Write CSV
        if "csv" in self.formats:
            if not write_csv(f"{self.base_path}.csv", self.results):
                success = False

        # Write JSON
        if "json" in self.formats:
            if not write_json(f"{self.base_path}.json", self.results):
                success = False

        # Write HTML
        if "html" in self.formats:
            if not write_html(f"{self.base_path}.html", self.results):
                success = False

        if success and clear_after_write:
            self.clear()
        
        return success

    def flush(self) -> bool:
        """Force write all results immediately."""
        return self.write()

    def clear(self) -> None:
        """Clear results buffer."""
        self.results.clear()

    def get_summary(self) -> Dict[str, int]:
        """Get results summary with count of each result type."""
        return dict(
            Counter(row.get("result", self.RESULT_ERROR) for row in self.results)
        )

    def print_summary(self) -> None:
        """Print results summary to logger."""
        summary = self.get_summary()
        total = sum(summary.values())

        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        if total == 0:
            logger.info("  No results recorded")
        else:
            for result_type, count in sorted(summary.items()):
                if count > 0:
                    logger.info(
                        f"  {result_type:<10}: {count:>4} ({count/total*100:>5.1f}%)"
                    )

        logger.info("-" * 60)
        logger.info(f"  {'TOTAL':<10}: {total:>4}")
        logger.info("=" * 60)

    @property
    def count(self) -> int:
        """Current number of results."""
        return len(self.results)
