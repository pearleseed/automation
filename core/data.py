"""
Data Module - CSV and JSON data read/write management.

This module provides safe file I/O operations with atomic writes,
proper error handling, and input validation.
"""

import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import ensure_directory, get_logger, validate_file_path

logger = get_logger(__name__)


# ==================== DATA LOADING ====================


def _validate_file(file_path: str, check_exists: bool = True) -> None:
    """Validate file path for security and existence.

    Args:
        file_path: Path to validate.
        check_exists: Whether to check if file exists.

    Raises:
        FileNotFoundError: If file doesn't exist and check_exists is True.
        ValueError: If path is invalid or unsafe.
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    if not validate_file_path(file_path, allow_absolute=True):
        raise ValueError(f"Invalid or unsafe file path: {file_path}")

    if check_exists and not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")


def load_csv(
    file_path: str, encoding: str = "utf-8-sig", delimiter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Read data from CSV file with auto-detection of delimiter.

    Args:
        file_path: Path to CSV file.
        encoding: File encoding (default: utf-8-sig).
        delimiter: CSV delimiter. If None, uses platform default (Windows=',', macOS=';').

    Returns:
        List of dictionaries containing CSV data.
    """
    import sys

    _validate_file(file_path)
    try:
        with open(file_path, encoding=encoding) as f:
            # Use platform-specific delimiter if not specified
            # Windows uses ',' while macOS uses ';'
            if delimiter is None:
                if sys.platform == "win32":
                    delimiter = ","
                else:
                    # macOS/Linux - use semicolon
                    delimiter = ";"

            data = list(csv.DictReader(f, delimiter=delimiter))
        logger.info(
            f"✓ Loaded {len(data)} rows from CSV: {file_path} (delimiter='{delimiter}')"
        )
        return data
    except Exception as e:
        logger.error(f"✗ Error loading CSV: {e}")
        return []


def load_json(file_path: str, encoding: str = "utf-8-sig") -> List[Dict[str, Any]]:
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


def load_data(
    file_path: str, encoding: str = "utf-8-sig", delimiter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Auto-detect and load data from CSV or JSON file.

    Args:
        file_path: Path to data file (.csv or .json).
        encoding: File encoding (default: 'utf-8-sig').
        delimiter: CSV delimiter (only for CSV files). If None, auto-detects.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing loaded data.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is not supported.
    """
    _validate_file(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return load_csv(file_path, encoding, delimiter)
    elif ext == ".json":
        return load_json(file_path, encoding)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")


# ==================== HOPPING DATA ====================


def load_hopping_data(
    file_path: str, encoding: str = "utf-8-sig"
) -> List[Dict[str, Any]]:
    """Load hopping dataset from CSV or JSON file.

    Pool Hopping dataset structure:
    - Course: Course identifier (1-6, EX1-EX6)
    - Spot: Position on the course (1-17+, 岸=shore/end)
    - Item: Item or special effect at this spot
    - Number: Quantity or value
    - Result: Verification result (OK/NG/Draw Unchecked)

    Args:
        file_path: Path to CSV or JSON file.
        encoding: File encoding (default: utf-8-sig).

    Returns:
        List of spot dictionaries with normalized keys.
    """
    raw_data = load_data(file_path, encoding)
    if not raw_data:
        return []

    # Normalize and process hopping data
    processed_data = []
    current_course = None

    for row in raw_data:
        # Handle both CSV column names and normalized names
        course = (row.get("Course", "") or row.get("course", "") or "").strip()
        spot = (row.get("Spot", "") or row.get("spot", "") or "").strip()
        # Handle typo in CSV header "Item/Spceial Effect "
        item = (
            row.get("Item/Spceial Effect ", "")
            or row.get("Item", "")
            or row.get("item", "")
            or ""
        ).strip()
        number = (row.get("Number", "") or row.get("number", "") or "").strip()
        result = (row.get("Result", "") or row.get("result", "") or "").strip()

        # Update current course if specified (for grouped CSV format)
        if course:
            current_course = course

        # Skip rows without spot data
        if not spot:
            continue

        processed_data.append(
            {
                "Course": current_course,
                "Spot": spot,
                "Item": item,
                "Number": number,
                "Result": result,
            }
        )

    logger.info(f"✓ Loaded {len(processed_data)} hopping spots from: {file_path}")
    return processed_data


def group_hopping_by_course(
    hopping_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Group hopping spots by course for automation.

    Args:
        hopping_data: List of spot data from load_hopping_data().

    Returns:
        List of course dictionaries with spots as nested list.
    """
    if not hopping_data:
        return []

    courses = {}
    for spot in hopping_data:
        course_name = spot.get("Course", "Unknown")
        if course_name not in courses:
            courses[course_name] = {
                "コース名": course_name,
                "spots": [],
            }
        courses[course_name]["spots"].append(spot)

    result = list(courses.values())
    logger.info(f"✓ Grouped into {len(result)} courses")
    return result


def find_hopping_spot(
    dataset: List[Dict[str, Any]],
    course: str,
    spot: str,
) -> Optional[Dict[str, Any]]:
    """Find a specific spot in the hopping dataset for validation.

    Args:
        dataset: Full hopping dataset.
        course: Course identifier.
        spot: Spot position.

    Returns:
        Matching spot data or None if not found.
    """
    for entry in dataset:
        if entry.get("Course") == course and entry.get("Spot") == spot:
            return entry
    return None


# ==================== DATA WRITING ====================


def _atomic_write(file_path: str, write_func, encoding: str = "utf-8-sig") -> bool:
    """Perform atomic file write using temp file and rename.

    This prevents file corruption if the process crashes during write.

    Args:
        file_path: Target file path.
        write_func: Function that takes file handle and writes data.
        encoding: File encoding.

    Returns:
        bool: True if write successful, False otherwise.
    """
    directory = os.path.dirname(file_path) or "."
    ensure_directory(directory)

    # Create temp file in same directory for atomic rename
    fd, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            write_func(f)

        # Atomic rename (works on same filesystem)
        os.replace(temp_path, file_path)
        return True
    except Exception as e:
        logger.error(f"Atomic write failed: {e}")
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        return False


def write_csv(
    file_path: str,
    data: List[Dict[str, Any]],
    encoding: str = "utf-8-sig",
    mode: str = "w",
) -> bool:
    """Write list of dictionaries to CSV file with atomic write.

    Uses atomic write (temp file + rename) to prevent corruption.

    Args:
        file_path: Output file path.
        data: List of dictionaries to write.
        encoding: File encoding (default: utf-8-sig for Excel compatibility).
        mode: Write mode (ignored, always uses atomic write).

    Returns:
        bool: True if write successful, False otherwise.
    """
    if not data:
        logger.warning("No data to write")
        return False

    try:
        _validate_file(file_path, check_exists=False)

        # Collect all unique keys from all dictionaries
        all_keys = set().union(*(d.keys() for d in data))

        # Prioritize common fields for better readability
        common_fields = ["test_case_id", "timestamp", "result", "error_message"]
        ordered_fields = [f for f in common_fields if f in all_keys]

        # Add フェス名 and フェスランク after common fields
        festival_fields = ["フェス名", "フェスランク"]
        ordered_fields.extend([f for f in festival_fields if f in all_keys])

        # Sort remaining fields with custom order: group by type (expected, extracted, status)
        # This makes it easier to compare expected vs extracted values
        remaining_keys = [k for k in all_keys if k not in ordered_fields]

        # Separate pre_ and post_ fields
        pre_fields = [k for k in remaining_keys if k.startswith("pre_")]
        post_fields = [k for k in remaining_keys if k.startswith("post_")]
        other_fields = [
            k for k in remaining_keys if not k.startswith(("pre_", "post_"))
        ]

        # Group fields by suffix type for better readability
        def sort_verification_fields_by_type(fields, prefix):
            # Separate by suffix type
            expected_fields = sorted([f for f in fields if f.endswith("_expected")])
            extracted_fields = sorted([f for f in fields if f.endswith("_extracted")])
            status_fields = sorted([f for f in fields if f.endswith("_status")])

            # Return in order: all expected, all extracted, all status
            return expected_fields + extracted_fields + status_fields

        pre_ordered = sort_verification_fields_by_type(pre_fields, "pre_")
        post_ordered = sort_verification_fields_by_type(post_fields, "post_")

        fieldnames = ordered_fields + pre_ordered + post_ordered + sorted(other_fields)

        def write_data(f):
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        if _atomic_write(file_path, write_data, encoding):
            logger.debug(f"Wrote {len(data)} rows to CSV: {file_path}")
            return True
        return False

    except ValueError as e:
        logger.error(f"Invalid file path: {e}")
        return False
    except Exception as e:
        logger.error(f"Error writing CSV: {e}")
        return False


def write_json(
    file_path: str, data: List[Dict[str, Any]], encoding: str = "utf-8-sig"
) -> bool:
    """Write list of dictionaries to JSON file with atomic write.

    Args:
        file_path: Output file path.
        data: List of dictionaries to write.
        encoding: File encoding.

    Returns:
        bool: True if write successful, False otherwise.
    """
    if not data:
        logger.warning("No data to write")
        return False

    try:
        _validate_file(file_path, check_exists=False)

        def write_data(f):
            json.dump(data, f, indent=2, ensure_ascii=False)

        if _atomic_write(file_path, write_data, encoding):
            logger.debug(f"Wrote {len(data)} items to JSON: {file_path}")
            return True
        return False

    except ValueError as e:
        logger.error(f"Invalid file path: {e}")
        return False
    except Exception as e:
        logger.error(f"Error writing JSON: {e}")
        return False



# ==================== RESULT WRITER CLASS ====================


class ResultWriter:
    """Utility class for writing test/automation results to multiple formats with buffering.

    This class provides incremental result writing with automatic resume capability,
    allowing interrupted automation sessions to continue from where they left off.
    Supports CSV and JSON formats.
    """

    # Standard result values
    RESULT_OK = "OK"
    RESULT_NG = "NG"
    RESULT_SKIP = "SKIP"
    RESULT_ERROR = "ERROR"
    # Hopping-specific result value for unverified draws
    RESULT_DRAW_UNCHECKED = "Draw Unchecked"

    def __init__(
        self,
        output_path: str,
        formats: Optional[List[str]] = None,
        auto_write: bool = True,
        resume: bool = True,
    ):
        """Initialize ResultWriter with output path and formats.

        Args:
            output_path: Base path for output files (extension will be adjusted for formats).
            formats: List of formats to write (['csv', 'json']). Default: ['csv'].
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
