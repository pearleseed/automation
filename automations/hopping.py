"""
Pool Hopping Automation

Standard flow for pool hopping with OCR-based verification:

1. Snapshot before -> save (course_X/01_before.png)
2. Touch use button in ROI (tpl_use.png)
3. Touch OK button (tpl_ok.png)
4. Snapshot item -> save (course_X/02_item.png)
5. Verification - ROI scan -> compare CSV -> record OK/NG
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from airtest.core.api import sleep

from core.agent import Agent
from core.base import BaseAutomation, CancellationError, ExecutionStep, StepResult
from core.config import HOPPING_ROI_CONFIG, get_hopping_config, merge_config
from core.data import (
    ResultWriter,
    find_hopping_spot,
    group_hopping_by_course,
    load_hopping_data,
)
from core.detector import (
    OCRTextProcessor,
    TemplateMatcher,
)
from core.utils import StructuredLogger, ensure_directory, get_logger

logger = get_logger(__name__)


class HoppingAutomation(BaseAutomation):
    """Automate Pool Hopping with OCR verification.

    This class automates pool hopping by navigating courses, using items,
    and verifying results through OCR text comparison with CSV data.
    """

    def __init__(
        self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None,
        pause_event=None, preview_callback=None
    ):
        base_config = get_hopping_config()
        cfg = merge_config(base_config, config) if config else base_config
        super().__init__(
            agent, cfg, HOPPING_ROI_CONFIG, 
            cancel_event=cancel_event,
            pause_event=pause_event,
            preview_callback=preview_callback
        )

        self.config = cfg

        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"hopping_{timestamp}.log")
        self.structured_logger = StructuredLogger(
            name="HoppingAutomation", log_file=log_file
        )

        # Detector support (template matching only)
        self.detector = None
        self.use_detector = cfg.get("use_detector", False)
        if self.use_detector:
            self.detector = self._create_detector(cfg, agent)

        # Fuzzy matching config
        fuzzy_config = cfg.get("fuzzy_matching", {})
        self.use_fuzzy_matching = fuzzy_config.get("enabled", True)
        self.fuzzy_threshold = fuzzy_config.get("threshold", 0.7)

        # Resume state file
        self.resume_state_file = os.path.join(self.results_dir, ".hopping_resume.json")

        logger.info("HoppingAutomation initialized")
        self.structured_logger.info(
            f"Log: {log_file} | Fuzzy: {self.use_fuzzy_matching} (threshold: {self.fuzzy_threshold})"
        )

    def _create_detector(self, cfg: Dict[str, Any], agent: Agent) -> Optional[Any]:
        """Factory method for creating template matcher detector."""
        template_config = cfg.get("template_config", {})
        logger.info("Using Template Matcher")
        return TemplateMatcher(
            templates_dir=template_config.get("templates_dir", self.templates_path),
            threshold=template_config.get("threshold", 0.85),
            ocr_engine=agent.ocr_engine,
        )

    def _manage_resume_state(self, action: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Manage resume state: load, save, complete, or clear.

        Uses atomic write (temp file + rename) to prevent corruption.
        """
        try:
            if action == "load":
                if not os.path.exists(self.resume_state_file):
                    return None
                with open(self.resume_state_file, "r", encoding="utf-8-sig") as f:
                    state = json.load(f)
                if state.get("status") == "in_progress":
                    logger.info(
                        f"✓ Resume: course {state.get('current_course')}/{state.get('total_courses')}"
                    )
                    return state
                return None

            elif action == "save":
                state = {
                    "data_path": kwargs["data_path"],
                    "output_path": kwargs["output_path"],
                    "use_detector": kwargs["use_detector"],
                    "start_course_index": kwargs.get("start_course_index", 1),
                    "current_course": kwargs["current_course"],
                    "total_courses": kwargs["total_courses"],
                    "timestamp": datetime.now().isoformat(),
                    "status": "in_progress",
                }
                self._atomic_write_json(self.resume_state_file, state)
                logger.debug(
                    f"Resume saved: {kwargs['current_course']}/{kwargs['total_courses']}"
                )

            elif action == "complete":
                if os.path.exists(self.resume_state_file):
                    with open(self.resume_state_file, "r", encoding="utf-8-sig") as f:
                        state = json.load(f)
                    state["status"] = "completed"
                    state["completed_at"] = datetime.now().isoformat()
                    self._atomic_write_json(self.resume_state_file, state)
                    logger.debug("Resume completed")

            elif action == "clear":
                if os.path.exists(self.resume_state_file):
                    os.remove(self.resume_state_file)
                    logger.debug("Resume cleared")

            return None

        except Exception as e:
            logger.warning(f"Resume state {action} failed: {e}")
            return None

    def _atomic_write_json(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Write JSON file atomically using temp file and rename.

        Args:
            file_path: Target file path.
            data: Dictionary to write as JSON.

        Returns:
            bool: True if successful, False otherwise.
        """
        import tempfile

        directory = os.path.dirname(file_path) or "."
        ensure_directory(directory)

        fd, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, file_path)
            return True
        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            return False

    def compare_results(
        self,
        extracted_data: Dict[str, Any],
        expected_data: Dict[str, Any],
        return_details: bool = True,
        roi_names: Optional[List[str]] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Compare extracted ROI data with expected values from CSV."""
        if not expected_data:
            return True, "No expected data", {} if return_details else None

        roi_fields = set(roi_names) if roi_names else set(extracted_data.keys())
        comparable_fields = {
            k: v for k, v in expected_data.items() if k in roi_fields and v
        }
        if not comparable_fields:
            return True, "No comparable fields", {} if return_details else None

        matches, mismatches, detailed_results = 0, [], {}

        for field, expected_value in comparable_fields.items():
            if field not in extracted_data:
                mismatches.append(f"{field}:missing")
                if return_details:
                    detailed_results[field] = {
                        "status": "missing",
                        "expected": expected_value,
                    }
                continue

            field_data = extracted_data[field]
            if isinstance(field_data, dict):
                extracted_text = field_data.get("text", "").strip()
            else:
                extracted_text = str(field_data).strip()

            validation_result = OCRTextProcessor.validate_field(
                field, extracted_text, expected_value
            )
            text_match = validation_result.status == "match"

            if return_details:
                detailed_results[field] = {
                    "status": validation_result.status,
                    "extracted_text": extracted_text,
                    "extracted_value": validation_result.extracted,
                    "expected": validation_result.expected,
                    "message": validation_result.message,
                    "confidence": validation_result.confidence,
                }

            if text_match:
                matches += 1
            else:
                mismatches.append(
                    f"{field}:{validation_result.extracted}≠{validation_result.expected}"
                )

        total = len(comparable_fields)
        is_ok = matches == total
        message = (
            f"✓ {matches}/{total} matched"
            if is_ok
            else f"✗ {matches}/{total} matched ({', '.join(mismatches[:3])})"
        )
        return is_ok, message, detailed_results if return_details else None

    def run_hopping_course(
        self, course_data: Dict[str, Any], course_idx: int, use_detector: bool = False
    ) -> Dict[str, Any]:
        """Run automation for a single pool hopping course with verification.

        Flow:
        1. Snapshot before
        2. Touch tpl_use in ROI
        3. Touch tpl_ok
        4. Snapshot item
        5. Verification - ROI scan -> compare CSV -> record OK/NG

        Args:
            course_data: Dictionary containing course information from CSV.
            course_idx: Course index number for logging and tracking.
            use_detector: Enable Template detector for item verification.

        Returns:
            Dict with success status and verification details.
        """
        course_name = course_data.get("コース名", f"Course_{course_idx}")
        folder_name = f"course_{course_idx}"

        max_retries = self.config.get("max_step_retries", 5)
        retry_delay = self.config.get("retry_delay", 1.0)

        start_time = time.time()
        course_info = f"Course: {course_name}"
        self.structured_logger.stage_start(course_idx, course_name, course_info)

        screenshot_before = None
        screenshot_item = None
        is_ok = False
        verification_details = {}

        try:
            # ==================== STEP 1: SNAPSHOT BEFORE ====================

            def _capture_before():
                nonlocal screenshot_before
                screenshot_before = self.snapshot_and_save(folder_name, "01_before.png")
                return screenshot_before is not None

            step1 = ExecutionStep(
                step_num=1,
                name="Snapshot Before",
                action=_capture_before,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step1.execute() != StepResult.SUCCESS:
                return {"success": False, "verified": False, "verification_details": {}}

            # ==================== STEP 2: TOUCH USE BUTTON IN ROI ====================

            step2 = ExecutionStep(
                step_num=2,
                name="Touch Use Button",
                # Adaptive wait for use button
                action=lambda: self.wait_and_touch_template("tpl_use.png", timeout=30),
                max_retries=1,
                retry_delay=0,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step2.execute() != StepResult.SUCCESS:
                return {"success": False, "verified": False, "verification_details": {}}

            # ==================== STEP 3: TOUCH OK BUTTON ====================

            step3 = ExecutionStep(
                step_num=3,
                name="Touch OK Button",
                # Adaptive wait for OK button
                action=lambda: self.wait_and_touch_template("tpl_ok.png", timeout=30),
                max_retries=1,
                retry_delay=0,
                post_delay=1.0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step3.execute() != StepResult.SUCCESS:
                return {"success": False, "verified": False, "verification_details": {}}

            # ==================== STEP 4: SNAPSHOT ITEM ====================

            def _capture_item():
                nonlocal screenshot_item
                screenshot_item = self.snapshot_and_save(folder_name, "02_item.png")
                return screenshot_item is not None

            step4 = ExecutionStep(
                step_num=4,
                name="Snapshot Item",
                action=_capture_item,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step4.execute() != StepResult.SUCCESS:
                return {"success": False, "verified": False, "verification_details": {}}

            # ==================== STEP 5: VERIFICATION ====================

            self.structured_logger.subsection_header("VERIFICATION")
            verification_rois = self.config.get(
                "verification_rois",
                ["アイテム名", "獲得数"],
            )

            is_verified = True  # Track if verification was successful

            def _verify():
                nonlocal is_ok, is_verified, screenshot_item, verification_details
                self.check_cancelled("Verification")

                # Scan ROIs from item screenshot
                extracted = self.scan_screen_roi(screenshot_item, verification_rois)

                # Check if we got any extracted data
                if not extracted or all(
                    not v.get("text", "").strip() if isinstance(v, dict) else not str(v).strip()
                    for v in extracted.values()
                ):
                    # No data extracted - mark as unverified (Draw Unchecked)
                    is_verified = False
                    is_ok = False
                    self.structured_logger.warning("Verification: No data extracted - Draw Unchecked")
                    return True

                # Compare with expected data from CSV
                is_ok, msg, verification_details = self.compare_results(
                    extracted,
                    course_data,
                    return_details=True,
                    roi_names=verification_rois,
                )
                self.structured_logger.info(f"Verification: {msg}")

                # Log detailed results
                if verification_details:
                    for field, details in verification_details.items():
                        status = details.get("status", "unknown")
                        if status == "match":
                            result_mark = "✓"
                        elif status == "unverified":
                            result_mark = "?"
                            is_verified = False
                        else:
                            result_mark = "✗"
                        self.structured_logger.info(
                            f"  {result_mark} {field}: {status.upper()} "
                            f"(expected: {details.get('expected', 'N/A')}, "
                            f"extracted: {details.get('extracted_value', 'N/A')})"
                        )
                return True

            step5 = ExecutionStep(
                step_num=5,
                name="Verification",
                action=_verify,
                max_retries=1,
                retry_delay=0,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step5.execute()

            # ==================== FINAL RESULT ====================

            duration = time.time() - start_time
            result_label = "OK" if is_ok else ("Draw Unchecked" if not is_verified else "NG")
            self.structured_logger.stage_end(course_idx, is_ok, duration)
            self.structured_logger.info(f"Course {course_idx} Result: {result_label}")

            return {
                "success": is_ok,
                "verified": is_verified,
                "verification_details": verification_details,
            }

        except CancellationError:
            self.structured_logger.warning(f"Course {course_idx} cancelled by user")
            return {"success": False, "verified": False, "verification_details": {}}
        except Exception as e:
            self.structured_logger.error(
                f"Course {course_idx} failed with exception: {e}"
            )
            import traceback

            self.structured_logger.error(traceback.format_exc())
            return {"success": False, "verified": False, "verification_details": {}}

    def run_all_courses(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        use_detector: bool = False,
        resume: bool = True,
        force_new_session: bool = False,
        start_course_index: int = 1,
    ) -> bool:
        """Run automation for all courses with incremental saving and resume support.

        Results are validated against the dataset before writing:
        - OK: Correct result verified
        - NG: Incorrect result verified
        - Draw Unchecked: Result could not be verified

        Args:
            data_path: Path to CSV/JSON file with hopping data.
            output_path: Output result path (None = auto-generate).
            use_detector: Use detector (Template).
            resume: Resume from existing results if available.
            force_new_session: Force start new session.
            start_course_index: Index of course to start from (1-based).

        Returns:
            bool: True if successful.
        """
        result_writer = None
        start_time = time.time()

        try:
            # Load hopping dataset (supports both CSV and JSON)
            hopping_dataset = load_hopping_data(data_path)
            if not hopping_dataset:
                self.structured_logger.error(
                    f"Failed to load hopping data from {data_path}"
                )
                return False

            # Store dataset for spot validation during automation
            self._hopping_dataset = hopping_dataset

            # Group spots by course for automation
            courses_data = group_hopping_by_course(hopping_dataset)
            if not courses_data:
                self.structured_logger.error("No courses found in hopping data")
                return False

            # Apply start course index filter
            original_count = len(courses_data)
            if start_course_index > 1:
                if start_course_index > original_count:
                    self.structured_logger.error(
                        f"Start course {start_course_index} exceeds total ({original_count})"
                    )
                    return False

                courses_data = courses_data[start_course_index - 1 :]
                self.structured_logger.info(
                    f"✓ Starting from course {start_course_index}/{original_count} | "
                    f"Remaining: {len(courses_data)}"
                )

            # Check for existing resume state
            resume_state = None
            if resume and not force_new_session:
                resume_state = self._manage_resume_state("load")

                if resume_state:
                    if (
                        resume_state.get("data_path") != data_path
                        or resume_state.get("use_detector") != use_detector
                        or resume_state.get("start_course_index", 1)
                        != start_course_index
                    ):
                        logger.warning("Resume mismatch, starting new session")
                        self._manage_resume_state("clear")
                        resume_state = None
                    else:
                        output_path = resume_state.get("output_path")
                        logger.info(f"✓ Resuming: {output_path}")
                        self.structured_logger.info(
                            f"RESUMING from course {resume_state.get('current_course', 1)}"
                        )

            # Setup output path
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detector_suffix = "_detector" if use_detector else ""
                output_path = f"{self.results_dir}/hopping_results_{timestamp}{detector_suffix}.csv"

            # Initialize ResultWriter
            result_writer = ResultWriter(
                output_path,
                formats=["csv", "json", "html"],
                auto_write=True,
                resume=resume,
            )

            # Log automation start
            mode = "Detector + OCR" if use_detector and self.detector else "OCR only"
            config_info = {
                "Mode": mode,
                "Total Courses": len(courses_data),
                "Start Course": (
                    f"{start_course_index}" if start_course_index > 1 else "1"
                ),
                "Output Path": output_path,
                "Data Source": data_path,
                "Resume": resume,
            }

            if resume and result_writer.completed_test_ids:
                config_info["Completed"] = len(result_writer.completed_test_ids)

            self.structured_logger.automation_start(
                "POOL HOPPING AUTOMATION", config_info
            )

            # Process each course
            success_count = 0
            failed_count = 0
            skipped_count = 0

            for idx, course_data in enumerate(courses_data, start_course_index):
                # Save resume state
                self._manage_resume_state(
                    "save",
                    data_path=data_path,
                    output_path=output_path,
                    use_detector=use_detector,
                    start_course_index=start_course_index,
                    current_course=idx,
                    total_courses=len(courses_data) + start_course_index - 1,
                )

                try:
                    self.check_cancelled(f"course {idx}")
                except CancellationError:
                    self.structured_logger.warning(
                        f"Cancellation requested, stopping at course {idx}"
                    )
                    result_writer.flush()
                    result_writer.print_summary()

                    duration = time.time() - start_time
                    summary = {
                        "Total Processed": idx - 1,
                        "Success": success_count,
                        "Failed": failed_count,
                        "Skipped": skipped_count,
                        "Duration": f"{duration:.2f}s",
                        "Status": "CANCELLED",
                        "Resume": f"Can resume from course {idx}",
                    }
                    self.structured_logger.automation_end(
                        "POOL HOPPING AUTOMATION", False, summary
                    )
                    return False

                # Prepare test case with ID
                test_case = course_data.copy()
                test_case["test_case_id"] = idx

                # Skip if already completed
                if resume and result_writer.is_completed(test_case):
                    course_name = course_data.get("コース名", f"Course_{idx}")
                    self.structured_logger.info(
                        f"✓ Course {idx} ({course_name}) already completed, skipping..."
                    )
                    skipped_count += 1
                    continue

                # Log course execution
                course_name = course_data.get("コース名", f"Course_{idx}")
                total = len(courses_data) + start_course_index - 1
                self.structured_logger.info(
                    f"▶ Executing Course {idx}/{total}: {course_name}"
                )

                # Run the course
                course_result = self.run_hopping_course(
                    course_data, idx, use_detector=use_detector
                )

                # Extract results
                is_ok = course_result.get("success", False)
                is_verified = course_result.get("verified", True)
                verification_details = course_result.get("verification_details", {})

                # Determine result status for pool hopping
                # OK = correct result, NG = incorrect, Draw Unchecked = unverified
                if is_verified:
                    if is_ok:
                        success_count += 1
                        result_status = ResultWriter.RESULT_OK
                    else:
                        failed_count += 1
                        result_status = ResultWriter.RESULT_NG
                else:
                    # Unverified draw - could not confirm result
                    failed_count += 1
                    result_status = ResultWriter.RESULT_DRAW_UNCHECKED

                # Prepare result data for pool hopping
                result_data = {
                    "test_case_id": test_case.get("test_case_id"),
                    "Course": course_data.get("コース名", ""),
                }

                # Add spots info if available
                spots = course_data.get("spots", [])
                if spots:
                    result_data["total_spots"] = len(spots)

                # Add verification results with hopping-specific status values
                for field, details in verification_details.items():
                    status = details.get("status", "unknown")
                    result_data[f"{field}_expected"] = details.get("expected", "")
                    result_data[f"{field}_extracted"] = details.get(
                        "extracted_value", ""
                    )
                    # Hopping result values: OK, NG, Draw Unchecked
                    if status == "match":
                        result_data[f"{field}_status"] = "OK"
                    elif status == "unverified":
                        result_data[f"{field}_status"] = "Draw Unchecked"
                    else:
                        result_data[f"{field}_status"] = "NG"

                # Determine error message based on result status
                error_msg = None
                if result_status == ResultWriter.RESULT_NG:
                    error_msg = "Verification failed"
                elif result_status == ResultWriter.RESULT_DRAW_UNCHECKED:
                    error_msg = "Draw result could not be verified"

                # Record result (validated against dataset)
                result_writer.add_result(
                    result_data,
                    result_status,
                    error_message=error_msg,
                )

                # Progress log
                total = len(courses_data) + start_course_index - 1
                self.structured_logger.info(
                    f"Progress: {idx}/{total} | ✓{success_count} ✗{failed_count}"
                )

                sleep(1.0)

            # Final save and summary
            result_writer.flush()
            result_writer.print_summary()

            self._manage_resume_state("complete")

            duration = time.time() - start_time
            total_processed = len(courses_data) - skipped_count
            success_rate = (
                (success_count / total_processed * 100) if total_processed > 0 else 0
            )

            summary = {
                "Total Courses": len(courses_data),
                "Processed": total_processed,
                "Skipped": skipped_count,
                "Success": success_count,
                "Failed": failed_count,
                "Success Rate": f"{success_rate:.1f}%",
                "Total Duration": f"{duration:.2f}s",
                "Results File": output_path,
            }

            all_success = failed_count == 0 and total_processed > 0
            self.structured_logger.automation_end(
                "POOL HOPPING AUTOMATION", all_success, summary
            )

            return True

        except CancellationError:
            if result_writer:
                result_writer.flush()
                result_writer.print_summary()
            self.structured_logger.warning("Automation cancelled by user")
            return False

        except Exception as e:
            self.structured_logger.error(f"Automation failed with exception: {e}")
            import traceback

            self.structured_logger.error(traceback.format_exc())

            if result_writer:
                result_writer.flush()

            return False

    def run(
        self,
        data_path: str,
        use_detector: bool = False,
        output_path: Optional[str] = None,
        force_new_session: bool = False,
        start_course_index: int = 1,
    ) -> bool:
        """Main entry point for Pool Hopping automation.

        Args:
            data_path: Path to CSV/JSON file with course data.
            use_detector: Use detector (Template).
            output_path: Output result path (None = auto-generate).
            force_new_session: Force start new session.
            start_course_index: Index of course to start from (1-based).

        Returns:
            bool: True if successful.
        """
        try:
            self.check_cancelled("before starting")
        except CancellationError:
            self.structured_logger.warning("Automation cancelled before starting")
            return False

        if not self.agent.is_device_connected():
            self.structured_logger.error("✗ Device not connected")
            return False

        try:
            return self.run_all_courses(
                data_path,
                output_path=output_path,
                use_detector=use_detector,
                resume=True,
                force_new_session=force_new_session,
                start_course_index=start_course_index,
            )
        except CancellationError:
            self.structured_logger.warning("Automation cancelled")
            return False
