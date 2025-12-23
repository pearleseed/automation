"""
Base Automation Module - Common functionality for all automations.

This module provides shared methods used by Festival, Gacha, and Hopping automations,
including template matching, OCR processing, screenshot capture, and cancellation support.
"""

import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import cv2
from airtest.core.api import Template, exists, sleep, wait
from airtest.core.error import TargetNotFoundError

from .agent import Agent
from .config import DEFAULT_TOUCH_TIMES
from .detector import TextProcessor
from .utils import ensure_directory, get_logger, safe_join_path, sanitize_filename

logger = get_logger(__name__)


class CancellationError(Exception):
    """Exception raised when automation is cancelled by user or system."""

    def __init__(self, message: str = "Operation cancelled", context: str = ""):
        self.context = context
        super().__init__(f"{message}{f' during {context}' if context else ''}")


class StepResult(Enum):
    """Result of a step execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStep:
    """Encapsulates a single execution step with retry logic and cancellation support.

    This class provides a structured way to execute automation steps with automatic
    retry on failure, cancellation checking, and detailed logging.
    """

    def __init__(
        self,
        step_num: int,
        name: str,
        action: Callable,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        optional: bool = False,
        post_delay: float = 0.5,
        cancel_checker: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        self.step_num = step_num
        self.name = name
        self.action = action
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.optional = optional
        self.post_delay = post_delay
        self.cancel_checker = cancel_checker
        self.logger = logger or get_logger(__name__)
        self.result: Optional[StepResult] = None
        self.error_message = ""

    def execute(self) -> StepResult:
        """Execute the step with retry logic and cancellation checking.

        Returns:
            StepResult: SUCCESS if step completed successfully, FAILED if all retries exhausted,
                       SKIPPED if step is optional and failed.
        """
        if self.cancel_checker:
            self.cancel_checker(self.name)

        if hasattr(self.logger, "step") and callable(
            getattr(self.logger, "step", None)
        ):
            getattr(self.logger, "step")(self.step_num, self.name)
        else:
            self.logger.info(f"[STEP {self.step_num:2d}] {self.name}")
        for attempt in range(1, self.max_retries + 1):
            if self.cancel_checker:
                self.cancel_checker(self.name)

            try:
                result = self.action()

                if result:
                    self.result = StepResult.SUCCESS
                    if hasattr(self.logger, "step_success") and callable(
                        getattr(self.logger, "step_success", None)
                    ):
                        getattr(self.logger, "step_success")(self.step_num, self.name)
                    else:
                        self.logger.info(
                            f"[STEP {self.step_num:2d}] ✓ {self.name} - SUCCESS"
                        )

                    if self.post_delay > 0:
                        sleep(self.post_delay)

                    return StepResult.SUCCESS

                if attempt < self.max_retries:
                    if hasattr(self.logger, "step_retry") and callable(
                        getattr(self.logger, "step_retry", None)
                    ):
                        getattr(self.logger, "step_retry")(
                            self.step_num, self.name, attempt, self.max_retries
                        )
                    else:
                        self.logger.warning(
                            f"[STEP {self.step_num:2d}] {self.name} - RETRY {attempt}/{self.max_retries}"
                        )
                    sleep(self.retry_delay)

            except CancellationError:
                raise
            except Exception as e:
                self.error_message = str(e)
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"[STEP {self.step_num:2d}] {self.name} - Error: {e}, retrying..."
                    )
                    sleep(self.retry_delay)
                else:
                    break
        if self.optional:
            self.result = StepResult.SKIPPED
            self.logger.info(
                f"[STEP {self.step_num:2d}] {self.name} - SKIPPED (optional)"
            )
            return StepResult.SKIPPED

        self.result = StepResult.FAILED
        error_info = f" | {self.error_message}" if self.error_message else ""
        if hasattr(self.logger, "step_failed") and callable(
            getattr(self.logger, "step_failed", None)
        ):
            getattr(self.logger, "step_failed")(
                self.step_num, self.name, self.error_message
            )
        else:
            self.logger.error(
                f"[STEP {self.step_num:2d}] ✗ {self.name} - FAILED{error_info}"
            )
        return StepResult.FAILED


class BaseAutomation:
    """Base class for all automation modules with common functionality.

    This class provides shared methods for Festival, Gacha, and Hopping automations,
    including template matching, OCR processing, screenshot capture, and cancellation support.
    """

    def __init__(
        self,
        agent: Agent,
        config: Dict[str, Any],
        roi_config_dict: Dict[str, list],
        cancel_event=None,
        pause_event=None,
        preview_callback: Optional[Callable] = None,
    ):
        self.agent = agent
        self.roi_config_dict = roi_config_dict
        self.cancel_event = cancel_event
        self.pause_event = pause_event  # For pause/resume support
        self.preview_callback = preview_callback  # For live preview updates
        self.templates_path = config["templates_path"]
        self.snapshot_dir = config["snapshot_dir"]
        self.results_dir = config["results_dir"]
        self.wait_after_touch = config["wait_after_touch"]
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self.cancel_event is not None and self.cancel_event.is_set()

    def wait_if_paused(self, context: str = "") -> None:
        """Wait if automation is paused, checking for cancellation periodically.
        
        Args:
            context: Description of current operation for logging.
        """
        if self.pause_event is None:
            return
        
        # Wait for pause_event to be set (not paused)
        while not self.pause_event.is_set():
            # Check for cancellation while paused
            if self.is_cancelled():
                raise CancellationError(context=context)
            # Wait with timeout to allow periodic cancellation check
            self.pause_event.wait(timeout=0.5)

    def check_cancelled(self, context: str = "") -> None:
        """Check cancellation and pause state, raise CancellationError if cancelled.

        Args:
            context: Description of current operation for logging.

        Raises:
            CancellationError: If cancellation was requested.
        """
        # First check for pause
        self.wait_if_paused(context)
        
        # Then check for cancellation
        if self.is_cancelled():
            logger.info(
                f"Cancellation requested during {context}"
                if context
                else "Cancellation requested"
            )
            raise CancellationError(context=context)

    def update_preview(self, screenshot=None, ocr_text: str = "", confidence: float = 0.0):
        """Update live preview if callback is set (thread-safe).
        
        Args:
            screenshot: Screenshot image array.
            ocr_text: OCR result text.
            confidence: OCR confidence score (0.0-1.0).
        """
        if self.preview_callback:
            try:
                self.preview_callback(screenshot, ocr_text, confidence)
            except Exception as e:
                logger.debug(f"Preview callback error: {e}")

    def touch_template(
        self,
        template_name: str,
        optional: bool = False,
        times: int = DEFAULT_TOUCH_TIMES,
        timeout: Optional[float] = None,
    ) -> bool:
        """Touch template image on screen with cancellation support and adaptive wait.

        Args:
            template_name: Name of template image file in templates directory.
            optional: If True, returns True even if template not found (default: False).
            times: Number of times to touch the template (default: 1).
            timeout: Max wait time in seconds (adaptive wait). If None, checks immediately (unless times > 1, where it might not wait at all).
                     If provided (e.g. 30), waits up to that many seconds for template to appear.

        Returns:
            bool: True if template found and touched successfully, or if optional=True.
        """
        try:
            self.check_cancelled(f"touch_template({template_name})")

            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path) or self.agent.device is None:
                if self.agent.device is None:
                    logger.error("Device not connected")
                return optional

            # Adaptive wait or immediate check
            pos = None
            if timeout is not None and timeout > 0:
                logger.info(f"Waiting for {template_name}... (timeout: {timeout}s)")

                # Helper to check cancellation during wait
                def _check_cancel_callback():
                    try:
                        self.check_cancelled(f"waiting for {template_name}")
                        return False  # Continue waiting
                    except CancellationError:
                        raise

                try:
                    pos = wait(
                        Template(template_path),
                        timeout=timeout,
                        interval=0.5,
                        intervalfunc=_check_cancel_callback
                    )
                except (TargetNotFoundError, CancellationError):
                    pos = None
            else:
                 pos = exists(Template(template_path))

            if not pos:
                if timeout is not None and timeout > 0 and not optional:
                     logger.warning(f"✗ Timeout waiting for {template_name} ({timeout}s)")
                return optional

            for i in range(times):
                self.check_cancelled(f"touch_template({template_name})")
                self.agent.safe_touch(pos)
                log_msg = f"✓ {template_name}" + (
                    f" (touch #{i+1}/{times})" if times > 1 else ""
                )
                logger.info(log_msg)
                
                # If touching multiple times, we might want to re-check existence? 
                # Original logic didn't re-check, so adhering to that for now.
                
                if i < times - 1:
                    sleep(self.wait_after_touch * 0.5)

            sleep(self.wait_after_touch)
            return True

        except CancellationError:
            return False
        except Exception as e:
            if not optional:
                logger.error(f"✗ {template_name}: {e}")
            return optional

    def wait_and_touch_template(
        self,
        template_name: str,
        timeout: Optional[float] = None,
        interval: float = 0.5,
    ) -> bool:
        """Wrapper for touch_template with timeout (for backward compatibility)."""
        return self.touch_template(template_name, timeout=timeout or 30)

    def touch_template_while_exists(
        self,
        template_name: str,
        max_attempts: int = 5,
        delay_between_touches: float = 0.3,
    ) -> int:
        """Touch template repeatedly while it exists (returns touch count)."""
        touch_count = 0
        try:
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path) or self.agent.device is None:
                logger.debug(f"Template not found: {template_name}")
                return 0

            template = Template(template_path)
            for _ in range(max_attempts):
                self.check_cancelled(f"touch_template_while_exists({template_name})")
                pos = exists(template)
                if not pos:
                    break

                self.agent.safe_touch(pos)
                touch_count += 1
                logger.info(f"✓ {template_name} (touch #{touch_count})")
                sleep(delay_between_touches)

            if touch_count > 0:
                logger.info(f"✓ Touched {template_name} {touch_count} time(s)")
            else:
                logger.debug(f"✓ {template_name} not found")

            return touch_count

        except CancellationError:
            logger.info(
                f"Cancelled during touch_template_while_exists({template_name})"
            )
            return touch_count
        except Exception as e:
            logger.error(f"✗ touch_template_while_exists({template_name}): {e}")
            return touch_count

    def get_screenshot(self, screenshot: Optional[Any] = None) -> Optional[Any]:
        """Get screenshot (use cached or take new)."""
        if screenshot is not None:
            return screenshot

        screenshot = self.agent.snapshot()
        if screenshot is None:
            logger.error("Cannot get screenshot")
        return screenshot

    def get_roi_config(self, roi_name: str) -> Optional[list]:
        """Get ROI configuration [x1, x2, y1, y2]."""
        if roi_name not in self.roi_config_dict:
            logger.error(f"ROI '{roi_name}' not found")
            return None
        return self.roi_config_dict[roi_name]

    def crop_roi(self, screenshot: Any, roi_name: str) -> Optional[Any]:
        """Crop ROI from screenshot."""
        roi_config = self.get_roi_config(roi_name)
        if roi_config is None:
            return None

        x1, x2, y1, y2 = roi_config  # [x1, x2, y1, y2]

        roi_image = screenshot[y1:y2, x1:x2]
        if roi_image is None or roi_image.size == 0:
            logger.warning(f"✗ ROI '{roi_name}': Invalid crop")
            return None

        return roi_image

    def snapshot_and_save(self, folder_name: str, filename: str) -> Optional[Any]:
        """Take screenshot and save to folder with sanitized filenames.

        Uses safe_join_path to prevent path traversal attacks.
        Updates live preview if callback is set.

        Args:
            folder_name: Folder name (will be sanitized).
            filename: File name (will be sanitized).

        Returns:
            Screenshot array if successful, None otherwise.
        """
        try:
            screenshot = self.agent.snapshot()
            if screenshot is None:
                return None

            # Update live preview
            self.update_preview(screenshot=screenshot)

            # Use safe_join_path to prevent path traversal
            folder_path = safe_join_path(self.snapshot_dir, folder_name)
            if folder_path is None:
                logger.error(f"✗ Invalid folder path: {folder_name}")
                return None

            if not ensure_directory(folder_path):
                logger.error(f"✗ Cannot create folder: {folder_path}")
                return None

            # Sanitize filename and create full path safely
            safe_filename = sanitize_filename(filename)
            file_path = safe_join_path(folder_path, safe_filename)
            if file_path is None:
                logger.error(f"✗ Invalid file path: {filename}")
                return None

            cv2.imwrite(file_path, screenshot)
            logger.info(f"✓ Saved: {safe_filename}")
            return screenshot

        except Exception as e:
            logger.error(f"✗ Snapshot: {e}")
            return None

    def ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> str:
        """OCR text from ROI region (returns cleaned text)."""
        try:
            # Get ROI config using helper
            roi_config = self.get_roi_config(roi_name)
            if roi_config is None:
                return ""

            x1, x2, y1, y2 = roi_config  # [x1, x2, y1, y2]
            region = (x1, y1, x2, y2)

            if screenshot is None:
                ocr_result = self.agent.ocr(region)
            else:
                roi_image = self.crop_roi(screenshot, roi_name)
                if roi_image is None or self.agent.ocr_engine is None:
                    return ""
                ocr_result = self.agent.ocr_engine.recognize(roi_image)

            if ocr_result is None:
                logger.warning(f"✗ ROI '{roi_name}': OCR failed")
                return ""

            text = self._clean_ocr_text(ocr_result.get("text", "").strip())
            confidence = ocr_result.get("confidence", 0.8)
            
            # Update live preview with OCR result
            self.update_preview(ocr_text=f"[{roi_name}] {text}", confidence=confidence)
            
            logger.debug(f"ROI '{roi_name}': '{text}'")
            return text

        except Exception as e:
            logger.error(f"✗ OCR ROI '{roi_name}': {e}")
            return ""

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Clean OCR text using TextProcessor."""
        if not text:
            return ""
        # Use TextProcessor for consistent text cleaning
        cleaned = TextProcessor.clean_ocr_artifacts(text)
        # Additional normalization: remove newlines and normalize spaces
        cleaned = cleaned.replace("\n", " ").replace("\r", "")
        return " ".join(cleaned.split()).strip()

    def scan_screen_roi(
        self, screenshot: Optional[Any] = None, roi_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Scan multiple ROI regions and return OCR results dict."""
        try:
            # Get screenshot using helper
            screenshot = self.get_screenshot(screenshot)
            if screenshot is None:
                return {}

            # Determine ROI list to scan
            if roi_names is None:
                roi_names = list(self.roi_config_dict.keys())

            # OCR each ROI
            results = {}
            for roi_name in roi_names:
                try:
                    text = self.ocr_roi(roi_name, screenshot)
                    results[roi_name] = text
                    logger.debug(f"✓ {roi_name}: '{text}'")
                except Exception as e:
                    logger.warning(f"✗ {roi_name}: {e}")
                    results[roi_name] = ""

            logger.info(f"Scanned {len(results)} ROIs")
            return results

        except Exception as e:
            logger.error(f"✗ Scan screen ROI: {e}")
            return {}

    def find_text(
        self,
        ocr_results: List[Dict[str, Any]],
        search_text: str,
        threshold: float = 0.7,
        use_fuzzy: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Find text in OCR results (use_fuzzy: similarity matching)."""
        if not ocr_results or not search_text:
            return None

        if use_fuzzy:
            best_match, best_similarity = None, 0.0

            for result in ocr_results:
                ocr_text = result.get("text", "")
                if not ocr_text:
                    continue

                similarity = TextProcessor.calculate_similarity(
                    TextProcessor.normalize_text(ocr_text),
                    TextProcessor.normalize_text(search_text),
                )

                normalized_ocr = TextProcessor.normalize_text(ocr_text)
                normalized_search = TextProcessor.normalize_text(search_text)
                if (
                    normalized_search in normalized_ocr
                    or normalized_ocr in normalized_search
                ):
                    similarity = max(similarity, 0.9)

                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = result
                    best_match["similarity"] = similarity

            if best_match:
                logger.debug(
                    f"Fuzzy match: '{best_match.get('text')}' ~ '{search_text}' (similarity: {best_similarity:.2f})"
                )
            return best_match
        else:
            search_lower = search_text.lower().strip()
            for result in ocr_results:
                if search_lower in result.get("text", "").lower():
                    result["similarity"] = 1.0
                    return result
            return None

    def ocr_roi_with_lines(self, roi_name: str) -> List[Dict[str, Any]]:
        """OCR ROI and return text lines with coordinates."""
        try:
            # Get ROI config using helper
            roi_config = self.get_roi_config(roi_name)
            if roi_config is None:
                return []

            x1, x2, y1, y2 = roi_config  # [x1, x2, y1, y2]
            region = (x1, y1, x2, y2)

            # OCR the ROI region
            ocr_result = self.agent.ocr(region)
            if ocr_result is None:
                logger.warning(f"✗ ROI '{roi_name}': OCR failed")
                return []

            # Parse individual lines from OCR result
            lines = ocr_result.get("lines", [])
            results = []

            for line in lines:
                text = line.get("text", "").strip()
                bbox = line.get("bounding_rect", {})
                if text and bbox:
                    # Coordinates are relative to region, convert to absolute screen coordinates
                    center_x = (bbox.get("x1", 0) + bbox.get("x3", 0)) / 2 + x1
                    center_y = (bbox.get("y1", 0) + bbox.get("y3", 0)) / 2 + y1
                    results.append({"text": text, "center": (center_x, center_y)})

            logger.debug(f"✓ ROI '{roi_name}': Found {len(results)} texts")
            return results

        except Exception as e:
            logger.error(f"✗ OCR ROI with lines '{roi_name}': {e}")
            return []

    def find_and_touch_in_roi(
        self,
        roi_name: str,
        search_text: str,
        threshold: float = 0.7,
        use_fuzzy: bool = True,
    ) -> bool:
        """Find and touch text in specified ROI region using OCR.

        Args:
            roi_name: Name of ROI region defined in ROI config.
            search_text: Text to search for in the ROI.
            threshold: Similarity threshold for fuzzy matching (0.0-1.0, default: 0.7).
            use_fuzzy: Enable fuzzy text matching (default: True).

        Returns:
            bool: True if text found and touched successfully, False otherwise.
        """
        try:
            self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")

            match_type = "fuzzy" if use_fuzzy else "exact"
            logger.info(
                f"Find & touch '{search_text}' in ROI '{roi_name}' ({match_type} matching)"
            )

            # OCR the ROI to get list of texts with coordinates
            ocr_results = self.ocr_roi_with_lines(roi_name)

            if not ocr_results:
                logger.warning(f"✗ No text found in ROI '{roi_name}'")
                return False

            # Log OCR results for debugging
            logger.debug(
                f"OCR found {len(ocr_results)} text(s) in ROI '{roi_name}': {[r.get('text', '') for r in ocr_results]}"
            )

            self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")

            # Find the text in results using fuzzy matching
            text_info = self.find_text(
                ocr_results, search_text, threshold=threshold, use_fuzzy=use_fuzzy
            )

            if text_info:
                self.check_cancelled(
                    f"find_and_touch_in_roi({roi_name}, {search_text})"
                )

                matched_text = text_info.get("text", "")
                similarity = text_info.get("similarity", 1.0)

                if use_fuzzy:
                    logger.info(
                        f"✓ Found '{matched_text}' ~ '{search_text}' in ROI '{roi_name}' "
                        f"(similarity: {similarity:.2f}) at {text_info['center']}"
                    )
                else:
                    logger.info(
                        f"✓ Found '{matched_text}' in ROI '{roi_name}' at {text_info['center']}"
                    )

                success = self.agent.safe_touch(text_info["center"])
                if success:
                    sleep(self.wait_after_touch)
                return success

            logger.warning(
                f"✗ Text '{search_text}' not found in ROI '{roi_name}' "
                f"(threshold: {threshold:.2f}, fuzzy: {use_fuzzy})"
            )
            return False

        except CancellationError:
            return False
        except Exception as e:
            logger.error(f"✗ Find and touch in ROI '{roi_name}': {e}")
            return False

    def drag_and_drop(
        self,
        object_template: str,
        target_template: str,
        hold_duration: float = 0.3,
        drag_duration: float = 0.5,
        optional: bool = False,
    ) -> bool:
        """Drag object with continuous touch (hold then drag in single swipe).

        Args:
            object_template: Source template filename.
            target_template: Target template filename.
            hold_duration: Hold time at start position (default: 0.3s).
            drag_duration: Drag time to target (default: 0.5s).
            optional: Return True if object not found instead of False.

        Returns:
            bool: True if successful or optional=True when object not found.
        """
        try:
            self.check_cancelled(
                f"drag_and_drop({object_template} -> {target_template})"
            )

            templates = {
                object_template: os.path.join(self.templates_path, object_template),
                target_template: os.path.join(self.templates_path, target_template),
            }

            for name, path in templates.items():
                if not os.path.exists(path):
                    logger.warning(f"Template not found: {name}")
                    return optional

            obj_pos = exists(Template(templates[object_template]))
            if not obj_pos:
                logger.debug(f"Object '{object_template}' not found on screen")
                return optional

            target_pos = exists(Template(templates[target_template]))
            if not target_pos:
                logger.warning(f"Target '{target_template}' not found on screen")
                return False

            total_duration = hold_duration + drag_duration
            steps = max(10, int(total_duration * 20))

            logger.info(
                f"Drag '{object_template}' → '{target_template}' "
                f"({total_duration:.1f}s: {hold_duration:.1f}s hold + {drag_duration:.1f}s drag)"
            )

            if self.agent.safe_swipe(
                obj_pos, target_pos, duration=total_duration, steps=steps
            ):
                sleep(self.wait_after_touch)
                return True

            logger.warning("Drag failed")
            return False

        except CancellationError:
            return False
        except Exception as e:
            if not optional:
                logger.error(f"Drag and drop error: {e}")
            return optional
