"""
Base Automation Module - Common functionality for all automations.

This module provides shared methods used by Festival, Gacha, and Hopping automations.
"""

import os
import cv2
from typing import Dict, List, Optional, Any, Callable
from airtest.core.api import Template, exists, sleep
from enum import Enum

from .agent import Agent
from .utils import get_logger, ensure_directory
from .detector import TextProcessor


logger = get_logger(__name__)


class CancellationError(Exception):
    """Exception raised when automation is cancelled."""
    pass


class StepResult(Enum):
    """Result of a step execution."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStep:
    """Encapsulates a single execution step with retry logic and cancellation support."""
    
    def __init__(self, step_num: int, name: str, action: Callable, max_retries: int = 5,
                 retry_delay: float = 1.0, optional: bool = False, post_delay: float = 0.5,
                 cancel_checker: Optional[Callable] = None, logger: Optional[Any] = None):
        self.step_num = step_num
        self.name = name
        self.action = action
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.optional = optional
        self.post_delay = post_delay
        self.cancel_checker = cancel_checker
        self.logger = logger or get_logger(__name__)
        self.result = None
        self.error_message = ""
    
    def execute(self) -> StepResult:
        """Execute the step with retry logic."""
        if self.cancel_checker:
            self.cancel_checker(self.name)
        
        if hasattr(self.logger, 'step') and callable(getattr(self.logger, 'step', None)):
            getattr(self.logger, 'step')(self.step_num, self.name)
        else:
            self.logger.info(f"[STEP {self.step_num:2d}] {self.name}")
        for attempt in range(1, self.max_retries + 1):
            if self.cancel_checker:
                self.cancel_checker(self.name)
            
            try:
                result = self.action()
                
                if result:
                    self.result = StepResult.SUCCESS
                    if hasattr(self.logger, 'step_success') and callable(getattr(self.logger, 'step_success', None)):
                        getattr(self.logger, 'step_success')(self.step_num, self.name)
                    else:
                        self.logger.info(f"[STEP {self.step_num:2d}] ✓ {self.name} - SUCCESS")
                    
                    if self.post_delay > 0:
                        sleep(self.post_delay)
                    
                    return StepResult.SUCCESS
                
                if attempt < self.max_retries:
                    if hasattr(self.logger, 'step_retry') and callable(getattr(self.logger, 'step_retry', None)):
                        getattr(self.logger, 'step_retry')(self.step_num, self.name, attempt, self.max_retries)
                    else:
                        self.logger.warning(f"[STEP {self.step_num:2d}] {self.name} - RETRY {attempt}/{self.max_retries}")
                    sleep(self.retry_delay)
                    
            except CancellationError:
                raise
            except Exception as e:
                self.error_message = str(e)
                if attempt < self.max_retries:
                    self.logger.warning(f"[STEP {self.step_num:2d}] {self.name} - Error: {e}, retrying...")
                    sleep(self.retry_delay)
                else:
                    break
        if self.optional:
            self.result = StepResult.SKIPPED
            self.logger.info(f"[STEP {self.step_num:2d}] {self.name} - SKIPPED (optional)")
            return StepResult.SKIPPED
        
        self.result = StepResult.FAILED
        error_info = f" | {self.error_message}" if self.error_message else ""
        if hasattr(self.logger, 'step_failed') and callable(getattr(self.logger, 'step_failed', None)):
            getattr(self.logger, 'step_failed')(self.step_num, self.name, self.error_message)
        else:
            self.logger.error(f"[STEP {self.step_num:2d}] ✗ {self.name} - FAILED{error_info}")
        return StepResult.FAILED


class BaseAutomation:
    """Base class for all automation modules with common functionality."""

    def __init__(self, agent: Agent, config: Dict[str, Any], roi_config_dict: Dict[str, Dict[str, Any]], cancel_event=None):
        self.agent = agent
        self.roi_config_dict = roi_config_dict
        self.cancel_event = cancel_event
        self.templates_path = config['templates_path']
        self.snapshot_dir = config['snapshot_dir']
        self.results_dir = config['results_dir']
        self.wait_after_touch = config['wait_after_touch']
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self.cancel_event is not None and self.cancel_event.is_set()
    
    def check_cancelled(self, context: str = ""):
        """Check cancellation and raise CancellationError if cancelled."""
        if self.is_cancelled():
            msg = f"Cancellation requested{f' during {context}' if context else ''}"
            logger.info(msg)
            raise CancellationError(msg)

    def touch_template(self, template_name: str, optional: bool = False) -> bool:
        """Touch template image."""
        try:
            self.check_cancelled(f"touch_template({template_name})")
            
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path):
                return optional

            if self.agent.device is None:
                logger.error("Device not connected")
                return False

            template = Template(template_path)
            pos = exists(template)

            if pos:
                self.check_cancelled(f"touch_template({template_name})")
                self.agent.safe_touch(pos)
                logger.info(f"✓ {template_name}")
                sleep(self.wait_after_touch)
                return True

            return optional if optional else False

        except CancellationError:
            return False
        except Exception as e:
            logger.error(f"✗ {template_name}: {e}") if not optional else None
            return optional

    def touch_template_while_exists(self, template_name: str, max_attempts: int = 5, 
                                   delay_between_touches: float = 0.3) -> int:
        """
        Continuously touch a template while it exists (similar to Airtest's exists mechanism).
        
        Args:
            template_name: Name of the template file
            max_attempts: Maximum number of touches to prevent infinite loops
            delay_between_touches: Delay between each touch attempt
            
        Returns:
            Number of times the template was found and touched
        """
        touch_count = 0
        try:
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path):
                logger.debug(f"Template not found: {template_name}")
                return 0

            if self.agent.device is None:
                logger.error("Device not connected")
                return 0

            template = Template(template_path)
            
            for attempt in range(max_attempts):
                self.check_cancelled(f"touch_template_while_exists({template_name})")
                
                pos = exists(template)
                
                if not pos:
                    # Template no longer exists
                    break
                
                # Template exists, touch it
                self.check_cancelled(f"touch_template_while_exists({template_name})")
                self.agent.safe_touch(pos)
                touch_count += 1
                logger.info(f"✓ {template_name} (touch #{touch_count})")
                
                # Wait before checking again
                sleep(delay_between_touches)
            
            if touch_count > 0:
                logger.info(f"✓ Touched {template_name} {touch_count} time(s)")
            else:
                logger.debug(f"✓ {template_name} not found")
            
            return touch_count

        except CancellationError:
            logger.info(f"Cancelled during touch_template_while_exists({template_name})")
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

    def get_roi_config(self, roi_name: str) -> Optional[Dict[str, Any]]:
        """Get ROI configuration."""
        if roi_name not in self.roi_config_dict:
            logger.error(f"ROI '{roi_name}' not found")
            return None
        return self.roi_config_dict[roi_name]

    def crop_roi(self, screenshot: Any, roi_name: str) -> Optional[Any]:
        """Crop ROI from screenshot."""
        roi_config = self.get_roi_config(roi_name)
        if roi_config is None:
            return None

        coords = roi_config['coords']  # [x1, x2, y1, y2]
        x1, x2, y1, y2 = coords
        
        roi_image = screenshot[y1:y2, x1:x2]
        if roi_image is None or roi_image.size == 0:
            logger.warning(f"✗ ROI '{roi_name}': Invalid crop")
            return None
        
        return roi_image

    def snapshot_and_save(self, folder_name: str, filename: str) -> Optional[Any]:
        """Take screenshot and save to folder."""
        try:
            screenshot = self.agent.snapshot()
            if screenshot is None:
                return None

            folder_path = os.path.join(self.snapshot_dir, folder_name)
            ensure_directory(folder_path)
            file_path = os.path.join(folder_path, filename)
            cv2.imwrite(file_path, screenshot)
            logger.info(f"✓ Saved: {filename}")
            return screenshot

        except Exception as e:
            logger.error(f"✗ Snapshot: {e}")
            return None

    def ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> str:
        """OCR specific ROI region using agent.ocr()."""
        try:
            # Get ROI config using helper
            roi_config = self.get_roi_config(roi_name)
            if roi_config is None:
                return ""

            coords = roi_config['coords']  # [x1, x2, y1, y2]
            x1, x2, y1, y2 = coords
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

            text = self._clean_ocr_text(ocr_result.get('text', '').strip())
            logger.debug(f"ROI '{roi_name}': '{text}'")
            return text

        except Exception as e:
            logger.error(f"✗ OCR ROI '{roi_name}': {e}")
            return ""

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Clean OCR text."""
        if not text:
            return ""
        text = text.replace('\n', ' ').replace('\r', '')
        return ' '.join(text.split()).strip()

    def scan_screen_roi(self, screenshot: Optional[Any] = None,
                       roi_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scan screen according to defined ROI regions."""
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

    def snapshot_and_ocr(self) -> List[Dict[str, Any]]:
        """Take screenshot and OCR to get text + coordinates using agent.ocr()."""
        try:
            # Use agent.ocr() directly - more efficient
            ocr_result = self.agent.ocr()
            if ocr_result is None:
                logger.error("OCR failed")
                return []

            lines = ocr_result.get('lines', [])

            results = []
            for line in lines:
                text = line.get('text', '').strip()
                bbox = line.get('bounding_rect', {})
                if text and bbox:
                    center_x = (bbox.get('x1', 0) + bbox.get('x3', 0)) / 2
                    center_y = (bbox.get('y1', 0) + bbox.get('y3', 0)) / 2
                    results.append({'text': text, 'center': (center_x, center_y)})

            logger.info(f"OCR: {len(results)} texts")
            return results

        except Exception as e:
            logger.error(f"✗ OCR: {e}")
            return []

    def find_text(self, ocr_results: List[Dict[str, Any]], search_text: str, 
                  threshold: float = 0.7, use_fuzzy: bool = True) -> Optional[Dict[str, Any]]:
        """Find text in OCR results using fuzzy matching."""
        if not ocr_results or not search_text:
            return None
        
        if use_fuzzy:
            best_match, best_similarity = None, 0.0
            
            for result in ocr_results:
                ocr_text = result.get('text', '')
                if not ocr_text:
                    continue
                
                similarity = TextProcessor.calculate_similarity(
                    TextProcessor.normalize_text(ocr_text),
                    TextProcessor.normalize_text(search_text)
                )
                
                normalized_ocr = TextProcessor.normalize_text(ocr_text)
                normalized_search = TextProcessor.normalize_text(search_text)
                if normalized_search in normalized_ocr or normalized_ocr in normalized_search:
                    similarity = max(similarity, 0.9)
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = result
                    best_match['similarity'] = similarity
            
            if best_match:
                logger.debug(f"Fuzzy match: '{best_match.get('text')}' ~ '{search_text}' (similarity: {best_similarity:.2f})")
            return best_match
        else:
            search_lower = search_text.lower().strip()
            for result in ocr_results:
                if search_lower in result.get('text', '').lower():
                    result['similarity'] = 1.0
                    return result
            return None

    def ocr_roi_with_lines(self, roi_name: str) -> List[Dict[str, Any]]:
        """OCR specific ROI region and return individual text lines with coordinates."""
        try:
            # Get ROI config using helper
            roi_config = self.get_roi_config(roi_name)
            if roi_config is None:
                return []

            coords = roi_config['coords']  # [x1, x2, y1, y2]
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # OCR the ROI region
            ocr_result = self.agent.ocr(region)
            if ocr_result is None:
                logger.warning(f"✗ ROI '{roi_name}': OCR failed")
                return []

            # Parse individual lines (similar to snapshot_and_ocr)
            lines = ocr_result.get('lines', [])
            results = []

            for line in lines:
                text = line.get('text', '').strip()
                bbox = line.get('bounding_rect', {})
                if text and bbox:
                    # Coordinates are relative to region, convert to absolute screen coordinates
                    center_x = (bbox.get('x1', 0) + bbox.get('x3', 0)) / 2 + x1
                    center_y = (bbox.get('y1', 0) + bbox.get('y3', 0)) / 2 + y1
                    results.append({'text': text, 'center': (center_x, center_y)})

            logger.debug(f"✓ ROI '{roi_name}': Found {len(results)} texts")
            return results

        except Exception as e:
            logger.error(f"✗ OCR ROI with lines '{roi_name}': {e}")
            return []

    def find_and_touch_in_roi(self, roi_name: str, search_text: str,
                              threshold: float = 0.7, use_fuzzy: bool = True) -> bool:
        """Find text in specific ROI region and touch it using fuzzy matching."""
        try:
            self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")

            match_type = "fuzzy" if use_fuzzy else "exact"
            logger.info(f"Find & touch '{search_text}' in ROI '{roi_name}' ({match_type} matching)")

            # OCR the ROI to get list of texts with coordinates
            ocr_results = self.ocr_roi_with_lines(roi_name)

            if not ocr_results:
                logger.warning(f"✗ No text found in ROI '{roi_name}'")
                return False

            # Log OCR results for debugging
            logger.debug(f"OCR found {len(ocr_results)} text(s) in ROI '{roi_name}': {[r.get('text', '') for r in ocr_results]}")

            self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")

            # Find the text in results using fuzzy matching
            text_info = self.find_text(ocr_results, search_text, threshold=threshold, use_fuzzy=use_fuzzy)

            if text_info:
                self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")
                
                matched_text = text_info.get('text', '')
                similarity = text_info.get('similarity', 1.0)
                
                if use_fuzzy:
                    logger.info(f"✓ Found '{matched_text}' ~ '{search_text}' in ROI '{roi_name}' "
                              f"(similarity: {similarity:.2f}) at {text_info['center']}")
                else:
                    logger.info(f"✓ Found '{matched_text}' in ROI '{roi_name}' at {text_info['center']}")
                
                success = self.agent.safe_touch(text_info['center'])
                if success:
                    sleep(self.wait_after_touch)
                return success

            logger.warning(f"✗ Text '{search_text}' not found in ROI '{roi_name}' "
                         f"(threshold: {threshold:.2f}, fuzzy: {use_fuzzy})")
            return False

        except CancellationError:
            return False
        except Exception as e:
            logger.error(f"✗ Find and touch in ROI '{roi_name}': {e}")
            return False
