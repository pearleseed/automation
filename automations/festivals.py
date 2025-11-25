"""
Festival Automation

Standard flow with fallback cache for long text handling:

1. Touch event button (tpl_event.png)
2. Snapshot before touch -> save (rank_E_stage_1/01_before_touch.png)
3. Find & touch festival name (フェス名) - OCR with fallback to cached position
4. Find & touch rank (フェスランク) - OCR with fallback to cached position
5. Snapshot after touch -> save (rank_E_stage_1/02_after_touch.png)
6. Pre-battle verification - ROI scan -> compare CSV -> record OK/NG
7. Touch challenge button (tpl_challenge.png)
8. Optional drag & drop (tpl_drag_object.png -> tpl_drop_target.png)
9. Touch OK button (confirmation dialog) - optional
10. Touch all skip button (tpl_allskip.png)
11. Touch OK button (after skip) - optional
12. Touch result button (tpl_result.png)
13. Snapshot result -> save (rank_E_stage_1/03_result.png)
14. Post-battle verification - ROI scan -> compare CSV -> record OK/NG
15. Touch OK buttons while exists (close all result dialogs)

Fallback mechanism:
- Steps 3 & 4 cache successful touch positions
- If OCR match fails, use cached position from previous stage
- Enables handling of long text that may be truncated or scrolling
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from airtest.core.api import sleep

from core.agent import Agent
from core.base import BaseAutomation, CancellationError, ExecutionStep, StepResult
from core.config import (
    FESTIVAL_CONFIG,
    FESTIVALS_ROI_CONFIG,
    get_festival_config,
    merge_config,
)
from core.data import ResultWriter, load_data
from core.detector import (
    YOLO_AVAILABLE,
    OCRTextProcessor,
    TemplateMatcher,
    YOLODetector,
)
from core.utils import StructuredLogger, ensure_directory, get_logger

logger = get_logger(__name__)


class FestivalAutomation(BaseAutomation):
    """Festival automation with OCR verification and optional detector support.

    This class automates festival gameplay by navigating menus, verifying game state
    through OCR and optional object detection, and recording results. Supports resume
    functionality for interrupted sessions.
    """

    def __init__(
        self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None
    ):
        base_config = get_festival_config()
        cfg = merge_config(base_config, config) if config else base_config
        super().__init__(agent, cfg, FESTIVALS_ROI_CONFIG, cancel_event=cancel_event)

        self.config = cfg

        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"festival_{timestamp}.log")
        self.structured_logger = StructuredLogger(
            name="FestivalAutomation", log_file=log_file
        )

        self.detector = None
        self.use_detector = cfg.get("use_detector")
        if self.use_detector:
            self.detector = self._create_detector(cfg, agent)

        fuzzy_config = cfg.get("fuzzy_matching", {})
        self.use_fuzzy_matching = fuzzy_config.get("enabled", True)
        self.fuzzy_threshold = fuzzy_config.get("threshold", 0.7)

        # Resume state file
        self.resume_state_file = os.path.join(self.results_dir, ".festival_resume.json")

        # Cache for fallback touch positions
        self.last_festival_position: Optional[Tuple[float, float]] = None
        self.last_rank_position: Optional[Tuple[float, float]] = None

        logger.info("FestivalAutomation initialized")
        self.structured_logger.info(
            f"Log: {log_file} | Fuzzy: {self.use_fuzzy_matching} (threshold: {self.fuzzy_threshold})"
        )

    def _create_detector(self, cfg: Dict[str, Any], agent: Agent) -> Optional[Any]:
        """Factory method for creating detectors (YOLO or Template Matching)."""
        detector_type = cfg.get("detector_type", "template")

        if detector_type == "auto":
            if YOLO_AVAILABLE:
                try:
                    yolo_config = cfg.get("yolo_config", {})
                    logger.info("Using YOLO Detector")
                    return YOLODetector(
                        agent=agent,
                        model_path=yolo_config.get("model_path", "yolo11n.pt"),
                        confidence=yolo_config.get("confidence", 0.25),
                        device=yolo_config.get("device", "cpu"),
                    )
                except Exception as e:
                    logger.warning(f"YOLO init failed: {e}, fallback to Template")
            detector_type = "template"

        if detector_type == "yolo":
            try:
                yolo_config = cfg.get("yolo_config", {})
                logger.info("Using YOLO Detector")
                return YOLODetector(
                    agent=agent,
                    model_path=yolo_config.get("model_path", "yolo11n.pt"),
                    confidence=yolo_config.get("confidence", 0.25),
                    device=yolo_config.get("device", "cpu"),
                )
            except Exception as e:
                logger.error(f"YOLO init failed: {e}")
                return None

        if detector_type == "template":
            template_config = cfg.get("template_config", {})
            logger.info("Using Template Matcher")
            return TemplateMatcher(
                templates_dir=template_config.get("templates_dir", self.templates_path),
                threshold=template_config.get("threshold", 0.85),
                ocr_engine=agent.ocr_engine,
            )

        return None

    def _manage_resume_state(self, action: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Manage resume state: load, save, complete, or clear.

        Args:
            action: 'load', 'save', 'complete', or 'clear'
            **kwargs: For 'save': data_path, output_path, use_detector, current_stage, total_stages

        Returns:
            Dict for 'load', None otherwise
        """
        try:
            if action == "load":
                if not os.path.exists(self.resume_state_file):
                    return None
                with open(self.resume_state_file, "r", encoding="utf-8-sig") as f:
                    state = json.load(f)
                if state.get("status") == "in_progress":
                    logger.info(
                        f"✓ Resume: stage {state.get('current_stage')}/{state.get('total_stages')}"
                    )
                    return state
                return None

            elif action == "save":
                state = {
                    "data_path": kwargs["data_path"],
                    "output_path": kwargs["output_path"],
                    "use_detector": kwargs["use_detector"],
                    "start_stage_index": kwargs.get("start_stage_index", 1),
                    "current_stage": kwargs["current_stage"],
                    "total_stages": kwargs["total_stages"],
                    "timestamp": datetime.now().isoformat(),
                    "status": "in_progress",
                }
                with open(self.resume_state_file, "w", encoding="utf-8-sig") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                logger.debug(
                    f"Resume saved: {kwargs['current_stage']}/{kwargs['total_stages']}"
                )

            elif action == "complete":
                if os.path.exists(self.resume_state_file):
                    with open(self.resume_state_file, "r", encoding="utf-8-sig") as f:
                        state = json.load(f)
                    state["status"] = "completed"
                    state["completed_at"] = datetime.now().isoformat()
                    with open(self.resume_state_file, "w", encoding="utf-8-sig") as f:
                        json.dump(state, f, indent=2, ensure_ascii=False)
                    logger.debug("Resume completed")

            elif action == "clear":
                if os.path.exists(self.resume_state_file):
                    os.remove(self.resume_state_file)
                    logger.debug("Resume cleared")

            return None

        except Exception as e:
            logger.warning(f"Resume state {action} failed: {e}")
            return None

    def detect_roi(
        self,
        roi_name: str,
        screenshot: Optional[Any] = None,
        roi_image: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Detect objects in ROI using detector (YOLO/Template)."""
        result = {
            "roi_name": roi_name,
            "detected": False,
            "detections": [],
            "detection_count": 0,
        }

        try:
            if self.detector is None:
                return result

            if roi_image is None:
                screenshot = self.get_screenshot(screenshot)
                if screenshot is None:
                    return result
                roi_image = self.crop_roi(screenshot, roi_name)
                if roi_image is None:
                    return result

            detections = self.detector.detect(roi_image)
            result["detections"] = detections
            result["detection_count"] = len(detections)
            result["detected"] = len(detections) > 0

            logger.info(f"✓ ROI '{roi_name}': {len(detections)} objects detected")
            return result

        except Exception as e:
            logger.error(f"✗ Detect ROI '{roi_name}': {e}")
            return result

    def scan_rois_combined(
        self, screenshot: Optional[Any] = None, roi_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Scan multiple ROIs with both OCR and detector (optimized: crop once per ROI)."""
        try:
            screenshot = self.get_screenshot(screenshot)
            if screenshot is None:
                return {}

            if roi_names is None:
                roi_names = list(FESTIVALS_ROI_CONFIG.keys())

            results = {}

            # Map ROI names to section types for reward detection
            reward_roi_map = {
                "初回クリア報酬": "first_clear",
                "Sランク報酬": "s_rank",
            }

            for roi_name in roi_names:
                roi_image = self.crop_roi(screenshot, roi_name)
                if roi_image is None:
                    results[roi_name] = {
                        "roi_name": roi_name, "text": "", "detected": False,
                        "detections": [], "detection_count": 0,
                    }
                    continue

                # OCR text
                text = ""
                if self.agent.ocr_engine is not None:
                    try:
                        ocr_result = self.agent.ocr_engine.recognize(roi_image)
                        if ocr_result:
                            text = self._clean_ocr_text(ocr_result.get("text", ""))
                    except Exception:
                        pass

                # Use reward section detection for reward ROIs
                if roi_name in reward_roi_map and self.detector and hasattr(self.detector, "detect_reward_section"):
                    section_type = reward_roi_map[roi_name]
                    reward_data = self.detector.detect_reward_section(roi_image, section_type=section_type)
                    items = reward_data.get("items", [])
                    results[roi_name] = {
                        "roi_name": roi_name,
                        "text": text,
                        "detected": bool(items),
                        "detections": items,
                        "detection_count": len(items),
                        "label": reward_data.get("label"),
                        "items_with_quantity": [
                            {"item": d.item, "quantity": d.quantity, "confidence": d.confidence}
                            for d in items
                        ],
                    }
                elif self.detector is not None:
                    detection_result = self.detect_roi(roi_name, roi_image=roi_image)
                    results[roi_name] = {
                        "roi_name": roi_name,
                        "text": text,
                        "detected": detection_result["detected"],
                        "detections": detection_result["detections"],
                        "detection_count": detection_result["detection_count"],
                    }
                else:
                    results[roi_name] = {
                        "roi_name": roi_name, "text": text, "detected": False,
                        "detections": [], "detection_count": 0,
                    }

            logger.info(f"Scanned {len(results)} ROIs (OCR + detector)")
            return results

        except Exception as e:
            logger.error(f"✗ Scan ROIs combined: {e}")
            return {}

    def compare_results(
        self,
        extracted_data: Dict[str, Any],
        expected_data: Dict[str, Any],
        return_details: bool = True,
        roi_names: Optional[List[str]] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Compare extracted ROI data with expected values.

        Args:
            extracted_data: Extracted ROI data from OCR/detector.
            expected_data: Expected values from CSV/JSON.
            return_details: If True, return detailed comparison results.
            roi_names: ROI names to compare. If None, compare all extracted ROIs.

        Returns:
            Tuple of (is_match, message, details_dict).
        """
        if not expected_data:
            return True, "No expected data", {} if return_details else None

        # Determine which ROIs to compare
        roi_fields = set(roi_names) if roi_names else set(extracted_data.keys())

        # Filter expected data to only comparable fields
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
                detected = field_data.get("detected", False)
                detection_count = field_data.get("detection_count", 0)
                detections = field_data.get("detections", [])
                # Handle both DetectionResult dataclass and dict
                if detections:
                    first_det = detections[0]
                    quantity = first_det.quantity if hasattr(first_det, "quantity") else first_det.get("quantity", 0)
                else:
                    quantity = 0
                has_quantity = quantity > 0
            else:
                extracted_text = str(field_data).strip()
                detected, detection_count, has_quantity, quantity = False, 0, False, 0

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
                    "detected": detected,
                    "detection_count": detection_count,
                    "has_quantity": has_quantity,
                    "quantity": quantity,
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

    def run_festival_stage(
        self, stage_data: Dict[str, Any], stage_idx: int, use_detector: bool = False
    ) -> Dict[str, Any]:
        """Run automation for a single festival stage with verification.

        Args:
            stage_data: Dictionary containing stage information (name, rank, expected rewards, etc.).
            stage_idx: Stage index number for logging and tracking.
            use_detector: Enable YOLO/Template detector for item verification (default: False).

        Returns:
            Dict[str, Any]: Result dictionary with keys:
                - success (bool): True if stage completed successfully
                - pre_battle_details (Dict): Pre-battle verification details
                - post_battle_details (Dict): Post-battle verification details
        """
        stage_name = stage_data.get("フェス名", "Unknown")
        rank = stage_data.get("推奨ランク", "Unknown")
        folder_name = f"rank_{rank}_stage_{stage_idx}"
        stage_text = stage_data.get("フェス名", "")
        rank_text = stage_data.get("フェスランク", "")

        max_retries = self.config.get("max_step_retries", 5)
        retry_delay = self.config.get("retry_delay", 1.0)

        start_time = time.time()
        stage_info = f"Rank: {rank} | Stage Text: {stage_text} | Rank Text: {rank_text}"
        self.structured_logger.stage_start(stage_idx, stage_name, stage_info)

        screenshot_after, screenshot_result, is_ok_before, is_ok_after = (
            None,
            None,
            False,
            False,
        )

        try:
            # ==================== NAVIGATION STEPS ====================

            step1 = ExecutionStep(
                step_num=1,
                name="Touch Event Button",
                action=lambda: self.touch_template("tpl_event.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step1.execute() != StepResult.SUCCESS:
                return {
                    "success": False,
                    "pre_battle_details": {},
                    "post_battle_details": {},
                }

            # Step 2: Snapshot Before Touch
            step2 = ExecutionStep(
                step_num=2,
                name="Snapshot Before Touch",
                action=lambda: self.snapshot_and_save(
                    folder_name, "01_before_touch.png"
                )
                is not None,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step2.execute() != StepResult.SUCCESS:
                return {
                    "success": False,
                    "pre_battle_details": {},
                    "post_battle_details": {},
                }

            # Step 3: Find and Touch Stage Name with fallback
            if not stage_text:
                self.structured_logger.error(
                    "Step 3: No stage text (フェス名) provided"
                )
                return {
                    "success": False,
                    "pre_battle_details": {},
                    "post_battle_details": {},
                }

            def _touch_festival():
                # Try OCR matching
                if self.find_and_touch_in_roi(
                    "フェス名",
                    stage_text,
                    threshold=self.fuzzy_threshold,
                    use_fuzzy=self.use_fuzzy_matching,
                ):
                    # Cache position on success
                    roi_config = self.get_roi_config("フェス名")
                    if roi_config:
                        x1, x2, y1, y2 = roi_config
                        self.last_festival_position = ((x1 + x2) / 2, (y1 + y2) / 2)
                    return True

                # Fallback: use cached position
                if self.last_festival_position:
                    self.structured_logger.warning(
                        f"OCR match failed for festival '{stage_text}', using cached position"
                    )
                    success = self.agent.safe_touch(self.last_festival_position)
                    if success:
                        sleep(self.wait_after_touch)
                    return success

                return False

            step3 = ExecutionStep(
                step_num=3,
                name=f"Find & Touch Stage Name '{stage_text}'",
                action=_touch_festival,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step3.execute() != StepResult.SUCCESS:
                return {
                    "success": False,
                    "pre_battle_details": {},
                    "post_battle_details": {},
                }

            # Step 4: Find and Touch Rank with fallback
            if not rank_text:
                self.structured_logger.error(
                    "Step 4: No rank text (フェスランク) provided"
                )
                return {
                    "success": False,
                    "pre_battle_details": {},
                    "post_battle_details": {},
                }

            def _touch_rank():
                # Try to find and touch rank with OCR
                if self.find_and_touch_in_roi(
                    "フェスランク",
                    rank_text,
                    threshold=self.fuzzy_threshold,
                    use_fuzzy=self.use_fuzzy_matching,
                ):
                    # Success - get ROI config to calculate rank position
                    roi_config = self.get_roi_config("フェスランク")
                    if roi_config:
                        x1, x2, y1, y2 = roi_config
                        # Estimate center of ROI as last known rank position
                        self.last_rank_position = ((x1 + x2) / 2, (y1 + y2) / 2)
                    return True

                # Fallback: if match failed but we have cached position, use it
                if self.last_rank_position:
                    self.structured_logger.warning(
                        f"OCR match failed for rank '{rank_text}', using cached position"
                    )
                    success = self.agent.safe_touch(self.last_rank_position)
                    if success:
                        sleep(self.wait_after_touch)
                    return success

                return False

            step4 = ExecutionStep(
                step_num=4,
                name=f"Find & Touch Rank '{rank_text}'",
                action=_touch_rank,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step4.execute() != StepResult.SUCCESS:
                return {
                    "success": False,
                    "pre_battle_details": {},
                    "post_battle_details": {},
                }

            # Step 5: Snapshot After Touch
            def _capture_after():
                nonlocal screenshot_after
                return (
                    screenshot_after := self.snapshot_and_save(
                        folder_name, "02_after_touch.png"
                    )
                ) is not None

            step5 = ExecutionStep(
                step_num=5,
                name="Snapshot After Touch",
                action=_capture_after,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step5.execute() != StepResult.SUCCESS:
                return {
                    "success": False,
                    "pre_battle_details": {},
                    "post_battle_details": {},
                }

            # ==================== PRE-BATTLE VERIFICATION ====================

            # Step 6: Pre-Battle Verification
            self.structured_logger.subsection_header("PRE-BATTLE VERIFICATION")
            pre_battle_rois = FESTIVAL_CONFIG.get(
                "pre_battle_rois",
                [
                    "勝利点数",
                    "推奨ランク",
                    "Sランクボーダー",
                    "初回クリア報酬",
                    "Sランク報酬",
                ],
            )

            pre_battle_details = {}

            def _verify_pre():
                nonlocal is_ok_before, screenshot_after, pre_battle_details
                self.check_cancelled("Pre-battle verification")
                extracted = (
                    self.scan_rois_combined(screenshot_after, pre_battle_rois)
                    if use_detector and self.detector
                    else self.scan_screen_roi(screenshot_after, pre_battle_rois)
                )
                is_ok_before, msg, pre_battle_details = self.compare_results(
                    extracted,
                    stage_data,
                    return_details=True,
                    roi_names=pre_battle_rois,
                )
                self.structured_logger.info(
                    f"Verification{' (with detector)' if use_detector and self.detector else ''}: {msg}"
                )
                # Log detailed results for each field
                if pre_battle_details:
                    for field, details in pre_battle_details.items():
                        status = details.get("status", "unknown")
                        result_mark = "✓" if status == "match" else "✗"
                        self.structured_logger.info(
                            f"  {result_mark} {field}: {status.upper()} "
                            f"(expected: {details.get('expected', 'N/A')}, "
                            f"extracted: {details.get('extracted_value', 'N/A')})"
                        )
                # Continue even if verification fails - don't retry
                return True

            step6 = ExecutionStep(
                step_num=6,
                name="Pre-Battle Verification",
                action=_verify_pre,
                max_retries=1,  # No retries, just verify once
                retry_delay=0,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step6.execute()  # Always continue regardless of result

            # ==================== BATTLE EXECUTION ====================

            self.structured_logger.subsection_header("BATTLE EXECUTION")

            # Step 7: Touch Challenge Button
            step7 = ExecutionStep(
                step_num=7,
                name="Touch Challenge Button",
                action=lambda: self.touch_template("tpl_challenge.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=2,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step7.execute()

            # Step 8: Optional Continuous Hold & Drag
            ExecutionStep(
                step_num=8,
                name="Continuous Hold & Drag Object to Target",
                action=lambda: self.drag_and_drop(
                    "tpl_drag_object.png",
                    "tpl_drop_target.png",
                    hold_duration=0.4,
                    drag_duration=0.5,
                    optional=True,
                ),
                max_retries=1,
                retry_delay=retry_delay,
                optional=True,
                post_delay=1.0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            ).execute()

            # Step 9: Touch OK (Confirmation Dialog)
            step9 = ExecutionStep(
                step_num=9,
                name="Touch OK (Confirmation)",
                action=lambda: self.touch_template("tpl_ok.png"),
                max_retries=max_retries,
                retry_delay=0.3,
                optional=True,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step9.execute()  # Optional, don't check result

            # Step 10: Touch All Skip Button
            step10 = ExecutionStep(
                step_num=10,
                name="Touch All Skip Button",
                action=lambda: self.touch_template("tpl_allskip.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step10.execute()

            # Step 11: Touch OK (After Skip) - Optional
            step11 = ExecutionStep(
                step_num=11,
                name="Touch OK (After Skip)",
                action=lambda: self.touch_template("tpl_ok.png", optional=True),
                max_retries=max_retries,
                retry_delay=0.3,
                optional=True,
                post_delay=2.0,  # Wait for battle to complete
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step11.execute()  # Optional, don't check result

            self.structured_logger.info("Waiting for battle completion...")

            # Step 12: Touch Result Button
            step12 = ExecutionStep(
                step_num=12,
                name="Touch Result Button",
                action=lambda: self.touch_template("tpl_result.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step12.execute() != StepResult.SUCCESS:
                return {
                    "success": False,
                    "pre_battle_details": pre_battle_details,
                    "post_battle_details": {},
                }

            # Step 13: Snapshot Result
            def _capture_result():
                nonlocal screenshot_result
                return (
                    screenshot_result := self.snapshot_and_save(
                        folder_name, "03_result.png"
                    )
                ) is not None

            step13 = ExecutionStep(
                step_num=13,
                name="Snapshot Result",
                action=_capture_result,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step13.execute() != StepResult.SUCCESS:
                return {
                    "success": False,
                    "pre_battle_details": pre_battle_details,
                    "post_battle_details": {},
                }

            # ==================== POST-BATTLE VERIFICATION ====================

            self.structured_logger.subsection_header("POST-BATTLE VERIFICATION")

            # Step 13: Post-Battle Verification
            post_battle_rois = FESTIVAL_CONFIG.get(
                "post_battle_rois",
                [
                    "獲得ザックマネー",
                    "獲得アイテム",
                    "獲得EXP-Ace",
                    "獲得EXP-NonAce",
                    "エース",
                    "非エース",
                ],
            )

            post_battle_details = {}

            def _verify_post():
                nonlocal is_ok_after, screenshot_result, post_battle_details
                self.check_cancelled("Post-battle verification")
                extracted = (
                    self.scan_rois_combined(screenshot_result, post_battle_rois)
                    if use_detector and self.detector
                    else self.scan_screen_roi(screenshot_result, post_battle_rois)
                )
                is_ok_after, msg, post_battle_details = self.compare_results(
                    extracted,
                    stage_data,
                    return_details=True,
                    roi_names=post_battle_rois,
                )
                self.structured_logger.info(
                    f"Verification{' (with detector)' if use_detector and self.detector else ''}: {msg}"
                )
                # Log detailed results for each field
                if post_battle_details:
                    for field, details in post_battle_details.items():
                        status = details.get("status", "unknown")
                        result_mark = "✓" if status == "match" else "✗"
                        self.structured_logger.info(
                            f"  {result_mark} {field}: {status.upper()} "
                            f"(expected: {details.get('expected', 'N/A')}, "
                            f"extracted: {details.get('extracted_value', 'N/A')})"
                        )
                # Continue even if verification fails - don't retry
                return True

            step14 = ExecutionStep(
                step_num=14,
                name="Post-Battle Verification",
                action=_verify_post,
                max_retries=1,  # No retries, just verify once
                retry_delay=0,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step14.execute()  # Always continue regardless of result

            # ==================== FINISHED ====================

            self.structured_logger.subsection_header("FINISHED")

            # Step 15: Touch OK buttons until none remain
            step15 = ExecutionStep(
                step_num=15,
                name="Touch OK (Close All Results)",
                action=lambda: self.touch_template_while_exists(
                    "tpl_ok.png", max_attempts=3, delay_between_touches=0.3
                )
                > 0,
                max_retries=1,
                retry_delay=0,
                optional=True,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step15.execute()  # Optional, don't check result

            # ==================== FINAL RESULT ====================

            final = is_ok_before and is_ok_after
            duration = time.time() - start_time
            self.structured_logger.stage_end(stage_idx, final, duration)

            # Return detailed results for saving
            return {
                "success": final,
                "pre_battle_details": pre_battle_details,
                "post_battle_details": post_battle_details,
            }

        except CancellationError:
            self.structured_logger.warning(f"Stage {stage_idx} cancelled by user")
            return {
                "success": False,
                "pre_battle_details": {},
                "post_battle_details": {},
            }
        except Exception as e:
            self.structured_logger.error(
                f"Stage {stage_idx} failed with exception: {e}"
            )
            import traceback

            self.structured_logger.error(traceback.format_exc())
            return {
                "success": False,
                "pre_battle_details": {},
                "post_battle_details": {},
            }

    def run_all_stages(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        use_detector: bool = False,
        resume: bool = True,
        force_new_session: bool = False,
        start_stage_index: int = 1,
    ) -> bool:
        """
        Run automation for all stages with incremental saving and resume support.

        Args:
            data_path: Path to CSV/JSON file with test data
            output_path: Output result path (None = auto-detect from resume or auto-generate)
            use_detector: Use detector (YOLO/Template)
            resume: Resume from existing results if available (default: True)
            force_new_session: Force start new session even if resume state exists (default: False)
            start_stage_index: Index of stage to start from (1-based, default: 1)

        Returns:
            bool: True if successful
        """
        # Initialize result_writer early to ensure it's available for error handling
        result_writer = None
        start_time = time.time()

        try:
            # Load data
            stages_data = load_data(data_path)
            if not stages_data:
                self.structured_logger.error(f"Failed to load data from {data_path}")
                return False

            # Apply start stage index filter
            original_count = len(stages_data)
            if start_stage_index > 1:
                if start_stage_index > original_count:
                    self.structured_logger.error(
                        f"Start stage {start_stage_index} exceeds total ({original_count})"
                    )
                    return False

                stages_data = stages_data[start_stage_index - 1 :]
                self.structured_logger.info(
                    f"✓ Starting from stage {start_stage_index}/{original_count} | "
                    f"Remaining: {len(stages_data)}"
                )

            # Check for existing resume state
            resume_state = None
            if resume and not force_new_session:
                resume_state = self._manage_resume_state("load")

                # Validate resume state matches current request
                if resume_state:
                    if (
                        resume_state.get("data_path") != data_path
                        or resume_state.get("use_detector") != use_detector
                        or resume_state.get("start_stage_index", 1) != start_stage_index
                    ):
                        logger.warning("Resume mismatch, starting new session")
                        self._manage_resume_state("clear")
                        resume_state = None
                    else:
                        # Use output_path from resume state
                        output_path = resume_state.get("output_path")
                        logger.info(f"✓ Resuming: {output_path}")
                        self.structured_logger.info(
                            f"RESUMING from stage {resume_state.get('current_stage', 1)}"
                        )

            # Setup output path
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detector_suffix = "_detector" if use_detector else ""
                output_path = (
                    f"{self.results_dir}/results_{timestamp}{detector_suffix}.csv"
                )

            # Initialize ResultWriter with auto-write and resume support
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
                "Total Stages": len(stages_data),
                "Start Stage": f"{start_stage_index}" if start_stage_index > 1 else "1",
                "Output Path": output_path,
                "Data Source": data_path,
                "Resume": resume,
                "Max Retries": self.config.get("max_step_retries", 5),
            }

            if resume and result_writer.completed_test_ids:
                config_info["Completed"] = len(result_writer.completed_test_ids)

            self.structured_logger.automation_start("FESTIVAL AUTOMATION", config_info)

            # Process each stage
            success_count = 0
            failed_count = 0
            skipped_count = 0

            for idx, stage_data in enumerate(stages_data, start_stage_index):
                # Save resume state at start of each stage
                self._manage_resume_state(
                    "save",
                    data_path=data_path,
                    output_path=output_path,
                    use_detector=use_detector,
                    start_stage_index=start_stage_index,
                    current_stage=idx,
                    total_stages=len(stages_data) + start_stage_index - 1,
                )

                try:
                    self.check_cancelled(f"stage {idx}")
                except CancellationError:
                    self.structured_logger.warning(
                        f"Cancellation requested, stopping at stage {idx}"
                    )
                    # Ensure results are saved before exiting
                    result_writer.flush()
                    result_writer.print_summary()

                    # Log summary
                    duration = time.time() - start_time
                    summary = {
                        "Total Processed": idx - 1,
                        "Success": success_count,
                        "Failed": failed_count,
                        "Skipped": skipped_count,
                        "Duration": f"{duration:.2f}s",
                        "Status": "CANCELLED",
                        "Resume": f"Can resume from stage {idx}",
                    }
                    self.structured_logger.automation_end(
                        "FESTIVAL AUTOMATION", False, summary
                    )
                    self.structured_logger.info(
                        f"💾 Resume state saved. Run again to continue from stage {idx}"
                    )
                    return False

                # Prepare test case with ID
                test_case = stage_data.copy()
                test_case["test_case_id"] = idx

                # Skip if already completed (resume support)
                if resume and result_writer.is_completed(test_case):
                    stage_name = stage_data.get("フェス名", "Unknown")
                    self.structured_logger.info(
                        f"✓ Stage {idx} ({stage_name}) already completed, skipping..."
                    )
                    skipped_count += 1
                    continue

                # Log stage execution
                stage_name = stage_data.get("フェス名", "Unknown")
                total = len(stages_data) + start_stage_index - 1
                self.structured_logger.info(
                    f"▶ Executing Stage {idx}/{total}: {stage_name}"
                )

                # Run the stage
                stage_result = self.run_festival_stage(
                    stage_data, idx, use_detector=use_detector
                )

                # Extract results
                is_ok = stage_result.get("success", False)
                pre_battle_details = stage_result.get("pre_battle_details", {})
                post_battle_details = stage_result.get("post_battle_details", {})

                # Track results
                if is_ok:
                    success_count += 1
                else:
                    failed_count += 1

                # Prepare result data with only necessary fields
                result_data = {
                    "test_case_id": test_case.get("test_case_id"),
                    "フェス名": stage_data.get("フェス名", ""),
                    "フェスランク": stage_data.get("フェスランク", ""),
                }

                # Add pre-battle verification results (expected, extracted, status)
                for field, details in pre_battle_details.items():
                    status = details.get("status", "unknown")
                    result_data[f"pre_{field}_expected"] = details.get("expected", "")
                    result_data[f"pre_{field}_extracted"] = details.get(
                        "extracted_value", ""
                    )
                    result_data[f"pre_{field}_status"] = (
                        "OK" if status == "match" else "NG"
                    )

                # Add post-battle verification results (expected, extracted, status)
                for field, details in post_battle_details.items():
                    status = details.get("status", "unknown")
                    result_data[f"post_{field}_expected"] = details.get("expected", "")
                    result_data[f"post_{field}_extracted"] = details.get(
                        "extracted_value", ""
                    )
                    result_data[f"post_{field}_status"] = (
                        "OK" if status == "match" else "NG"
                    )

                # Save result (ResultWriter auto-writes immediately)
                result_writer.add_result(
                    result_data,
                    ResultWriter.RESULT_OK if is_ok else ResultWriter.RESULT_NG,
                    error_message=None if is_ok else "Verification failed",
                )

                # Progress log
                total = len(stages_data) + start_stage_index - 1
                self.structured_logger.info(
                    f"Progress: {idx}/{total} | ✓{success_count} ✗{failed_count}"
                )

                # Delay between stages
                sleep(1.0)

            # Final save and summary
            result_writer.flush()
            result_writer.print_summary()

            # Mark resume completed
            self._manage_resume_state("complete")

            # Log completion with detailed summary
            duration = time.time() - start_time
            total_processed = len(stages_data) - skipped_count
            success_rate = (
                (success_count / total_processed * 100) if total_processed > 0 else 0
            )

            summary = {
                "Total Stages": len(stages_data),
                "Processed": total_processed,
                "Skipped": skipped_count,
                "Success": success_count,
                "Failed": failed_count,
                "Success Rate": f"{success_rate:.1f}%",
                "Total Duration": f"{duration:.2f}s",
                "Avg per Stage": (
                    f"{duration/total_processed:.2f}s" if total_processed > 0 else "N/A"
                ),
                "Results File": output_path,
            }

            all_success = failed_count == 0 and total_processed > 0
            self.structured_logger.automation_end(
                "FESTIVAL AUTOMATION", all_success, summary
            )

            return True

        except CancellationError:
            # Ensure results are saved on cancellation
            if result_writer:
                result_writer.flush()
                result_writer.print_summary()

            self.structured_logger.warning("Automation cancelled by user")
            return False

        except Exception as e:
            self.structured_logger.error(f"Automation failed with exception: {e}")
            import traceback

            self.structured_logger.error(traceback.format_exc())

            # Ensure results are saved even on error
            if result_writer:
                result_writer.flush()

            return False

    def run(
        self,
        data_path: str,
        use_detector: bool = False,
        output_path: Optional[str] = None,
        force_new_session: bool = False,
        start_stage_index: int = 1,
    ) -> bool:
        """
        Main entry point.

        Args:
            data_path: Path to CSV/JSON file with test data
            use_detector: Use detector (YOLO/Template)
            output_path: Output result path (None = auto-detect from resume or auto-generate)
            force_new_session: Force start new session even if resume state exists
            start_stage_index: Index of stage to start from (1-based, default: 1)

        Returns:
            bool: True if successful
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
            return self.run_all_stages(
                data_path,
                output_path=output_path,
                use_detector=use_detector,
                resume=True,  # Always enable resume by default
                force_new_session=force_new_session,
                start_stage_index=start_stage_index,
            )
        except CancellationError:
            self.structured_logger.warning("Automation cancelled")
            return False
