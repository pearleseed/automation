"""
Festival Automation
Standard flow:
1. touch(Template("tpl_event.png"))
2. snapshot -> save to folder (rank_E_stage_1/01_before_touch.png)
3. find_and_touch_in_roi('フェス名', stage_text) - OCR in フェス名 ROI -> find text -> touch
3.5. find_and_touch_in_roi('フェスランク', rank_text) - OCR in フェスランク ROI -> find text -> touch
4. snapshot -> save to folder (rank_E_stage_1/02_after_touch.png)
5. ROI scan -> compare CSV -> record OK/NG (Pre-battle verification)
6. touch(Template("tpl_challenge.png"))
7. touch(Template("tpl_ok.png")) - confirmation dialog
8. touch(Template("tpl_allskip.png"))
9. touch(Template("tpl_ok.png")) - after skip
10. touch(Template("tpl_result.png"))
11. snapshot -> save to folder (rank_E_stage_1/03_result.png)
12. ROI scan -> compare CSV -> record OK/NG (Post-battle verification)
13. touch(Template("tpl_ok.png")) if exists - close result (first)
14. touch(Template("tpl_ok.png")) if exists - close result (second)
15. Repeat
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from airtest.core.api import sleep
import time
import os
import json

from core.base import BaseAutomation, CancellationError, ExecutionStep, StepResult
from core.agent import Agent
from core.utils import get_logger, StructuredLogger, ensure_directory
from core.data import ResultWriter, load_data
from core.config import (
    FESTIVALS_ROI_CONFIG,
    FESTIVAL_CONFIG, get_festival_config, merge_config
)
from core.detector import YOLODetector, TemplateMatcher, YOLO_AVAILABLE, OCRTextProcessor

logger = get_logger(__name__)


class FestivalAutomation(BaseAutomation):
    """Festival automation with OCR verification and optional detector support."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None):
        base_config = get_festival_config()
        cfg = merge_config(base_config, config) if config else base_config
        super().__init__(agent, cfg, FESTIVALS_ROI_CONFIG, cancel_event=cancel_event)
        
        self.config = cfg
        
        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"festival_{timestamp}.log")
        self.structured_logger = StructuredLogger(name="FestivalAutomation", log_file=log_file)
        
        self.detector = None
        self.use_detector = cfg.get('use_detector')
        if self.use_detector:
            self.detector = self._create_detector(cfg, agent)
        
        fuzzy_config = cfg.get('fuzzy_matching', {})
        self.use_fuzzy_matching = fuzzy_config.get('enabled', True)
        self.fuzzy_threshold = fuzzy_config.get('threshold', 0.7)
        
        # Resume state file
        self.resume_state_file = os.path.join(self.results_dir, ".festival_resume.json")
        
        logger.info("FestivalAutomation initialized")
        self.structured_logger.info(f"Log: {log_file} | Fuzzy: {self.use_fuzzy_matching} (threshold: {self.fuzzy_threshold})")

    def _create_detector(self, cfg: Dict[str, Any], agent: Agent) -> Optional[Any]:
        """Factory method for creating detectors (YOLO or Template Matching)."""
        detector_type = cfg.get('detector_type', 'template')
        
        if detector_type == 'auto':
            if YOLO_AVAILABLE:
                try:
                    yolo_config = cfg.get('yolo_config', {})
                    logger.info("Using YOLO Detector")
                    return YOLODetector(agent=agent, model_path=yolo_config.get('model_path', 'yolo11n.pt'),
                                      confidence=yolo_config.get('confidence', 0.25), device=yolo_config.get('device', 'cpu'))
                except Exception as e:
                    logger.warning(f"YOLO init failed: {e}, fallback to Template")
            detector_type = 'template'
        
        if detector_type == 'yolo':
            try:
                yolo_config = cfg.get('yolo_config', {})
                logger.info("Using YOLO Detector")
                return YOLODetector(agent=agent, model_path=yolo_config.get('model_path', 'yolo11n.pt'),
                                  confidence=yolo_config.get('confidence', 0.25), device=yolo_config.get('device', 'cpu'))
            except Exception as e:
                logger.error(f"YOLO init failed: {e}")
                return None
        
        if detector_type == 'template':
            template_config = cfg.get('template_config', {})
            logger.info("Using Template Matcher")
            return TemplateMatcher(templates_dir=template_config.get('templates_dir', self.templates_path),
                                 threshold=template_config.get('threshold', 0.85))
        
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
            if action == 'load':
                if not os.path.exists(self.resume_state_file):
                    return None
                with open(self.resume_state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                if state.get('status') == 'in_progress':
                    logger.info(f"✓ Resume: stage {state.get('current_stage')}/{state.get('total_stages')}")
                    return state
                return None
                
            elif action == 'save':
                state = {
                    'data_path': kwargs['data_path'],
                    'output_path': kwargs['output_path'],
                    'use_detector': kwargs['use_detector'],
                    'current_stage': kwargs['current_stage'],
                    'total_stages': kwargs['total_stages'],
                    'timestamp': datetime.now().isoformat(),
                    'status': 'in_progress'
                }
                with open(self.resume_state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                logger.debug(f"Resume saved: {kwargs['current_stage']}/{kwargs['total_stages']}")
                
            elif action == 'complete':
                if os.path.exists(self.resume_state_file):
                    with open(self.resume_state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    state['status'] = 'completed'
                    state['completed_at'] = datetime.now().isoformat()
                    with open(self.resume_state_file, 'w', encoding='utf-8') as f:
                        json.dump(state, f, indent=2, ensure_ascii=False)
                    logger.debug("Resume completed")
                    
            elif action == 'clear':
                if os.path.exists(self.resume_state_file):
                    os.remove(self.resume_state_file)
                    logger.debug("Resume cleared")
                    
        except Exception as e:
            logger.warning(f"Resume state {action} failed: {e}")
            return None

    def detect_roi(self, roi_name: str, screenshot: Optional[Any] = None,
                   roi_image: Optional[Any] = None) -> Dict[str, Any]:
        """Detect objects in ROI using detector (YOLO/Template)."""
        result = {'roi_name': roi_name, 'detected': False, 'detections': [], 'detection_count': 0}

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
            result['detections'] = detections
            result['detection_count'] = len(detections)
            result['detected'] = len(detections) > 0
            
            logger.info(f"✓ ROI '{roi_name}': {len(detections)} objects detected")
            return result

        except Exception as e:
            logger.error(f"✗ Detect ROI '{roi_name}': {e}")
            return result

    def scan_rois_combined(self, screenshot: Optional[Any] = None,
                          roi_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Scan multiple ROIs with both OCR and detector (optimized: crop once per ROI)."""
        try:
            screenshot = self.get_screenshot(screenshot)
            if screenshot is None:
                return {}

            if roi_names is None:
                roi_names = list(FESTIVALS_ROI_CONFIG.keys())

            results = {}
            for roi_name in roi_names:
                roi_image = self.crop_roi(screenshot, roi_name)
                if roi_image is None:
                    results[roi_name] = {
                        'roi_name': roi_name,
                        'text': '',
                        'detected': False,
                        'detections': [],
                        'detection_count': 0
                    }
                    continue

                text = ''
                if self.agent.ocr_engine is not None:
                    try:
                        ocr_result = self.agent.ocr_engine.recognize(roi_image)
                        if ocr_result:
                            text = self._clean_ocr_text(ocr_result.get('text', ''))
                    except Exception:
                        pass
                
                if self.detector is not None:
                    detection_result = self.detect_roi(roi_name, roi_image=roi_image)
                    results[roi_name] = {
                        'roi_name': roi_name,
                        'text': text,
                        'detected': detection_result['detected'],
                        'detections': detection_result['detections'],
                        'detection_count': detection_result['detection_count']
                    }
                else:
                    results[roi_name] = {
                        'roi_name': roi_name,
                        'text': text,
                        'detected': False,
                        'detections': [],
                        'detection_count': 0
                    }

            logger.info(f"Scanned {len(results)} ROIs (OCR + detector)")
            return results

        except Exception as e:
            logger.error(f"✗ Scan ROIs combined: {e}")
            return {}

    def compare_results(self, extracted_data: Dict[str, Any], expected_data: Dict[str, Any],
                       return_details: bool = True) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Compare OCR/Detector data with expected CSV data using OCRTextProcessor."""
        if not expected_data:
            return True, "No expected data", {} if return_details else None

        roi_fields = set(FESTIVALS_ROI_CONFIG.keys())
        roi_fields.discard('フェスランク')
        comparable_fields = {k: v for k, v in expected_data.items() if k in roi_fields and v}
        if not comparable_fields:
            return True, "No comparable fields", {} if return_details else None

        matches, mismatches, detailed_results = 0, [], {}

        for field, expected_value in comparable_fields.items():
            if field not in extracted_data:
                mismatches.append(f"{field}:missing")
                if return_details:
                    detailed_results[field] = {'status': 'missing', 'expected': expected_value}
                continue

            field_data = extracted_data[field]
            if isinstance(field_data, dict):
                extracted_text = field_data.get('text', '').strip()
                detected, detection_count = field_data.get('detected', False), field_data.get('detection_count', 0)
                detections = field_data.get('detections', [])
                quantity = detections[0].get('quantity', 0) if detections else 0
                has_quantity = quantity > 0
            else:
                extracted_text = str(field_data).strip()
                detected, detection_count, has_quantity, quantity = False, 0, False, 0

            validation_result = OCRTextProcessor.validate_field(field, extracted_text, expected_value)
            text_match = validation_result.status == 'match'
            
            if return_details:
                detailed_results[field] = {
                    'status': validation_result.status,
                    'extracted_text': extracted_text,
                    'extracted_value': validation_result.extracted,
                    'expected': validation_result.expected,
                    'detected': detected,
                    'detection_count': detection_count,
                    'has_quantity': has_quantity,
                    'quantity': quantity,
                    'message': validation_result.message,
                    'confidence': validation_result.confidence
                }
            
            if text_match:
                matches += 1
            else:
                mismatches.append(f"{field}:{validation_result.extracted}≠{validation_result.expected}")

        total = len(comparable_fields)
        is_ok = matches == total
        message = f"✓ {matches}/{total} matched" if is_ok else f"✗ {matches}/{total} matched ({', '.join(mismatches[:3])})"
        return is_ok, message, detailed_results if return_details else None

    def run_festival_stage(self, stage_data: Dict[str, Any], stage_idx: int,
                          use_detector: bool = False) -> bool:
        """Run automation for 1 stage with OCR verification and optional detector."""
        stage_name = stage_data.get('フェス名', 'Unknown')
        rank = stage_data.get('推奨ランク', 'Unknown')
        folder_name = f"rank_{rank}_stage_{stage_idx}"
        stage_text = stage_data.get('フェス名', '')
        rank_text = stage_data.get('フェスランク', '')
        
        max_retries = self.config.get('max_step_retries', 5)
        retry_delay = self.config.get('retry_delay', 1.0)
        
        start_time = time.time()
        stage_info = f"Rank: {rank} | Stage Text: {stage_text} | Rank Text: {rank_text}"
        self.structured_logger.stage_start(stage_idx, stage_name, stage_info)
        
        screenshot_after, screenshot_result, is_ok_before, is_ok_after = None, None, False, False
        
        try:
            # ==================== NAVIGATION STEPS ====================
            
            # Step 1: Touch Event Button (with back fallback if not found)
            def _touch_event():
                if self.touch_template("tpl_event.png"):
                    return True
                # Fallback: touch back button once and retry event
                logger.info("Event button not found, touching back button")
                if self.touch_template("tpl_back.png"):
                    sleep(0.5)
                    return self.touch_template("tpl_event.png")
                return False
            
            step1 = ExecutionStep(
                step_num=1,
                name="Touch Event Button",
                action=_touch_event,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step1.execute() != StepResult.SUCCESS:
                return False
            
            # Step 2: Snapshot Before Touch
            step2 = ExecutionStep(
                step_num=2,
                name="Snapshot Before Touch",
                action=lambda: self.snapshot_and_save(folder_name, "01_before_touch.png") is not None,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step2.execute() != StepResult.SUCCESS:
                return False
            
            # Step 3: Find and Touch Stage Name
            if not stage_text:
                self.structured_logger.error(f"Step 3: No stage text (フェス名) provided")
                return False
            
            step3 = ExecutionStep(
                step_num=3,
                name=f"Find & Touch Stage Name '{stage_text}'",
                action=lambda: self.find_and_touch_in_roi(
                    'フェス名', stage_text, 
                    threshold=self.fuzzy_threshold,
                    use_fuzzy=self.use_fuzzy_matching
                ),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step3.execute() != StepResult.SUCCESS:
                return False
            
            # Step 4: Find and Touch Rank
            if not rank_text:
                self.structured_logger.error(f"Step 4: No rank text (フェスランク) provided")
                return False
            
            step4 = ExecutionStep(
                step_num=4,
                name=f"Find & Touch Rank '{rank_text}'",
                action=lambda: self.find_and_touch_in_roi(
                    'フェスランク', rank_text,
                    threshold=self.fuzzy_threshold,
                    use_fuzzy=self.use_fuzzy_matching
                ),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step4.execute() != StepResult.SUCCESS:
                return False
            
            # Step 5: Snapshot After Touch
            def _capture_after():
                nonlocal screenshot_after
                return (screenshot_after := self.snapshot_and_save(folder_name, "02_after_touch.png")) is not None
            
            step5 = ExecutionStep(
                step_num=5, name="Snapshot After Touch",
                action=_capture_after,
                max_retries=max_retries, retry_delay=retry_delay, post_delay=0,
                cancel_checker=self.check_cancelled, logger=self.structured_logger
            )
            if step5.execute() != StepResult.SUCCESS:
                return False
            
            # ==================== PRE-BATTLE VERIFICATION ====================
            
            # Step 6: Pre-Battle Verification
            self.structured_logger.subsection_header("PRE-BATTLE VERIFICATION")
            pre_battle_rois = FESTIVAL_CONFIG.get('pre_battle_rois', 
                ['勝利点数', '推奨ランク', 'Sランクボーダー', '初回クリア報酬', 'Sランク報酬'])
            
            def _verify_pre():
                nonlocal is_ok_before, screenshot_after
                self.check_cancelled("Pre-battle verification")
                extracted = (self.scan_rois_combined(screenshot_after, pre_battle_rois) 
                           if use_detector and self.detector 
                           else self.scan_screen_roi(screenshot_after, pre_battle_rois))
                is_ok_before, msg, _ = self.compare_results(extracted, stage_data, 
                                                            return_details=bool(use_detector and self.detector))
                self.structured_logger.info(f"Verification{' (with detector)' if use_detector and self.detector else ''}: {msg}")
                if not is_ok_before:
                    screenshot_after = self.snapshot_and_save(folder_name, "02_after_touch_retry.png")
                return is_ok_before
            
            step6 = ExecutionStep(
                step_num=6, name="Pre-Battle Verification",
                action=_verify_pre,
                max_retries=max_retries, retry_delay=retry_delay, post_delay=0,
                cancel_checker=self.check_cancelled, logger=self.structured_logger
            )
            if step6.execute() != StepResult.SUCCESS:
                return False
            
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
                logger=self.structured_logger
            )
            step7.execute()
            
            # Step 8: Touch OK (Confirmation Dialog)
            step8 = ExecutionStep(
                step_num=8,
                name="Touch OK (Confirmation)",
                action=lambda: self.touch_template("tpl_ok.png"),
                max_retries=max_retries,
                retry_delay=0.3,
                optional=True,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            step8.execute()  # Optional, don't check result
            
            # Step 9: Touch All Skip Button
            step9 = ExecutionStep(
                step_num=9,
                name="Touch All Skip Button",
                action=lambda: self.touch_template("tpl_allskip.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            step9.execute()
            
            # Step 10: Touch OK (After Skip) - Optional
            step10 = ExecutionStep(
                step_num=10,
                name="Touch OK (After Skip)",
                action=lambda: self.touch_template("tpl_ok.png", optional=True),
                max_retries=max_retries,
                retry_delay=0.3,
                optional=True,
                post_delay=2.0,  # Wait for battle to complete
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            step10.execute()  # Optional, don't check result
            
            self.structured_logger.info("Waiting for battle completion...")
            
            # Step 11: Touch Result Button
            step11 = ExecutionStep(
                step_num=11,
                name="Touch Result Button",
                action=lambda: self.touch_template("tpl_result.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step11.execute() != StepResult.SUCCESS:
                return False
            
            # Step 12: Snapshot Result
            def _capture_result():
                nonlocal screenshot_result
                return (screenshot_result := self.snapshot_and_save(folder_name, "03_result.png")) is not None
            
            step12 = ExecutionStep(
                step_num=12, name="Snapshot Result",
                action=_capture_result,
                max_retries=max_retries, retry_delay=retry_delay, post_delay=0,
                cancel_checker=self.check_cancelled, logger=self.structured_logger
            )
            if step12.execute() != StepResult.SUCCESS:
                return False
            
            # ==================== POST-BATTLE VERIFICATION ====================
            
            self.structured_logger.subsection_header("POST-BATTLE VERIFICATION")
            
            # Step 13: Post-Battle Verification
            post_battle_rois = FESTIVAL_CONFIG.get('post_battle_rois',
                ['獲得ザックマネー', '獲得アイテム', '獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース'])
            
            def _verify_post():
                nonlocal is_ok_after, screenshot_result
                self.check_cancelled("Post-battle verification")
                extracted = (self.scan_rois_combined(screenshot_result, post_battle_rois)
                           if use_detector and self.detector
                           else self.scan_screen_roi(screenshot_result, post_battle_rois))
                is_ok_after, msg, _ = self.compare_results(extracted, stage_data,
                                                           return_details=bool(use_detector and self.detector))
                self.structured_logger.info(f"Verification{' (with detector)' if use_detector and self.detector else ''}: {msg}")
                if not is_ok_after:
                    screenshot_result = self.snapshot_and_save(folder_name, "03_result_retry.png")
                return is_ok_after
            
            step13 = ExecutionStep(
                step_num=13, name="Post-Battle Verification",
                action=_verify_post,
                max_retries=max_retries, retry_delay=retry_delay, post_delay=0,
                cancel_checker=self.check_cancelled, logger=self.structured_logger
            )
            if step13.execute() != StepResult.SUCCESS:
                return False
            
            # ==================== CLEANUP ====================
            
            self.structured_logger.subsection_header("CLEANUP")
            
            # Step 14: Touch OK buttons until none remain
            step14 = ExecutionStep(
                step_num=14,
                name="Touch OK (Close All Results)",
                action=lambda: self.touch_template_while_exists("tpl_ok.png", max_attempts=10, delay_between_touches=0.3) >= 0,
                max_retries=1,
                retry_delay=0.3,
                optional=True,
                post_delay=0.3,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            step14.execute()  # Optional, don't check result
            
            # ==================== FINAL RESULT ====================
            
            final = is_ok_before and is_ok_after
            duration = time.time() - start_time
            self.structured_logger.stage_end(stage_idx, final, duration)
            
            return final

        except CancellationError:
            self.structured_logger.warning(f"Stage {stage_idx} cancelled by user")
            return False
        except Exception as e:
            self.structured_logger.error(f"Stage {stage_idx} failed with exception: {e}")
            import traceback
            self.structured_logger.error(traceback.format_exc())
            return False

    def run_all_stages(self, data_path: str, output_path: Optional[str] = None,
                      use_detector: bool = False, resume: bool = True, 
                      force_new_session: bool = False) -> bool:
        """
        Run automation for all stages with incremental saving and resume support.

        Args:
            data_path: Path to CSV/JSON file with test data
            output_path: Output result path (None = auto-detect from resume or auto-generate)
            use_detector: Use detector (YOLO/Template)
            resume: Resume from existing results if available (default: True)
            force_new_session: Force start new session even if resume state exists (default: False)

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

            # Check for existing resume state
            resume_state = None
            if resume and not force_new_session:
                resume_state = self._manage_resume_state('load')
                
                # Validate resume state matches current request
                if resume_state:
                    if (resume_state.get('data_path') != data_path or 
                        resume_state.get('use_detector') != use_detector):
                        logger.warning("Resume mismatch, starting new session")
                        self._manage_resume_state('clear')
                        resume_state = None
                    else:
                        # Use output_path from resume state
                        output_path = resume_state.get('output_path')
                        logger.info(f"✓ Resuming: {output_path}")
                        self.structured_logger.info(f"RESUMING from stage {resume_state.get('current_stage', 1)}")

            # Setup output path
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detector_suffix = "_detector" if use_detector else ""
                output_path = f"{self.results_dir}/results_{timestamp}{detector_suffix}.csv"

            # Initialize ResultWriter with auto-write and resume support
            result_writer = ResultWriter(output_path, auto_write=True, resume=resume)

            # Log automation start with configuration
            mode = "Detector + OCR" if use_detector and self.detector else "OCR only"
            config_info = {
                "Mode": mode,
                "Total Stages": len(stages_data),
                "Output Path": output_path,
                "Data Source": data_path,
                "Resume Enabled": resume,
                "Max Retries": self.config.get('max_step_retries', 5)
            }
            
            if resume and result_writer.completed_test_ids:
                config_info["Already Completed"] = len(result_writer.completed_test_ids)
            
            self.structured_logger.automation_start("FESTIVAL AUTOMATION", config_info)

            # Process each stage
            success_count = 0
            failed_count = 0
            skipped_count = 0
            
            for idx, stage_data in enumerate(stages_data, 1):
                # Save resume state at start of each stage
                self._manage_resume_state('save', data_path=data_path, output_path=output_path,
                                         use_detector=use_detector, current_stage=idx, 
                                         total_stages=len(stages_data))
                
                try:
                    self.check_cancelled(f"stage {idx}")
                except CancellationError:
                    self.structured_logger.warning(f"Cancellation requested, stopping at stage {idx}")
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
                        "Resume": f"Can resume from stage {idx}"
                    }
                    self.structured_logger.automation_end("FESTIVAL AUTOMATION", False, summary)
                    self.structured_logger.info(f"💾 Resume state saved. Run again to continue from stage {idx}")
                    return False
                    
                # Prepare test case with ID
                test_case = stage_data.copy()
                test_case['test_case_id'] = idx

                # Skip if already completed (resume support)
                if resume and result_writer.is_completed(test_case):
                    stage_name = stage_data.get('フェス名', 'Unknown')
                    self.structured_logger.info(f"✓ Stage {idx} ({stage_name}) already completed, skipping...")
                    skipped_count += 1
                    continue
                
                # Log stage execution
                stage_name = stage_data.get('フェス名', 'Unknown')
                self.structured_logger.info(f"▶ Executing Stage {idx}/{len(stages_data)}: {stage_name}")

                # Run the stage
                is_ok = self.run_festival_stage(stage_data, idx, use_detector=use_detector)
                
                # Track results
                if is_ok:
                    success_count += 1
                else:
                    failed_count += 1
                
                # Save result (ResultWriter auto-writes immediately)
                result_writer.add_result(
                    test_case,
                    ResultWriter.RESULT_OK if is_ok else ResultWriter.RESULT_NG,
                    error_message=None if is_ok else "Verification failed"
                )
                
                # Progress log
                self.structured_logger.info(f"Progress: {idx}/{len(stages_data)} stages | Success: {success_count} | Failed: {failed_count}")
                
                # Delay between stages
                sleep(1.0)

            # Final save and summary
            result_writer.flush()
            result_writer.print_summary()
            
            # Mark resume completed
            self._manage_resume_state('complete')
            
            # Log completion with detailed summary
            duration = time.time() - start_time
            total_processed = len(stages_data) - skipped_count
            success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
            
            summary = {
                "Total Stages": len(stages_data),
                "Processed": total_processed,
                "Skipped": skipped_count,
                "Success": success_count,
                "Failed": failed_count,
                "Success Rate": f"{success_rate:.1f}%",
                "Total Duration": f"{duration:.2f}s",
                "Avg per Stage": f"{duration/total_processed:.2f}s" if total_processed > 0 else "N/A",
                "Results File": output_path
            }
            
            all_success = failed_count == 0 and total_processed > 0
            self.structured_logger.automation_end("FESTIVAL AUTOMATION", all_success, summary)
            
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

    def run(self, data_path: str, use_detector: bool = False, 
            output_path: Optional[str] = None, force_new_session: bool = False) -> bool:
        """
        Main entry point.

        Args:
            data_path: Path to CSV/JSON file with test data
            use_detector: Use detector (YOLO/Template)
            output_path: Output result path (None = auto-detect from resume or auto-generate)
            force_new_session: Force start new session even if resume state exists

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
            success = self.run_all_stages(
                data_path, 
                output_path=output_path,
                use_detector=use_detector, 
                resume=True,  # Always enable resume by default
                force_new_session=force_new_session
            )
            return success
        except CancellationError:
            self.structured_logger.warning("Automation cancelled")
            return False
