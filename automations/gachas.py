"""
Gacha Automation

Gacha flow:
1. touch(Template("tpl_gacha.png"))
2. touch(Template("tpl_single_pull.png")) or touch(Template("tpl_multi_pull.png"))
3. snapshot -> save to folder (gacha_01_before_pull.png)
4. touch(Template("tpl_confirm.png"))
5. touch(Template("tpl_skip.png")) (if animation exists)
6. snapshot -> save to folder (gacha_02_after_pull.png)
7. ROI scan -> check gacha results
8. touch(Template("tpl_ok.png")) to close
9. Repeat according to pull count
"""

from typing import Dict, Optional, Any
from datetime import datetime
from airtest.core.api import sleep
import time
import os

from core.base import BaseAutomation, CancellationError, ExecutionStep, StepResult
from core.agent import Agent
from core.utils import get_logger, StructuredLogger, ensure_directory
from core.data import ResultWriter, load_data
from core.config import (
    GACHA_ROI_CONFIG, get_gacha_config, merge_config
)
from core.detector import OCRTextProcessor

logger = get_logger(__name__)


class GachaAutomation(BaseAutomation):
    """Automate Gacha pulls with OCR result verification."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None):
        base_config = get_gacha_config()
        cfg = merge_config(base_config, config) if config else base_config
        super().__init__(agent, cfg, GACHA_ROI_CONFIG, cancel_event=cancel_event)

        self.config = cfg
        self.wait_after_pull = cfg['wait_after_pull']
        self.max_pulls = cfg['max_pulls']
        self.pull_type = cfg['pull_type']
        
        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"gacha_{timestamp}.log")
        self.structured_logger = StructuredLogger(name="GachaAutomation", log_file=log_file)

        logger.info("GachaAutomation initialized")
        self.structured_logger.info(f"Log: {log_file}")

    def process_gacha_result(self, scan_results: Dict[str, str]) -> Dict[str, Any]:
        """Process gacha scan results with OCR text processing."""
        result = {'rarity': 'Unknown', 'character': 'Unknown', 'rarity_confidence': 0.0,
                 'character_confidence': 0.0, 'raw_rarity': '', 'raw_character': ''}
        
        try:
            # Process rarity (normalize text)
            raw_rarity = scan_results.get('rarity', '')
            if raw_rarity:
                result['raw_rarity'] = raw_rarity
                # Normalize rarity text
                normalized_rarity = OCRTextProcessor.normalize_text_for_comparison(raw_rarity)
                
                # Match against known rarities
                known_rarities = ['ssr', 'sr', 'r', 'n', '★★★★★', '★★★★', '★★★', '★★', '★']
                best_match = None
                best_similarity = 0.0
                
                for known in known_rarities:
                    normalized_known = OCRTextProcessor.normalize_text_for_comparison(known)
                    if normalized_rarity == normalized_known:
                        best_match = known.upper()
                        best_similarity = 1.0
                        break
                    elif normalized_known in normalized_rarity or normalized_rarity in normalized_known:
                        # Partial match
                        similarity = min(len(normalized_known), len(normalized_rarity)) / max(len(normalized_known), len(normalized_rarity))
                        if similarity > best_similarity:
                            best_match = known.upper()
                            best_similarity = similarity
                
                if best_match:
                    result['rarity'] = best_match
                    result['rarity_confidence'] = best_similarity
                else:
                    result['rarity'] = raw_rarity.upper()
                    result['rarity_confidence'] = 0.5
            
            # Process character name (clean OCR artifacts)
            raw_character = scan_results.get('character', '')
            if raw_character:
                result['raw_character'] = raw_character
                # Clean common OCR artifacts
                cleaned_character = raw_character.strip()
                # Remove extra spaces
                cleaned_character = ' '.join(cleaned_character.split())
                result['character'] = cleaned_character
                result['character_confidence'] = 0.8 if cleaned_character else 0.0
            
            logger.debug(f"Processed gacha result: {result['rarity']} - {result['character']} "
                        f"(rarity conf: {result['rarity_confidence']:.2f}, char conf: {result['character_confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing gacha result: {e}")
        
        return result

    def validate_gacha_result(self, processed_result: Dict[str, Any],
                             expected_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate gacha result against expected data (if provided)."""
        validation = {'status': 'pass', 'rarity_match': True, 'character_match': True,
                     'message': 'No validation data provided'}
        
        if not expected_data:
            return validation
        
        try:
            # Validate rarity
            if 'expected_rarity' in expected_data:
                expected_rarity = str(expected_data['expected_rarity']).upper()
                actual_rarity = processed_result['rarity'].upper()
                
                rarity_match = OCRTextProcessor.compare_with_template(
                    actual_rarity, expected_rarity, threshold=0.7
                )
                validation['rarity_match'] = rarity_match
                
                if not rarity_match:
                    validation['status'] = 'fail'
                    validation['message'] = f"Rarity mismatch: got {actual_rarity}, expected {expected_rarity}"
            
            # Validate character
            if 'expected_character' in expected_data:
                expected_char = str(expected_data['expected_character'])
                actual_char = processed_result['character']
                
                char_match = OCRTextProcessor.compare_with_template(
                    actual_char, expected_char, threshold=0.6
                )
                validation['character_match'] = char_match
                
                if not char_match:
                    validation['status'] = 'fail'
                    if validation['message'] == 'No validation data provided':
                        validation['message'] = f"Character mismatch: got {actual_char}, expected {expected_char}"
                    else:
                        validation['message'] += f"; Character mismatch: got {actual_char}, expected {expected_char}"
            
            if validation['status'] == 'pass':
                validation['message'] = 'All validations passed'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['message'] = f"Validation error: {str(e)}"
        
        return validation

    def run_gacha_stage(self, pull_data: Dict[str, Any], pull_idx: int,
                      pull_type: str = "single") -> Dict[str, Any]:
        """Run gacha stage."""
        folder_name = f"gacha_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = {
            'pull_idx': pull_idx,
            'pull_type': pull_type,
            'rarity': 'Unknown',
            'character': 'Unknown',
            'success': False
        }
        
        max_retries = self.config.get('max_step_retries', 3)
        retry_delay = self.config.get('retry_delay', 1.0)
        
        start_time = time.time()
        pull_info = f"Type: {pull_type.upper()}"
        self.structured_logger.stage_start(pull_idx, f"{pull_type.upper()} PULL", pull_info)
        
        screenshot_result = None

        try:
            # Step 1: Touch Gacha
            step1 = ExecutionStep(
                step_num=1,
                name="Touch Gacha Button",
                action=lambda: self.touch_template("tpl_gacha.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step1.execute() != StepResult.SUCCESS:
                result['success'] = False
                return result

            # Step 2: Choose pull type
            template_name = "tpl_single_pull.png" if pull_type == "single" else "tpl_multi_pull.png"
            step2 = ExecutionStep(
                step_num=2,
                name=f"Choose {pull_type} pull",
                action=lambda: self.touch_template(template_name),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step2.execute() != StepResult.SUCCESS:
                result['success'] = False
                return result

            # Step 3: Snapshot before pull
            step3 = ExecutionStep(
                step_num=3,
                name="Snapshot Before Pull",
                action=lambda: self.snapshot_and_save(folder_name, f"{pull_idx:02d}_before.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step3.execute() != StepResult.SUCCESS:
                result['success'] = False
                return result

            # Step 4: Confirm pull
            step4 = ExecutionStep(
                step_num=4,
                name="Confirm Pull",
                action=lambda: self.touch_template("tpl_confirm.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step4.execute() != StepResult.SUCCESS:
                result['success'] = False
                return result

            # Step 5: Skip animation (optional)
            step5 = ExecutionStep(
                step_num=5,
                name="Skip Animation",
                action=lambda: self.touch_template("tpl_skip.png", optional=True),
                max_retries=1,
                retry_delay=0.3,
                optional=True,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            step5.execute()  # Optional, don't check result

            # Step 6: Snapshot result
            def capture_result():
                nonlocal screenshot_result
                screenshot_result = self.snapshot_and_save(folder_name, f"{pull_idx:02d}_result.png")
                return screenshot_result is not None
            
            step6 = ExecutionStep(
                step_num=6,
                name="Snapshot Result",
                action=capture_result,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step6.execute() != StepResult.SUCCESS:
                result['success'] = False
                return result

            # Step 7: Scan and process result
            self.structured_logger.subsection_header("RESULT PROCESSING")
            
            def scan_and_process():
                self.check_cancelled("Result scanning")
                scan_results = self.scan_screen_roi(screenshot_result)
                processed = self.process_gacha_result(scan_results)
                validation = self.validate_gacha_result(processed, pull_data)
                
                result.update({
                    'rarity': processed['rarity'],
                    'character': processed['character'],
                    'rarity_confidence': processed['rarity_confidence'],
                    'character_confidence': processed['character_confidence'],
                    'raw_rarity': processed['raw_rarity'],
                    'raw_character': processed['raw_character'],
                    'validation': validation,
                    'scan_data': scan_results,
                    'success': True
                })
                
                self.structured_logger.info(f"Result: {processed['rarity']} - {processed['character']} "
                                          f"(conf: {processed['rarity_confidence']:.2f})")
                
                if validation['status'] != 'pass' and validation['message'] != 'No validation data provided':
                    self.structured_logger.warning(f"Validation: {validation['message']}")
                
                return True
            
            step7 = ExecutionStep(
                step_num=7,
                name="Scan & Process Result",
                action=scan_and_process,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            if step7.execute() != StepResult.SUCCESS:
                result['success'] = False
                return result

            # Step 8: Close result
            self.structured_logger.subsection_header("CLEANUP")
            step8 = ExecutionStep(
                step_num=8,
                name="Close Result",
                action=lambda: self.touch_template("tpl_ok.png", optional=True),
                max_retries=1,
                retry_delay=0.3,
                optional=True,
                post_delay=0.3,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger
            )
            step8.execute()  # Optional, don't check result

            duration = time.time() - start_time
            self.structured_logger.stage_end(pull_idx, result['success'], duration)
            return result

        except CancellationError:
            self.structured_logger.warning(f"Pull {pull_idx} cancelled by user")
            result['success'] = False
            return result
        except Exception as e:
            self.structured_logger.error(f"Pull {pull_idx} failed with exception: {e}")
            import traceback
            self.structured_logger.error(traceback.format_exc())
            result['success'] = False
            return result

    def run_all_pulls(self, data_path: Optional[str] = None, num_pulls: Optional[int] = None,
                      pull_type: str = "single", output_path: Optional[str] = None) -> bool:
        """
        Run all pulls.

        Args:
            data_path: Path to CSV/JSON file with test data (mode 1)
            num_pulls: Number of pulls to run (mode 2, if no data_path)
            pull_type: Pull type ('single' or 'multi', mode 2 only)
            output_path: Output result path (None = auto-generate)

        Returns:
            bool: True if successful
        """
        result_writer = None
        start_time = time.time()
        
        try:
            # Mode 1: Load from data file
            if data_path:
                test_data = load_data(data_path)
                if not test_data:
                    self.structured_logger.error(f"Failed to load data from {data_path}")
                    return False

                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{self.results_dir}/gacha_batch_{timestamp}.csv"

                result_writer = ResultWriter(output_path, auto_write=True, resume=True)
                
                config_info = {
                    "Mode": "Batch (Data File)",
                    "Total Sessions": len(test_data),
                    "Output Path": output_path,
                    "Data Source": data_path
                }
                self.structured_logger.automation_start("GACHA AUTOMATION", config_info)

                all_success = True

                # Process each session from data
                for idx, session_data in enumerate(test_data, 1):
                    session_id = session_data.get('session_id', idx)
                    session_num_pulls = int(session_data.get('num_pulls', 1))
                    session_pull_type = session_data.get('pull_type', 'single')

                    # Override timing configs if provided
                    if 'wait_after_pull' in session_data:
                        self.wait_after_pull = float(session_data['wait_after_pull'])
                    if 'wait_after_touch' in session_data:
                        self.wait_after_touch = float(session_data['wait_after_touch'])

                    self.structured_logger.section_header(f"SESSION {idx}/{len(test_data)}: {session_num_pulls} {session_pull_type} pull(s)")

                    # Run pulls for this session
                    session_start = datetime.now()
                    successful_pulls = 0
                    rarity_counts = {}

                    for pull_idx in range(1, session_num_pulls + 1):
                        try:
                            self.check_cancelled(f"pull {pull_idx}")
                        except CancellationError:
                            logger.info(f"Cancellation requested, stopping at pull {pull_idx}")
                            break
                        pull_result = self.run_gacha_stage({}, pull_idx, session_pull_type)
                        if pull_result['success']:
                            successful_pulls += 1
                            rarity = pull_result.get('rarity', 'Unknown')
                            rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1
                        sleep(0.5)

                    session_end = datetime.now()
                    session_duration = (session_end - session_start).total_seconds()
                    session_success = successful_pulls == session_num_pulls

                    if not session_success:
                        all_success = False

                    # Add session summary
                    result_writer.add_result(
                        test_case={
                            'session_id': session_id,
                            'pull_type': session_pull_type,
                            'num_pulls': session_num_pulls,
                            'successful_pulls': successful_pulls,
                            'failed_pulls': session_num_pulls - successful_pulls,
                            'success_rate': f"{(successful_pulls/session_num_pulls*100):.1f}%",
                            'duration_seconds': f"{session_duration:.1f}",
                            'rarity_distribution': str(rarity_counts),
                        },
                        result=ResultWriter.RESULT_OK if session_success else ResultWriter.RESULT_NG,
                        error_message=None if session_success else f"Only {successful_pulls}/{session_num_pulls} pulls succeeded"
                    )

                    self.structured_logger.info(f"Session {idx} completed: {successful_pulls}/{session_num_pulls} successful | Rarity: {rarity_counts}")
                    sleep(0.5)

                # Save results
                result_writer.flush()
                result_writer.print_summary()
                
                # Log completion
                duration = time.time() - start_time
                summary = {
                    "Total Sessions": len(test_data),
                    "Success": "All" if all_success else "Partial",
                    "Duration": f"{duration:.2f}s",
                    "Results File": output_path
                }
                self.structured_logger.automation_end("GACHA AUTOMATION", all_success, summary)
                
                return all_success

            # Mode 2: Direct num_pulls
            elif num_pulls:
                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{self.results_dir}/gacha_results_{timestamp}.csv"

                result_writer = ResultWriter(output_path, auto_write=True, resume=True)
                
                config_info = {
                    "Mode": "Direct",
                    "Total Pulls": num_pulls,
                    "Pull Type": pull_type,
                    "Output Path": output_path
                }
                self.structured_logger.automation_start("GACHA AUTOMATION", config_info)

                all_results = []

                # Process each pull
                for idx in range(1, num_pulls + 1):
                    try:
                        self.check_cancelled(f"pull {idx}")
                    except CancellationError:
                        logger.info(f"Cancellation requested, stopping at pull {idx}")
                        break
                    pull_result = self.run_gacha_stage({}, idx, pull_type)
                    all_results.append(pull_result)

                    # Add to result writer
                    test_case = {
                        'pull_number': idx,
                        'pull_type': pull_type,
                        'rarity': pull_result.get('rarity', ''),
                        'character': pull_result.get('character', ''),
                    }
                    result_writer.add_result(test_case,
                                           ResultWriter.RESULT_OK if pull_result['success'] else ResultWriter.RESULT_NG,
                                           error_message=None if pull_result['success'] else "Pull failed")

                    sleep(0.5)

                # Save results
                result_writer.flush()
                result_writer.print_summary()

                # Summary statistics
                successful_pulls = [r for r in all_results if r['success']]
                rarity_counts = {}
                for r in successful_pulls:
                    rarity = r.get('rarity', 'Unknown')
                    rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1

                duration = time.time() - start_time
                success_rate = (len(successful_pulls) / num_pulls * 100) if num_pulls > 0 else 0
                
                summary = {
                    "Total Pulls": num_pulls,
                    "Successful": len(successful_pulls),
                    "Success Rate": f"{success_rate:.1f}%",
                    "Rarity Distribution": str(rarity_counts),
                    "Duration": f"{duration:.2f}s",
                    "Results File": output_path
                }
                self.structured_logger.automation_end("GACHA AUTOMATION", True, summary)

                return True

            else:
                self.structured_logger.error("✗ Either 'data_path' or 'num_pulls' must be provided")
                return False

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

    def run(self, config: Optional[Dict[str, Any]] = None, data_path: Optional[str] = None) -> bool:
        """
        Main entry point for Gacha automation.

        Supports 2 modes:
        1. Config mode: Pass config dict with num_pulls, pull_type
        2. Data mode: Pass data_path to CSV/JSON file

        Args:
            config: Configuration dict (mode 1)
            data_path: Path to CSV/JSON data file (mode 2)

        Returns:
            bool: True if successful

        Example usage:
            # Mode 1: Direct config
            gacha.run(config={'num_pulls': 10, 'pull_type': 'single'})

            # Mode 2: Load from file
            gacha.run(data_path='./data/gacha_tests.csv')
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
            # Mode 2: Load from data file
            if data_path:
                success = self.run_all_pulls(data_path=data_path)
            # Mode 1: Direct config
            elif config:
                num_pulls = config.get('num_pulls', 1)
                pull_type = config.get('pull_type', 'single')
                success = self.run_all_pulls(num_pulls=num_pulls, pull_type=pull_type)
            else:
                self.structured_logger.error("✗ Either 'config' or 'data_path' must be provided")
                return False

            return success
        except CancellationError:
            self.structured_logger.warning("Automation cancelled")
            return False
