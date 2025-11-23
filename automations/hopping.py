"""
Hopping Automation

Standard flow for world hopping with OCR-based verification:

1. Check current world -> snapshot & OCR world name
2. Touch world map button (tpl_world_map.png)
3. Touch hop button (tpl_hop_button.png)
4. Confirm hop (tpl_confirm_hop.png)
5. Wait for loading transition (configurable duration)
6. Check new world -> snapshot & OCR world name
7. Verify hop success - Compare world names (before vs after)

OCR verification:
- World names extracted from ROI using OCR
- Text normalized using TextProcessor for accurate comparison
- Enhanced comparison detects OCR variations (>90% similarity = same world)
- Hop success confirmed when world names differ significantly
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from airtest.core.api import sleep

from core.agent import Agent
from core.base import BaseAutomation, CancellationError, ExecutionStep, StepResult
from core.config import HOPPING_ROI_CONFIG, get_hopping_config, merge_config
from core.data import ResultWriter, load_data
from core.detector import TextProcessor
from core.utils import StructuredLogger, ensure_directory, get_logger

logger = get_logger(__name__)


class HoppingAutomation(BaseAutomation):
    """Automate World Hopping with OCR verification.

    This class automates world hopping by navigating menus, confirming hops,
    and verifying successful world transitions through OCR text comparison.
    """

    def __init__(
        self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None
    ):
        base_config = get_hopping_config()
        cfg = merge_config(base_config, config) if config else base_config
        super().__init__(agent, cfg, HOPPING_ROI_CONFIG, cancel_event=cancel_event)

        self.config = cfg
        self.loading_wait = cfg["loading_wait"]
        self.cooldown_wait = cfg["cooldown_wait"]
        self.max_hops = cfg["max_hops"]
        self.retry_on_fail = cfg["retry_on_fail"]
        self.max_retries = cfg["max_retries"]

        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"hopping_{timestamp}.log")
        self.structured_logger = StructuredLogger(
            name="HoppingAutomation", log_file=log_file
        )

        logger.info("HoppingAutomation initialized")
        self.structured_logger.info(f"Log: {log_file}")

    def process_world_name(self, raw_world_name: str) -> Dict[str, Any]:
        """Process world name with OCR text cleaning using TextProcessor."""
        result = {
            "world_name": "Unknown",
            "normalized_name": "",
            "raw_name": raw_world_name,
            "confidence": 0.0,
        }

        try:
            if raw_world_name:
                # Clean OCR artifacts using TextProcessor
                cleaned = TextProcessor.clean_ocr_artifacts(raw_world_name.strip())
                # Remove extra spaces
                cleaned = " ".join(cleaned.split())

                # Normalize for comparison using TextProcessor
                normalized = TextProcessor.normalize_text(
                    cleaned, remove_spaces=True, lowercase=True
                )

                result["world_name"] = cleaned
                result["normalized_name"] = normalized
                result["confidence"] = 0.9 if cleaned and len(cleaned) > 2 else 0.5

                logger.debug(
                    f"Processed world name: '{cleaned}' (normalized: '{normalized}')"
                )

        except Exception as e:
            logger.error(f"Error processing world name: {e}")

        return result

    def verify_hop_success(
        self, before_world: str, after_world: str, use_enhanced_comparison: bool = True
    ) -> bool:
        """Check if hop was successful (world changed)."""
        if not before_world or not after_world:
            return False

        if use_enhanced_comparison:
            # Use OCRTextProcessor for better comparison
            # Process both world names
            before_processed = self.process_world_name(before_world)
            after_processed = self.process_world_name(after_world)

            # Compare normalized names
            before_norm = before_processed["normalized_name"]
            after_norm = after_processed["normalized_name"]

            # Worlds are different if normalized names don't match
            worlds_differ = before_norm != after_norm

            # Additional check: names should be sufficiently different
            # (not just minor OCR variations)
            if worlds_differ and before_norm and after_norm:
                # Calculate similarity to detect OCR variations
                similarity = sum(
                    1 for a, b in zip(before_norm, after_norm) if a == b
                ) / max(len(before_norm), len(after_norm))

                # If similarity is very high (>90%), might be same world with OCR noise
                if similarity > 0.9:
                    logger.warning(
                        f"World names very similar ({similarity:.2%}), might be OCR variation: "
                        f"'{before_world}' vs '{after_world}'"
                    )
                    worlds_differ = False

            logger.debug(
                f"Hop verification: '{before_world}' -> '{after_world}' = {worlds_differ}"
            )
            return worlds_differ
        else:
            # Legacy comparison (simple lowercase comparison)
            return before_world.strip().lower() != after_world.strip().lower()

    def run_hopping_stage(
        self, hop_data: Dict[str, Any], hop_idx: int
    ) -> Dict[str, Any]:
        """Run hopping stage."""
        folder_name = f"hopping_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = {
            "hop_idx": hop_idx,
            "world_before": "Unknown",
            "world_after": "Unknown",
            "success": False,
        }

        max_retries = self.config.get("max_step_retries", 3)
        retry_delay = self.config.get("retry_delay", 1.0)

        start_time = time.time()
        self.structured_logger.stage_start(
            hop_idx, "WORLD HOP", f"Loading wait: {self.loading_wait}s"
        )

        screenshot_before = None
        screenshot_after = None

        try:
            # Step 1: Check current world (before hop)
            def capture_before():
                nonlocal screenshot_before
                screenshot_before = self.snapshot_and_save(
                    folder_name, f"{hop_idx:02d}_before.png"
                )
                if screenshot_before is not None:
                    world_info_before = self.scan_screen_roi(
                        screenshot_before, ["world_name"]
                    )
                    raw_world_before = world_info_before.get("world_name", "Unknown")
                    processed_before = self.process_world_name(raw_world_before)
                    result["world_before"] = processed_before["world_name"]
                    result["world_before_confidence"] = processed_before["confidence"]
                    self.structured_logger.info(
                        f"Current world: {result['world_before']} (conf: {result['world_before_confidence']:.2f})"
                    )
                return screenshot_before is not None

            step1 = ExecutionStep(
                step_num=1,
                name="Check Current World",
                action=capture_before,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step1.execute() != StepResult.SUCCESS:
                result["success"] = False
                return result

            # Step 2: Touch World Map
            step2 = ExecutionStep(
                step_num=2,
                name="Touch World Map",
                action=lambda: self.touch_template("tpl_world_map.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step2.execute() != StepResult.SUCCESS:
                result["success"] = False
                return result

            # Step 3: Touch Hop Button
            step3 = ExecutionStep(
                step_num=3,
                name="Touch Hop Button",
                action=lambda: self.touch_template("tpl_hop_button.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step3.execute() != StepResult.SUCCESS:
                result["success"] = False
                return result

            # Step 4: Confirm hop
            step4 = ExecutionStep(
                step_num=4,
                name="Confirm Hop",
                action=lambda: self.touch_template("tpl_confirm_hop.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step4.execute() != StepResult.SUCCESS:
                result["success"] = False
                return result

            # Step 5: Wait for loading
            self.structured_logger.subsection_header("LOADING")
            self.structured_logger.info(
                f"Waiting {self.loading_wait}s for world transition..."
            )
            sleep(self.loading_wait)

            # Step 6: Check new world (after hop)
            self.structured_logger.subsection_header("VERIFICATION")

            def capture_after():
                nonlocal screenshot_after
                screenshot_after = self.snapshot_and_save(
                    folder_name, f"{hop_idx:02d}_after.png"
                )
                if screenshot_after is not None:
                    world_info_after = self.scan_screen_roi(
                        screenshot_after, ["world_name"]
                    )
                    raw_world_after = world_info_after.get("world_name", "Unknown")
                    processed_after = self.process_world_name(raw_world_after)
                    result["world_after"] = processed_after["world_name"]
                    result["world_after_confidence"] = processed_after["confidence"]
                    self.structured_logger.info(
                        f"New world: {result['world_after']} (conf: {result['world_after_confidence']:.2f})"
                    )
                return screenshot_after is not None

            step6 = ExecutionStep(
                step_num=6,
                name="Check New World",
                action=capture_after,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step6.execute() != StepResult.SUCCESS:
                result["success"] = False
                return result

            # Step 7: Verify hop success
            def verify_hop():
                self.check_cancelled("Hop verification")
                success = self.verify_hop_success(
                    result["world_before"],
                    result["world_after"],
                    use_enhanced_comparison=True,
                )
                result["success"] = success

                if success:
                    self.structured_logger.info(
                        f"✓ Hop successful: {result['world_before']} → {result['world_after']}"
                    )
                else:
                    self.structured_logger.warning(
                        f"✗ Hop failed: {result['world_before']} → {result['world_after']}"
                    )

                return success

            step7 = ExecutionStep(
                step_num=7,
                name="Verify Hop Success",
                action=verify_hop,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step7.execute() != StepResult.SUCCESS:
                result["success"] = False
                return result

            duration = time.time() - start_time
            self.structured_logger.stage_end(hop_idx, bool(result["success"]), duration)
            return result

        except CancellationError:
            self.structured_logger.warning(f"Hop {hop_idx} cancelled by user")
            result["success"] = False
            return result
        except Exception as e:
            self.structured_logger.error(f"Hop {hop_idx} failed with exception: {e}")
            import traceback

            self.structured_logger.error(traceback.format_exc())
            result["success"] = False
            return result

    def run_all_hops(
        self,
        data_path: Optional[str] = None,
        num_hops: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> bool:
        """
        Run all hops.

        Args:
            data_path: Path to CSV/JSON file with test data (mode 1)
            num_hops: Number of hops to run (mode 2, if no data_path)
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
                    self.structured_logger.error(
                        f"Failed to load data from {data_path}"
                    )
                    return False

                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{self.results_dir}/hopping_batch_{timestamp}.csv"

                result_writer = ResultWriter(
                    output_path,
                    formats=["csv", "json", "html"],
                    auto_write=True,
                    resume=True,
                )

                config_info = {
                    "Mode": "Batch (Data File)",
                    "Total Sessions": len(test_data),
                    "Output Path": output_path,
                    "Data Source": data_path,
                }
                self.structured_logger.automation_start(
                    "HOPPING AUTOMATION", config_info
                )

                all_success = True

                # Process each session from data
                for idx, session_data in enumerate(test_data, 1):
                    session_id = session_data.get("session_id", idx)
                    session_num_hops = int(session_data.get("num_hops", 1))

                    # Override timing configs if provided
                    if "loading_wait" in session_data:
                        self.loading_wait = float(session_data["loading_wait"])
                    if "cooldown_wait" in session_data:
                        self.cooldown_wait = float(session_data["cooldown_wait"])

                    self.structured_logger.section_header(
                        f"SESSION {idx}/{len(test_data)}: {session_num_hops} hops"
                    )

                    # Run hops for this session
                    session_start = datetime.now()
                    successful_hops = 0

                    for hop_idx in range(1, session_num_hops + 1):
                        try:
                            self.check_cancelled(f"hop {hop_idx}")
                        except CancellationError:
                            logger.info(
                                f"Cancellation requested, stopping at hop {hop_idx}"
                            )
                            break
                        hop_result = self.run_hopping_stage({}, hop_idx)
                        if hop_result["success"]:
                            successful_hops += 1
                        sleep(0.5)

                    session_end = datetime.now()
                    session_duration = (session_end - session_start).total_seconds()
                    session_success = successful_hops == session_num_hops

                    if not session_success:
                        all_success = False

                    # Add session summary
                    result_writer.add_result(
                        test_case={
                            "session_id": session_id,
                            "num_hops": session_num_hops,
                            "successful_hops": successful_hops,
                            "failed_hops": session_num_hops - successful_hops,
                            "success_rate": f"{(successful_hops/session_num_hops*100):.1f}%",
                            "duration_seconds": f"{session_duration:.1f}",
                        },
                        result=(
                            ResultWriter.RESULT_OK
                            if session_success
                            else ResultWriter.RESULT_NG
                        ),
                        error_message=(
                            None
                            if session_success
                            else f"Only {successful_hops}/{session_num_hops} hops succeeded"
                        ),
                    )

                    self.structured_logger.info(
                        f"Session {idx} completed: {successful_hops}/{session_num_hops} successful"
                    )
                    sleep(2.0)

                # Save results
                result_writer.flush()
                result_writer.print_summary()

                # Log completion
                duration = time.time() - start_time
                summary = {
                    "Total Sessions": len(test_data),
                    "Success": "All" if all_success else "Partial",
                    "Duration": f"{duration:.2f}s",
                    "Results File": output_path,
                }
                self.structured_logger.automation_end(
                    "HOPPING AUTOMATION", all_success, summary
                )

                return all_success

            # Mode 2: Direct num_hops
            elif num_hops:
                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{self.results_dir}/hopping_results_{timestamp}.csv"

                result_writer = ResultWriter(
                    output_path,
                    formats=["csv", "json", "html"],
                    auto_write=True,
                    resume=True,
                )

                config_info = {
                    "Mode": "Direct",
                    "Total Hops": num_hops,
                    "Output Path": output_path,
                }
                self.structured_logger.automation_start(
                    "HOPPING AUTOMATION", config_info
                )

                successful_hops = 0

                # Process each hop
                for idx in range(1, num_hops + 1):
                    try:
                        self.check_cancelled(f"hop {idx}")
                    except CancellationError:
                        logger.info(f"Cancellation requested, stopping at hop {idx}")
                        break
                    hop_result = self.run_hopping_stage({}, idx)

                    if hop_result["success"]:
                        successful_hops += 1

                    # Add to result writer
                    test_case = {
                        "hop_number": idx,
                        "world_before": hop_result.get("world_before", ""),
                        "world_after": hop_result.get("world_after", ""),
                        "success": hop_result.get("success", False),
                    }
                    result_writer.add_result(
                        test_case,
                        (
                            ResultWriter.RESULT_OK
                            if hop_result["success"]
                            else ResultWriter.RESULT_NG
                        ),
                        error_message=(
                            None if hop_result["success"] else "Hop verification failed"
                        ),
                    )

                    sleep(0.5)

                # Save results
                result_writer.flush()
                result_writer.print_summary()

                # Summary statistics
                duration = time.time() - start_time
                success_rate = (successful_hops / num_hops) * 100 if num_hops > 0 else 0

                summary = {
                    "Total Hops": num_hops,
                    "Successful": successful_hops,
                    "Success Rate": f"{success_rate:.1f}%",
                    "Duration": f"{duration:.2f}s",
                    "Results File": output_path,
                }
                self.structured_logger.automation_end(
                    "HOPPING AUTOMATION", True, summary
                )

                return True

            else:
                self.structured_logger.error(
                    "✗ Either 'data_path' or 'num_hops' must be provided"
                )
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

    def run(
        self, config: Optional[Dict[str, Any]] = None, data_path: Optional[str] = None
    ) -> bool:
        """
        Main entry point for Hopping automation.

        Supports 2 modes:
        1. Config mode: Pass config dict with num_hops
        2. Data mode: Pass data_path to CSV/JSON file

        Args:
            config: Configuration dict (mode 1)
            data_path: Path to CSV/JSON data file (mode 2)

        Returns:
            bool: True if successful

        Example usage:
            # Mode 1: Direct config
            hopping.run(config={'num_hops': 5})

            # Mode 2: Load from file
            hopping.run(data_path='./data/hopping_tests.csv')
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
                success = self.run_all_hops(data_path=data_path)
            # Mode 1: Direct config
            elif config:
                num_hops = config.get("num_hops", 1)
                success = self.run_all_hops(num_hops=num_hops)
            else:
                self.structured_logger.error(
                    "✗ Either 'config' or 'data_path' must be provided"
                )
                return False

            return success
        except CancellationError:
            self.structured_logger.warning("Automation cancelled")
            return False
