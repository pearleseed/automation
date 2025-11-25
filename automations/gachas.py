"""
Gacha Automation

Standard flow for gacha banner pulls with template-based verification:

1. Find & touch gacha banner (scroll if needed)
2. Choose pull type (single/multi pull button)
3. Snapshot before pull -> save (banner_01_before.png)
4. Confirm pull (tpl_ok.png)
5. Skip animation (tpl_skip.png) - optional
6. Snapshot after pull -> save (banner_01_after.png)
7. Result verification - Check SSR/SR + Swimsuit templates
   - If both found -> Special snapshot saved

Template matching:
- Banners scrolled until found (max 10 attempts)
- Rarity templates (tpl_ssr.png, tpl_sr.png) matched on result screen
- Swimsuit character templates matched in banner folder
- Special snapshots saved when both rarity and character match
"""

import glob
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from airtest.core.api import Template, exists, sleep

from core.agent import Agent
from core.base import BaseAutomation, CancellationError, ExecutionStep, StepResult
from core.config import GACHA_ROI_CONFIG, get_gacha_config, merge_config
from core.data import ResultWriter
from core.utils import StructuredLogger, ensure_directory, get_logger

logger = get_logger(__name__)


def match_template_in_image(
    image: np.ndarray, template: np.ndarray, threshold: float = 0.8
) -> Tuple[bool, float]:
    """Match template in image using cv2.matchTemplate.

    Args:
        image: Source image (BGR or grayscale).
        template: Template image (BGR or grayscale).
        threshold: Match threshold (0.0-1.0).

    Returns:
        Tuple of (found, confidence).
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Check template size
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        return False, 0.0

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    return max_val >= threshold, max_val


class GachaAutomation(BaseAutomation):
    """Automate Gacha pulls with template matching result verification.

    This class automates gacha banner pulls by finding banners, executing pulls,
    and verifying results through template matching for rarity and character detection.
    """

    def __init__(
        self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None
    ):
        base_config = get_gacha_config()
        cfg = merge_config(base_config, config) if config else base_config
        super().__init__(agent, cfg, GACHA_ROI_CONFIG, cancel_event=cancel_event)

        self.config = cfg
        self.wait_after_pull = cfg.get("wait_after_pull", 2.0)
        self.max_scroll_attempts = cfg.get("max_scroll_attempts", 10)

        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"gacha_{timestamp}.log")
        self.structured_logger = StructuredLogger(
            name="GachaAutomation", log_file=log_file
        )

        logger.info("GachaAutomation initialized")
        self.structured_logger.info(
            f"Log: {log_file} | Wait after pull: {self.wait_after_pull}s | Max scroll attempts: {self.max_scroll_attempts}"
        )

    def find_banner(self, banner_path: str, roi_name: str = "banner") -> bool:
        """Find banner in ROI by scrolling if needed.

        Args:
            banner_path: Path to banner template image
            roi_name: ROI name from GACHA_ROI_CONFIG (default: "banner")

        Returns:
            bool: True if banner found in ROI, False otherwise
        """
        if not os.path.exists(banner_path):
            logger.error(f"Banner template not found: {banner_path}")
            return False

        banner_template = cv2.imread(banner_path)
        if banner_template is None:
            logger.error(f"Failed to load banner template: {banner_path}")
            return False

        threshold = 0.8

        # Check current screen
        screenshot = self.get_screenshot()
        if screenshot is not None:
            roi_image = self.crop_roi(screenshot, roi_name)
            if roi_image is not None:
                found, confidence = match_template_in_image(roi_image, banner_template, threshold)
                if found:
                    logger.info(f"Banner found in ROI (confidence: {confidence:.2f})")
                    return True

        # Scroll and search
        for attempt in range(1, self.max_scroll_attempts + 1):
            self.check_cancelled("find_banner")
            if not self.touch_template("tpl_button_down.png", optional=True):
                break
            sleep(0.5)

            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            roi_image = self.crop_roi(screenshot, roi_name)
            if roi_image is None:
                continue

            found, confidence = match_template_in_image(roi_image, banner_template, threshold)
            if found:
                logger.info(f"Banner found after {attempt} scroll(s) (confidence: {confidence:.2f})")
                return True

        logger.error(f"Banner not found in ROI after {self.max_scroll_attempts} scrolls")
        return False

    def check_result(
        self, rarity: str, banner_folder: str, banner_path: str = ""
    ) -> Dict[str, Any]:
        """Check gacha result using template matching."""
        result = {"has_rarity": False, "has_swimsuit": False, "should_snapshot": False}

        self.structured_logger.subsection_header("RESULT VERIFICATION")

        # Check rarity template
        rarity_tpl = f"tpl_{rarity.lower()}.png"
        rarity_path = os.path.join(self.templates_path, rarity_tpl)
        if os.path.exists(rarity_path) and exists(Template(rarity_path)):
            result["has_rarity"] = True
            logger.info(f"✓ Found {rarity.upper()}")
            self.structured_logger.info(f"✓ Rarity check: Found {rarity.upper()}")
        else:
            self.structured_logger.info(f"✗ Rarity check: {rarity.upper()} not found")

        # Check swimsuit templates in banner folder (exclude banner file)
        banner_basename = os.path.basename(banner_path).lower() if banner_path else ""
        swimsuit_found = None
        for pattern in ["*.png", "*.jpg", "*.jpeg"]:
            for file_path in glob.glob(os.path.join(banner_folder, pattern)):
                filename = os.path.basename(file_path).lower()
                # Skip banner file (exact match)
                if filename == banner_basename:
                    continue
                if exists(Template(file_path)):
                    result["has_swimsuit"] = True
                    swimsuit_found = os.path.basename(file_path)
                    logger.info(f"✓ Found swimsuit: {swimsuit_found}")
                    self.structured_logger.info(
                        f"✓ Swimsuit check: Found {swimsuit_found}"
                    )
                    break
            if result["has_swimsuit"]:
                break

        if not result["has_swimsuit"]:
            self.structured_logger.info("✗ Swimsuit check: No swimsuit template found")

        result["should_snapshot"] = result["has_rarity"] and result["has_swimsuit"]
        if result["should_snapshot"]:
            logger.info("✓✓✓ SPECIAL MATCH! Both rarity and swimsuit found!")
            self.structured_logger.info(
                "✓✓✓ SPECIAL MATCH! Both rarity and swimsuit found!"
            )

        return result

    def run_pull(
        self, gacha: Dict[str, Any], pull_idx: int, folder: str
    ) -> Dict[str, Any]:
        """Run single gacha pull with verification.

        Args:
            gacha: Dictionary containing gacha configuration (name, banner_path, pull_type, rarity, etc.).
            pull_idx: Pull index number for logging and tracking.
            folder: Folder name for saving screenshots.

        Returns:
            Dict containing:
                - success (bool): True if pull completed successfully
                - result_details (dict): Verification results (rarity, swimsuit, etc.)
        """
        pull_type = gacha.get("pull_type", "single")
        gacha_name = gacha.get("name", "Unknown")
        start_time = time.time()

        pull_info = (
            f"Type: {pull_type.upper()} | Rarity: {gacha.get('rarity', 'ssr').upper()}"
        )
        self.structured_logger.stage_start(pull_idx, gacha_name, pull_info)

        max_retries = self.config.get("max_step_retries", 3)
        retry_delay = self.config.get("retry_delay", 1.0)
        screenshot = None
        result_details = {}

        try:
            # ==================== BANNER SELECTION ====================
            self.structured_logger.subsection_header("BANNER SELECTION")

            # Step 1: Find & Touch Banner
            banner_path = gacha["banner_path"]
            banner_name = os.path.basename(banner_path)
            self.structured_logger.info(f"Looking for banner: {banner_name}")

            step1 = ExecutionStep(
                step_num=1,
                name="Find & Touch Banner",
                action=lambda: self.find_banner(banner_path)
                and self.touch_template(banner_path),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=1.0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step1.execute() != StepResult.SUCCESS:
                self.structured_logger.stage_end(
                    pull_idx, False, time.time() - start_time
                )
                return {"success": False, "result_details": {}}

            # ==================== PULL EXECUTION ====================
            self.structured_logger.subsection_header("PULL EXECUTION")

            # Step 2: Choose Pull Type
            pull_tpl = (
                "tpl_single_pull.png" if pull_type == "single" else "tpl_multi_pull.png"
            )
            self.structured_logger.info(f"Selecting {pull_type} pull button")

            step2 = ExecutionStep(
                step_num=2,
                name=f"Choose {pull_type} pull",
                action=lambda: self.touch_template(pull_tpl),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step2.execute() != StepResult.SUCCESS:
                self.structured_logger.stage_end(
                    pull_idx, False, time.time() - start_time
                )
                return {"success": False, "result_details": {}}

            # Step 3: Snapshot Before
            step3 = ExecutionStep(
                step_num=3,
                name="Snapshot Before",
                action=lambda: self.snapshot_and_save(
                    folder, f"{pull_idx:02d}_before.png"
                )
                is not None,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step3.execute() != StepResult.SUCCESS:
                self.structured_logger.stage_end(
                    pull_idx, False, time.time() - start_time
                )
                return {"success": False, "result_details": {}}

            # Step 4: Confirm Pull
            self.structured_logger.info("Confirming pull...")
            step4 = ExecutionStep(
                step_num=4,
                name="Confirm Pull",
                action=lambda: self.touch_template("tpl_ok.png"),
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=self.wait_after_pull,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step4.execute() != StepResult.SUCCESS:
                self.structured_logger.stage_end(
                    pull_idx, False, time.time() - start_time
                )
                return {"success": False, "result_details": {}}

            # Step 5: Skip Animation
            self.structured_logger.info("Attempting to skip animation...")
            step5 = ExecutionStep(
                step_num=5,
                name="Skip Animation",
                action=lambda: self.touch_template("tpl_skip.png", optional=True),
                max_retries=1,
                retry_delay=0.5,
                optional=True,
                post_delay=1.0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step5.execute()  # Optional, don't check result

            # Step 6: Snapshot After
            def _capture_after():
                nonlocal screenshot
                return (
                    screenshot := self.snapshot_and_save(
                        folder, f"{pull_idx:02d}_after.png"
                    )
                ) is not None

            step6 = ExecutionStep(
                step_num=6,
                name="Snapshot After",
                action=_capture_after,
                max_retries=max_retries,
                retry_delay=retry_delay,
                post_delay=0.5,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            if step6.execute() != StepResult.SUCCESS:
                self.structured_logger.stage_end(
                    pull_idx, False, time.time() - start_time
                )
                return {"success": False, "result_details": {}}

            # ==================== RESULT VERIFICATION ====================

            # Step 7: Check Result
            banner_folder = gacha.get("banner_folder", "")
            banner_path_for_check = gacha.get("banner_path", "")

            def _verify_result():
                nonlocal result_details
                self.check_cancelled("Result verification")

                if banner_folder and os.path.exists(banner_folder):
                    result_details = self.check_result(
                        gacha.get("rarity", "ssr"), banner_folder, banner_path_for_check
                    )

                    # Log detailed verification results
                    if result_details.get("has_rarity"):
                        self.structured_logger.info(
                            f"  ✓ Rarity: {gacha.get('rarity', 'ssr').upper()} found"
                        )
                    else:
                        self.structured_logger.info(
                            f"  ✗ Rarity: {gacha.get('rarity', 'ssr').upper()} not found"
                        )

                    if result_details.get("has_swimsuit"):
                        self.structured_logger.info("  ✓ Swimsuit: Found")
                    else:
                        self.structured_logger.info("  ✗ Swimsuit: Not found")

                    if result_details.get("should_snapshot"):
                        self.structured_logger.info("Saving special snapshot...")
                        self.snapshot_and_save(folder, f"{pull_idx:02d}_SPECIAL.png")
                else:
                    self.structured_logger.info("No banner folder for verification")

                # Always continue regardless of verification result
                return True

            step7 = ExecutionStep(
                step_num=7,
                name="Result Verification",
                action=_verify_result,
                max_retries=1,  # No retries, just verify once
                retry_delay=0,
                post_delay=0,
                cancel_checker=self.check_cancelled,
                logger=self.structured_logger,
            )
            step7.execute()  # Always continue regardless of result

            # ==================== FINAL RESULT ====================

            duration = time.time() - start_time
            self.structured_logger.stage_end(pull_idx, True, duration)

            return {"success": True, "result_details": result_details}

        except CancellationError:
            self.structured_logger.warning(f"Pull {pull_idx} cancelled by user")
            self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
            return {"success": False, "result_details": {}}
        except Exception as e:
            logger.error(f"Pull {pull_idx} failed: {e}")
            self.structured_logger.error(f"Pull {pull_idx} failed with exception: {e}")
            import traceback

            self.structured_logger.error(traceback.format_exc())
            self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
            return {"success": False, "result_details": {}}

    def run(self, gachas_config: List[Dict[str, Any]]) -> bool:
        """Run gacha automation for multiple gacha banners.

        Args:
            gachas_config: List of gacha configurations, each containing banner path,
                          pull type (single/multi), number of pulls, and rarity.

        Returns:
            bool: True if all gacha pulls completed successfully, False otherwise.
        """
        if not self.agent.is_device_connected():
            logger.error("Device not connected")
            self.structured_logger.error("Device not connected")
            return False

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.results_dir}/gacha_{timestamp}.csv"
        result_writer = ResultWriter(
            output_path, formats=["csv", "json", "html"], auto_write=True
        )

        # Calculate total pulls
        total_pulls = sum(g.get("num_pulls", 0) for g in gachas_config)

        # Log automation start with configuration
        config_info = {
            "Total Gachas": len(gachas_config),
            "Total Pulls": total_pulls,
            "Output Path": output_path,
            "Wait After Pull": f"{self.wait_after_pull}s",
            "Max Scroll Attempts": self.max_scroll_attempts,
            "Max Retries": self.config.get("max_step_retries", 3),
        }

        self.structured_logger.automation_start("GACHA AUTOMATION", config_info)

        all_success = True
        total_success = 0
        total_failed = 0

        for idx, gacha in enumerate(gachas_config, 1):
            try:
                self.check_cancelled(f"gacha {idx}")
            except CancellationError:
                self.structured_logger.warning(
                    f"Cancellation requested, stopping at gacha {idx}"
                )
                # Ensure results are saved before exiting
                result_writer.flush()

                # Log summary
                duration = time.time() - start_time
                summary = {
                    "Total Processed": idx - 1,
                    "Total Success": total_success,
                    "Total Failed": total_failed,
                    "Duration": f"{duration:.2f}s",
                    "Status": "CANCELLED",
                }
                self.structured_logger.automation_end(
                    "GACHA AUTOMATION", False, summary
                )
                return False

            gacha_name = gacha.get("name", f"Gacha {idx}")
            pull_type = gacha.get("pull_type", "single")
            num_pulls = gacha.get("num_pulls", 0)

            folder = f"{idx:02d}_{gacha_name}_{timestamp}"
            self.structured_logger.section_header(
                f"GACHA {idx}/{len(gachas_config)}: {gacha_name}"
            )
            self.structured_logger.info(
                f"Configuration: {pull_type.upper()} pull x{num_pulls} | Rarity: {gacha.get('rarity', 'ssr').upper()}"
            )

            success_count = 0
            failed_count = 0

            for pull_idx in range(1, num_pulls + 1):
                try:
                    self.check_cancelled(f"gacha {idx} pull {pull_idx}")
                except CancellationError:
                    self.structured_logger.warning(
                        f"Cancellation requested, stopping at pull {pull_idx}"
                    )
                    all_success = False
                    break

                # Run the pull
                pull_result = self.run_pull(gacha, pull_idx, folder)

                # Extract results
                is_ok = pull_result.get("success", False)
                result_details = pull_result.get("result_details", {})

                # Track results
                if is_ok:
                    success_count += 1
                    total_success += 1
                else:
                    failed_count += 1
                    total_failed += 1
                    all_success = False

                sleep(0.5)

            if success_count != num_pulls:
                all_success = False

            # Log gacha completion summary
            self.structured_logger.info(
                f"Gacha {idx} completed: {success_count}/{num_pulls} successful | Failed: {failed_count}"
            )
            sleep(1.0)

        result_writer.flush()

        # Log completion with detailed summary
        duration = time.time() - start_time
        success_rate = (total_success / total_pulls * 100) if total_pulls > 0 else 0
        avg_per_pull = (duration / total_pulls) if total_pulls > 0 else 0

        summary = {
            "Total Gachas": len(gachas_config),
            "Total Pulls": total_pulls,
            "Success": total_success,
            "Failed": total_failed,
            "Success Rate": f"{success_rate:.1f}%",
            "Total Duration": f"{duration:.2f}s",
            "Avg per Pull": f"{avg_per_pull:.2f}s" if total_pulls > 0 else "N/A",
            "Results File": output_path,
        }

        self.structured_logger.automation_end("GACHA AUTOMATION", all_success, summary)

        return all_success
