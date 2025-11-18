"""
Gacha Automation

Gacha flow:
1. Find and touch gacha banner (scroll if needed)
2. Touch single/multi pull button
3. Snapshot before pull
4. Confirm pull (tpl_ok.png)
5. Skip animation (tpl_skip.png)
6. Snapshot after pull
7. Check SSR/SR + Swimsuit -> Special snapshot if both found
"""

from typing import Dict, Optional, Any, List
from datetime import datetime
from airtest.core.api import sleep, Template, exists
import os
import glob
import time

from core.base import BaseAutomation, CancellationError, ExecutionStep, StepResult
from core.agent import Agent
from core.utils import get_logger, StructuredLogger, ensure_directory
from core.data import ResultWriter
from core.config import GACHA_ROI_CONFIG, get_gacha_config, merge_config

logger = get_logger(__name__)


class GachaAutomation(BaseAutomation):
    """Automate Gacha pulls with template matching result verification."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None):
        base_config = get_gacha_config()
        cfg = merge_config(base_config, config) if config else base_config
        super().__init__(agent, cfg, GACHA_ROI_CONFIG, cancel_event=cancel_event)

        self.config = cfg
        self.wait_after_pull = cfg.get('wait_after_pull', 2.0)
        self.max_scroll_attempts = cfg.get('max_scroll_attempts', 10)
        
        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"gacha_{timestamp}.log")
        self.structured_logger = StructuredLogger(name="GachaAutomation", log_file=log_file)
        
        logger.info("GachaAutomation initialized")
        self.structured_logger.info(f"Log: {log_file} | Wait after pull: {self.wait_after_pull}s | Max scroll attempts: {self.max_scroll_attempts}")

    def find_banner(self, banner_path: str) -> bool:
        """Find banner by scrolling down if needed."""
        if not os.path.exists(banner_path):
            logger.error(f"Banner not found: {banner_path}")
            return False

        template = Template(banner_path)
        if exists(template):
            return True
        
        # Scroll to find
        for attempt in range(1, self.max_scroll_attempts + 1):
            self.check_cancelled(f"find_banner")
            if not self.touch_template("tpl_button_down.png", optional=True):
                break
            sleep(0.5)
            if exists(template):
                logger.info(f"Found banner after {attempt} scroll(s)")
                return True
        
        logger.error(f"Banner not found after {self.max_scroll_attempts} scrolls")
        return False

    def check_result(self, rarity: str, banner_folder: str, banner_path: str = "") -> Dict[str, Any]:
        """Check gacha result using template matching."""
        result = {'has_rarity': False, 'has_swimsuit': False, 'should_snapshot': False}
        
        self.structured_logger.subsection_header("RESULT VERIFICATION")
        
        # Check rarity template
        rarity_tpl = f"tpl_{rarity.lower()}.png"
        rarity_path = os.path.join(self.templates_path, rarity_tpl)
        if os.path.exists(rarity_path) and exists(Template(rarity_path)):
            result['has_rarity'] = True
            logger.info(f"✓ Found {rarity.upper()}")
            self.structured_logger.info(f"✓ Rarity check: Found {rarity.upper()}")
        else:
            self.structured_logger.info(f"✗ Rarity check: {rarity.upper()} not found")
        
        # Check swimsuit templates in banner folder (exclude banner file)
        banner_basename = os.path.basename(banner_path).lower() if banner_path else ""
        swimsuit_found = None
        for pattern in ['*.png', '*.jpg', '*.jpeg']:
            for file_path in glob.glob(os.path.join(banner_folder, pattern)):
                filename = os.path.basename(file_path).lower()
                # Skip banner file (exact match)
                if filename == banner_basename:
                    continue
                if exists(Template(file_path)):
                    result['has_swimsuit'] = True
                    swimsuit_found = os.path.basename(file_path)
                    logger.info(f"✓ Found swimsuit: {swimsuit_found}")
                    self.structured_logger.info(f"✓ Swimsuit check: Found {swimsuit_found}")
                    break
            if result['has_swimsuit']:
                break
        
        if not result['has_swimsuit']:
            self.structured_logger.info("✗ Swimsuit check: No swimsuit template found")
        
        result['should_snapshot'] = result['has_rarity'] and result['has_swimsuit']
        if result['should_snapshot']:
            logger.info("✓✓✓ SPECIAL MATCH! Both rarity and swimsuit found!")
            self.structured_logger.info("✓✓✓ SPECIAL MATCH! Both rarity and swimsuit found!")
        
        return result

    def run_pull(self, gacha: Dict[str, Any], pull_idx: int, folder: str) -> bool:
        """Run single gacha pull."""
        pull_type = gacha.get('pull_type', 'single')
        start_time = time.time()
        self.structured_logger.stage_start(pull_idx, f"{pull_type.upper()} PULL", gacha['name'])
        
        max_retries = 3
        screenshot = None

        try:
            # ==================== BANNER SELECTION ====================
            self.structured_logger.subsection_header("BANNER SELECTION")
            
            # Step 1: Find & Touch Banner
            banner_path = gacha['banner_path']
            banner_name = os.path.basename(banner_path)
            self.structured_logger.info(f"Looking for banner: {banner_name}")
            
            if not ExecutionStep(
                1, "Find & Touch Banner",
                lambda: self.find_banner(banner_path) and self.touch_template(banner_path),
                max_retries, 1.0, False, 1.0, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
                return False

            # ==================== PULL EXECUTION ====================
            self.structured_logger.subsection_header("PULL EXECUTION")
            
            # Step 2: Choose Pull Type
            pull_tpl = "tpl_single_pull.png" if pull_type == "single" else "tpl_multi_pull.png"
            self.structured_logger.info(f"Selecting {pull_type} pull button")
            
            if not ExecutionStep(
                2, f"Choose {pull_type} pull",
                lambda: self.touch_template(pull_tpl),
                max_retries, 1.0, False, 0.5, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
                return False

            # Step 3: Snapshot Before
            if not ExecutionStep(
                3, "Snapshot Before",
                lambda: self.snapshot_and_save(folder, f"{pull_idx:02d}_before.png"),
                max_retries, 1.0, False, 0, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
                return False

            # Step 4: Confirm Pull
            self.structured_logger.info("Confirming pull...")
            if not ExecutionStep(
                4, "Confirm Pull",
                lambda: self.touch_template("tpl_ok.png"),
                max_retries, 1.0, False, self.wait_after_pull, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
                return False

            # Step 5: Skip Animation
            self.structured_logger.info("Attempting to skip animation...")
            ExecutionStep(
                5, "Skip Animation",
                lambda: self.touch_template("tpl_skip.png", optional=True),
                1, 0.5, True, 1.0, self.check_cancelled, self.structured_logger
            ).execute()

            # Step 6: Snapshot After
            def capture():
                nonlocal screenshot
                screenshot = self.snapshot_and_save(folder, f"{pull_idx:02d}_after.png")
                return screenshot is not None
            
            if not ExecutionStep(
                6, "Snapshot After",
                capture, max_retries, 1.0, False, 0.5, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
                return False

            # Step 7: Check Result
            banner_folder = gacha.get('banner_folder', '')
            banner_path = gacha.get('banner_path', '')
            if banner_folder and os.path.exists(banner_folder):
                check_result = self.check_result(gacha.get('rarity', 'ssr'), banner_folder, banner_path)
                if check_result['should_snapshot']:
                    self.structured_logger.info("Saving special snapshot...")
                    self.snapshot_and_save(folder, f"{pull_idx:02d}_SPECIAL.png")

            duration = time.time() - start_time
            self.structured_logger.stage_end(pull_idx, True, duration)
            return True

        except CancellationError:
            self.structured_logger.warning(f"Pull {pull_idx} cancelled by user")
            self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
            return False
        except Exception as e:
            logger.error(f"Pull {pull_idx} failed: {e}")
            self.structured_logger.error(f"Pull {pull_idx} failed with exception: {e}")
            import traceback
            self.structured_logger.error(traceback.format_exc())
            self.structured_logger.stage_end(pull_idx, False, time.time() - start_time)
            return False

    def run(self, gachas_config: List[Dict[str, Any]]) -> bool:
        """Run gacha automation for multiple gachas."""
        if not self.agent.is_device_connected():
            logger.error("Device not connected")
            self.structured_logger.error("Device not connected")
            return False

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.results_dir}/gacha_{timestamp}.csv"
        result_writer = ResultWriter(output_path, auto_write=True)
        
        # Calculate total pulls
        total_pulls = sum(g.get('num_pulls', 0) for g in gachas_config)
        
        # Log automation start with configuration
        config_info = {
            "Total Gachas": len(gachas_config),
            "Total Pulls": total_pulls,
            "Output Path": output_path,
            "Wait After Pull": f"{self.wait_after_pull}s",
            "Max Scroll Attempts": self.max_scroll_attempts,
            "Max Retries": self.config.get('max_step_retries', 3)
        }
        
        self.structured_logger.automation_start("GACHA AUTOMATION", config_info)

        all_success = True
        total_success = 0
        total_failed = 0

        for idx, gacha in enumerate(gachas_config, 1):
            try:
                self.check_cancelled(f"gacha {idx}")
            except CancellationError:
                self.structured_logger.warning(f"Cancellation requested, stopping at gacha {idx}")
                # Ensure results are saved before exiting
                result_writer.flush()
                
                # Log summary
                duration = time.time() - start_time
                summary = {
                    "Total Processed": idx - 1,
                    "Total Success": total_success,
                    "Total Failed": total_failed,
                    "Duration": f"{duration:.2f}s",
                    "Status": "CANCELLED"
                }
                self.structured_logger.automation_end("GACHA AUTOMATION", False, summary)
                return False
            
            gacha_name = gacha.get('name', f'Gacha {idx}')
            pull_type = gacha.get('pull_type', 'single')
            num_pulls = gacha.get('num_pulls', 0)
            
            folder = f"{idx:02d}_{gacha_name}_{timestamp}"
            self.structured_logger.section_header(f"GACHA {idx}/{len(gachas_config)}: {gacha_name}")
            self.structured_logger.info(f"Configuration: {pull_type.upper()} pull x{num_pulls} | Rarity: {gacha.get('rarity', 'ssr').upper()}")
            
            success_count = 0
            failed_count = 0
            
            for pull_idx in range(1, num_pulls + 1):
                try:
                    self.check_cancelled(f"gacha {idx} pull {pull_idx}")
                except CancellationError:
                    self.structured_logger.warning(f"Cancellation requested, stopping at pull {pull_idx}")
                    all_success = False
                    break
                
                if self.run_pull(gacha, pull_idx, folder):
                    success_count += 1
                    total_success += 1
                else:
                    failed_count += 1
                    total_failed += 1
                
                sleep(0.5)
            
            if success_count != num_pulls:
                all_success = False
            
            # Log gacha completion summary
            self.structured_logger.info(f"Gacha {idx} completed: {success_count}/{num_pulls} successful | Failed: {failed_count}")
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
            "Results File": output_path
        }
        
        self.structured_logger.automation_end("GACHA AUTOMATION", all_success, summary)
        
        return all_success
