"""
Gacha Automation

Gacha flow:
1. Find and touch gacha banner (scroll if needed)
2. Touch single/multi pull button
3. Snapshot before pull
4. Confirm pull (tpl_ok.png)
5. Skip animation (tpl_allskip.png)
6. Snapshot after pull
7. Check SSR/SR + Swimsuit -> Special snapshot if both found
8. Close result
"""

from typing import Dict, Optional, Any, List
from datetime import datetime
from airtest.core.api import sleep, Template, exists
import time
import os
import glob

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

        self.wait_after_pull = cfg.get('wait_after_pull', 2.0)
        self.max_scroll_attempts = cfg.get('max_scroll_attempts', 10)
        
        log_dir = os.path.join(self.results_dir, "logs")
        ensure_directory(log_dir)
        log_file = os.path.join(log_dir, f"gacha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.structured_logger = StructuredLogger(name="GachaAutomation", log_file=log_file)
        
        logger.info(f"GachaAutomation initialized | Log: {log_file}")

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

    def check_result(self, rarity: str, banner_folder: str) -> Dict[str, Any]:
        """Check gacha result using template matching."""
        result = {'has_rarity': False, 'has_swimsuit': False, 'should_snapshot': False}
        
        # Check rarity template
        rarity_tpl = f"tpl_{rarity.lower()}.png"
        rarity_path = os.path.join(self.templates_path, rarity_tpl)
        if os.path.exists(rarity_path) and exists(Template(rarity_path)):
            result['has_rarity'] = True
            logger.info(f"✓ Found {rarity.upper()}")
        
        # Check swimsuit templates in banner folder (exclude banner.* files)
        for pattern in ['*.png', '*.jpg', '*.jpeg']:
            for file_path in glob.glob(os.path.join(banner_folder, pattern)):
                filename = os.path.basename(file_path).lower()
                # Skip banner file
                if filename.startswith('banner.'):
                    continue
                if exists(Template(file_path)):
                    result['has_swimsuit'] = True
                    logger.info(f"✓ Found swimsuit: {os.path.basename(file_path)}")
                    break
            if result['has_swimsuit']:
                break
        
        result['should_snapshot'] = result['has_rarity'] and result['has_swimsuit']
        if result['should_snapshot']:
            logger.info("✓✓✓ SPECIAL MATCH! Both rarity and swimsuit found!")
        
        return result

    def run_pull(self, gacha: Dict[str, Any], pull_idx: int, folder: str) -> bool:
        """Run single gacha pull."""
        pull_type = gacha.get('pull_type', 'single')
        self.structured_logger.stage_start(pull_idx, f"{pull_type.upper()} PULL", gacha['name'])
        
        max_retries = 3
        screenshot = None

        try:
            # Step 1: Find & Touch Banner
            banner_path = gacha['banner_path']
            if not ExecutionStep(
                1, "Find & Touch Banner",
                lambda: self.find_banner(banner_path) and self.touch_template(banner_path),
                max_retries, 1.0, False, 1.0, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                return False

            # Step 2: Choose Pull Type
            pull_tpl = "tpl_single_pull.png" if pull_type == "single" else "tpl_multi_pull.png"
            if not ExecutionStep(
                2, f"Choose {pull_type} pull",
                lambda: self.touch_template(pull_tpl),
                max_retries, 1.0, False, 0.5, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                return False

            # Step 3: Snapshot Before
            if not ExecutionStep(
                3, "Snapshot Before",
                lambda: self.snapshot_and_save(folder, f"{pull_idx:02d}_before.png"),
                max_retries, 1.0, False, 0, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                return False

            # Step 4: Confirm Pull
            if not ExecutionStep(
                4, "Confirm Pull",
                lambda: self.touch_template("tpl_ok.png"),
                max_retries, 1.0, False, self.wait_after_pull, self.check_cancelled, self.structured_logger
            ).execute() == StepResult.SUCCESS:
                return False

            # Step 5: Skip Animation (optional)
            ExecutionStep(
                5, "Skip Animation",
                lambda: self.touch_template("tpl_allskip.png", optional=True),
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
                return False

            # Step 7: Check Result
            banner_folder = gacha.get('banner_folder', '')
            if banner_folder and os.path.exists(banner_folder):
                check_result = self.check_result(gacha.get('rarity', 'ssr'), banner_folder)
                if check_result['should_snapshot']:
                    self.snapshot_and_save(folder, f"{pull_idx:02d}_SPECIAL.png")

            # Step 8: Close (optional)
            ExecutionStep(
                8, "Close Result",
                lambda: self.touch_template("tpl_ok.png", optional=True),
                1, 0.3, True, 0.5, self.check_cancelled, self.structured_logger
            ).execute()

            self.structured_logger.stage_end(pull_idx, True, 0)
            return True

        except (CancellationError, Exception) as e:
            logger.error(f"Pull {pull_idx} failed: {e}")
            return False

    def run(self, gachas_config: List[Dict[str, Any]]) -> bool:
        """Run gacha automation for multiple gachas."""
        if not self.agent.is_device_connected():
            logger.error("Device not connected")
            return False

        output_path = f"{self.results_dir}/gacha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_writer = ResultWriter(output_path, auto_write=True)
        
        self.structured_logger.automation_start("GACHA AUTOMATION", {
            "Total Gachas": len(gachas_config),
            "Output": output_path
        })

        all_success = True

        for idx, gacha in enumerate(gachas_config, 1):
            folder = f"{idx:02d}_{gacha['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.structured_logger.section_header(f"GACHA {idx}/{len(gachas_config)}: {gacha['name']}")
            
            success_count = 0
            for pull_idx in range(1, gacha['num_pulls'] + 1):
                try:
                    self.check_cancelled(f"gacha {idx} pull {pull_idx}")
                except CancellationError:
                    all_success = False
                    break
                
                if self.run_pull(gacha, pull_idx, folder):
                    success_count += 1
                sleep(0.5)
            
            if success_count != gacha['num_pulls']:
                all_success = False
            
            self.structured_logger.info(f"Completed: {success_count}/{gacha['num_pulls']} successful")
            sleep(1.0)

        result_writer.flush()
        self.structured_logger.automation_end("GACHA AUTOMATION", all_success, {
            "Success": "All" if all_success else "Partial",
            "Results": output_path
        })
        
        return all_success
