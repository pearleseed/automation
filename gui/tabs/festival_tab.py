"""
Festival Tab for GUI
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict

from automations.festivals import FestivalAutomation
from core.agent import Agent
from core.utils import get_logger
from gui.components.base_tab import BaseAutomationTab

logger = get_logger(__name__)


class FestivalTab(BaseAutomationTab):
    """Tab for Festival Automation with resume support and progress tracking.

    This tab provides UI for configuring and running festival automation,
    including data file selection, output configuration, and resume options
    for interrupted sessions.
    """

    def __init__(self, parent, agent: Agent):
        super().__init__(parent, agent, "Festival", FestivalAutomation)

    def _setup_tab_specific_vars(self):
        """Setup Festival-specific variables."""
        self.output_file_var = tk.StringVar(value="")
        self.force_new_session_var = tk.BooleanVar(value=False)
        self.start_stage_var = tk.StringVar(value="1")  # Start stage index
        self.loaded_stages = []  # Store loaded stage data for selection

    def _setup_file_section(self, parent):
        """Override to combine file selection and stage selection in one compact section."""
        file_section = ttk.LabelFrame(parent, text="Data & Stage Selection", padding=8)
        file_section.pack(fill="x", pady=(0, 6))

        # Row 1: File selection
        row1 = ttk.Frame(file_section)
        row1.pack(fill="x", pady=(0, 3))

        ttk.Label(row1, text="Data File:", font=("Segoe UI", 9, "bold"), width=12).pack(
            side="left"
        )
        ttk.Entry(row1, textvariable=self.file_path_var, font=("Segoe UI", 9)).pack(
            side="left", fill="x", expand=True, padx=3
        )
        ttk.Button(row1, text="Browse", command=self.browse_file, width=8).pack(
            side="left", padx=2
        )
        ttk.Button(row1, text="Preview", command=self.preview_data, width=8).pack(
            side="left", padx=2
        )

        # Row 2: Stage selection
        row2 = ttk.Frame(file_section)
        row2.pack(fill="x", pady=(3, 0))

        ttk.Label(
            row2, text="Start Stage:", font=("Segoe UI", 9, "bold"), width=12
        ).pack(side="left")
        self.stage_combo = ttk.Combobox(
            row2,
            textvariable=self.start_stage_var,
            values=["1"],
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.stage_combo.pack(side="left", fill="x", expand=True, padx=3)

        # Help text
        ttk.Label(
            file_section,
            text="Select data file, then choose starting stage from dropdown",
            font=("Segoe UI", 8),
            foreground="#666",
        ).pack(anchor="w", pady=(3, 0))

    def _create_config_ui(self, parent):
        """Create Festival-specific configuration UI with compact design."""

        # Output Configuration
        output_section = ttk.LabelFrame(parent, text="Output Settings", padding=8)
        output_section.pack(fill="x", pady=(0, 6))

        output_inner = ttk.Frame(output_section)
        output_inner.pack(fill="x")

        # Output file (optional)
        ttk.Label(output_inner, text="Output File:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(
            output_inner, textvariable=self.output_file_var, font=("Segoe UI", 9)
        ).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(output_inner, text="...", command=self.browse_output, width=3).grid(
            row=0, column=2, pady=3, ipady=2
        )
        ttk.Label(
            output_inner,
            text="Optional - auto-generated if empty",
            font=("Segoe UI", 8),
            foreground="#666",
        ).grid(row=1, column=0, columnspan=3, sticky="w")

        output_inner.columnconfigure(1, weight=1)

        # Resume Configuration
        resume_section = ttk.LabelFrame(parent, text="Resume Options", padding=8)
        resume_section.pack(fill="x", pady=(0, 6))

        resume_inner = ttk.Frame(resume_section)
        resume_inner.pack(fill="x")

        # Force new session checkbox
        ttk.Checkbutton(
            resume_inner,
            text="Force New Session (ignore previous resume state)",
            variable=self.force_new_session_var,
        ).pack(anchor="w", pady=3)
        ttk.Label(
            resume_inner,
            text="Auto-resume enabled by default if session was interrupted",
            font=("Segoe UI", 8),
            foreground="#2E7D32",
        ).pack(anchor="w", padx=15, pady=(0, 3))

    def _get_automation_config(self) -> Dict[str, Any]:
        """Get Festival-specific automation config."""
        start_stage_idx = self._parse_stage_index(self.start_stage_var.get())
        return {"start_stage_index": start_stage_idx}

    def _parse_stage_index(self, stage_str: str) -> int:
        """Parse stage index from selection string."""
        try:
            return int(stage_str.split(".")[0])
        except (ValueError, IndexError, AttributeError):
            return 1

    def browse_output(self):
        """Browse output file (optional)."""
        filename = filedialog.asksaveasfilename(
            title="Save results to (optional)",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            self.output_file_var.set(filename)
            logger.info(f"Output file: {filename}")

    def browse_file(self):
        """Override to load stages when file is selected."""
        super().browse_file()
        # Load stages after file selection
        self._load_stages_for_selection()

    def _load_stages_for_selection(self):
        """Load stages from selected file and populate stage selection dropdown."""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            return

        try:
            from core.data import load_data

            self.loaded_stages = load_data(file_path)
            if not self.loaded_stages:
                return

            # Create stage options: "idx. [rank] fest_name - fest_rank"
            stage_options = [
                f"{idx}. [{stage.get('推奨ランク', '?')}] "
                f"{stage.get('フェス名', 'Unknown')} - {stage.get('フェスランク', 'Unknown')}"
                for idx, stage in enumerate(self.loaded_stages, 1)
            ]

            # Update combobox
            self.stage_combo["values"] = stage_options
            self.start_stage_var.set(stage_options[0] if stage_options else "1")

            logger.info(f"Loaded {len(self.loaded_stages)} stages for selection")

        except Exception as e:
            logger.error(f"Failed to load stages: {e}")

    def start_automation(self):
        """Override to add Festival-specific validation and progress initialization."""
        # Validate
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid CSV/JSON file!")
            return

        # Check device
        if not self.agent.is_device_connected():
            messagebox.showerror(
                "Error", "Device not connected!\nPlease connect device first."
            )
            return

        # Load data to get count for progress tracking
        from core.data import load_data

        try:
            data_list = load_data(file_path)
            if not data_list:
                messagebox.showerror("Error", "No data in file!")
                return

            # Get start stage index and validate
            start_stage_idx = self._parse_stage_index(self.start_stage_var.get())

            if start_stage_idx > len(data_list):
                messagebox.showerror(
                    "Error",
                    f"Start stage {start_stage_idx} exceeds total stages ({len(data_list)})!",
                )
                return

            # Calculate remaining stages
            total_count = len(data_list) - start_stage_idx + 1
            if start_stage_idx > 1:
                logger.info(
                    f"Starting from stage {start_stage_idx}/{len(data_list)} | "
                    f"Remaining: {total_count} stages"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load data:\n{str(e)}")
            return

        # Check for resume state if not forcing new session
        if not self.force_new_session_var.get():
            try:
                temp_automation = self.automation_class(
                    self.agent, {}, cancel_event=None
                )
                resume_state = temp_automation._manage_resume_state("load")

                if resume_state:
                    # Build resume message
                    start_stage = resume_state.get("start_stage_index", 1)
                    stage_info = (
                        f"• Start Stage: {start_stage}\n" if start_stage > 1 else ""
                    )

                    resume_msg = (
                        f"Found unfinished session:\n\n"
                        f"• Data: {os.path.basename(resume_state.get('data_path', ''))}\n"
                        f"{stage_info}"
                        f"• Progress: Stage {resume_state.get('current_stage', 1)}/"
                        f"{resume_state.get('total_stages', total_count)}\n"
                        f"• Output: {os.path.basename(resume_state.get('output_path', ''))}\n\n"
                        f"Continue from where you left off?"
                    )

                    resume_choice = messagebox.askyesnocancel(
                        "Resume Session?", resume_msg, icon="question"
                    )

                    if resume_choice is None:  # Cancel
                        return
                    elif resume_choice is False:  # No - start new
                        self.force_new_session_var.set(True)
                        temp_automation._manage_resume_state("clear")
                        logger.info("Starting new session")
            except Exception as err:
                logger.warning(f"Resume check failed: {err}")

        # Call parent implementation
        super().start_automation()

    def _run_automation(self, file_path: str, config: Dict[str, Any]):
        """Override to handle Festival-specific automation logic."""
        try:
            # Initialize automation instance with thread-safe access
            with self._state_lock:
                if not self._automation_instance:
                    self._automation_instance = self.automation_class(
                        self.agent, config,
                        cancel_event=self.thread_cancel_event,
                        pause_event=self.pause_event,
                    )
                    if not self._automation_instance:
                        raise RuntimeError("Failed to initialize Festival automation")

                # Get reference while holding lock
                automation = self._automation_instance

            # Get output path if specified
            output_file = self.output_file_var.get().strip()
            output_path = output_file if output_file else None

            # Get force new session flag
            force_new_session = self.force_new_session_var.get()

            # Get start stage index
            start_stage_index = config.get("start_stage_index", 1)

            # Run with all parameters (outside lock to allow cancellation)
            success = automation.run(
                data_path=file_path,
                use_detector=False,  # Can be extended to support detector option in GUI
                output_path=output_path,
                force_new_session=force_new_session,
                start_stage_index=start_stage_index,
            )

            # Call completion callback if not cancelled
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(success))

            return success

        except Exception as e:
            logger.error(f"Festival automation error: {e}")
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(False, str(e)))
            raise
