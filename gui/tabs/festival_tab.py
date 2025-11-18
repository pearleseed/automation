"""
Festival Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from typing import Dict, Any

from automations.festivals import FestivalAutomation
from core.agent import Agent
from core.utils import get_logger
from gui.components.base_tab import BaseAutomationTab

logger = get_logger(__name__)


class FestivalTab(BaseAutomationTab):
    """Tab for Festival Automation with improved UI."""

    def __init__(self, parent, agent: Agent):
        super().__init__(parent, agent, "Festival", FestivalAutomation)

    def _setup_tab_specific_vars(self):
        """Setup Festival-specific variables."""
        self.output_file_var = tk.StringVar(value="")
        self.force_new_session_var = tk.BooleanVar(value=False)

    def _create_config_ui(self, parent):
        """Create Festival-specific configuration UI."""
        # Output Configuration
        output_section = ttk.LabelFrame(parent, text="Output", padding=10)
        output_section.pack(fill='x', pady=5)

        output_inner = ttk.Frame(output_section)
        output_inner.pack(fill='x')

        # Output file (optional)
        ttk.Label(output_inner, text="Output File:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(output_inner, textvariable=self.output_file_var, width=30, font=('', 10)).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(output_inner, text="Browse", command=self.browse_output, width=8).grid(row=0, column=2, pady=2, ipady=5)
        ttk.Label(output_inner, text="(Optional - auto-generated if empty)", font=('', 8), foreground='gray').grid(row=1, column=0, columnspan=3, sticky='w')

        output_inner.columnconfigure(1, weight=1)

        # Resume Configuration
        resume_section = ttk.LabelFrame(parent, text="Resume Options", padding=10)
        resume_section.pack(fill='x', pady=5)

        resume_inner = ttk.Frame(resume_section)
        resume_inner.pack(fill='x')

        # Force new session checkbox
        ttk.Checkbutton(
            resume_inner,
            text="Force New Session (ignore previous resume state)",
            variable=self.force_new_session_var
        ).pack(anchor='w', pady=2)
        ttk.Label(
            resume_inner, 
            text="✓ Auto-resume is enabled by default if previous session was interrupted",
            font=('', 8), 
            foreground='#2E7D32'
        ).pack(anchor='w', padx=20)

    def _get_automation_config(self) -> Dict[str, Any]:
        """Get Festival-specific automation config."""
        return {}

    def browse_output(self):
        """Browse output file (optional)."""
        filename = filedialog.asksaveasfilename(
            title="Save results to (optional)",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_file_var.set(filename)
            logger.info(f"Output file: {filename}")

    def start_automation(self):
        """Override to add Festival-specific validation and progress initialization."""
        # Validate
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid CSV/JSON file!")
            return

        # Check device
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!\nPlease connect device first.")
            return

        # Load data to get count for progress tracking
        from core.data import load_data
        try:
            data_list = load_data(file_path)
            if not data_list:
                messagebox.showerror("Error", "No data in file!")
                return
            total_count = len(data_list)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load data:\n{str(e)}")
            return

        # Check for resume state if not forcing new session
        if not self.force_new_session_var.get():
            try:
                temp_automation = self.automation_class(self.agent, {}, cancel_event=None)
                resume_state = temp_automation._manage_resume_state('load')
                
                if resume_state:
                    # Ask user if they want to resume
                    resume_msg = (
                        f"Found unfinished session:\n\n"
                        f"• Data: {os.path.basename(resume_state.get('data_path', ''))}\n"
                        f"• Progress: Stage {resume_state.get('current_stage', 1)}/{resume_state.get('total_stages', total_count)}\n"
                        f"• Output: {os.path.basename(resume_state.get('output_path', ''))}\n\n"
                        f"Continue from where you left off?"
                    )
                    
                    resume_choice = messagebox.askyesnocancel("Resume Session?", resume_msg, icon='question')
                    
                    if resume_choice is None:  # Cancel
                        return
                    elif resume_choice is False:  # No - start new
                        self.force_new_session_var.set(True)
                        temp_automation._manage_resume_state('clear')
                        logger.info("Starting new session")
            except Exception as e:
                logger.warning(f"Resume check failed: {e}")

        # Initialize progress
        self.progress_panel.start(total_count)

        # Call parent implementation
        super().start_automation()

    def _run_automation(self, file_path: str, config: Dict[str, Any]):
        """Override to handle Festival-specific automation logic."""
        try:
            # Initialize automation instance if not already done
            if not self.automation_instance:
                self.automation_instance = self.automation_class(self.agent, config, cancel_event=self.thread_cancel_event)
                if not self.automation_instance:
                    raise RuntimeError("Failed to initialize Festival automation")
            
            # Get output path if specified
            output_file = self.output_file_var.get().strip()
            output_path = output_file if output_file else None
            
            # Get force new session flag
            force_new_session = self.force_new_session_var.get()

            # Run with all parameters (always use the new run method that supports resume)
            success = self.automation_instance.run(
                data_path=file_path,
                use_detector=False,  # Can be extended to support detector option in GUI
                output_path=output_path,
                force_new_session=force_new_session
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
