"""
Main GUI Application
"""

import sys
import os
# Add parent directory to path so we can import core module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import logging
import queue
from typing import Optional

from core.agent import Agent
from core.utils import get_logger
from gui.tabs.festival_tab import FestivalTab
from gui.tabs.gacha_tab import GachaTab
from gui.tabs.hopping_tab import HoppingTab
from gui.utils.logging_utils import OptimizedQueueHandler, OptimizedLogViewer
from gui.utils.thread_utils import get_thread_manager, shutdown_thread_manager, BackgroundTaskRunner

logger = get_logger(__name__)


class AutoCPeachGUI(tk.Tk):
    """Main GUI Application"""

    def __init__(self):
        super().__init__()

        self.title("Auto C-Peach")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Setup logging queue
        self.log_queue = queue.Queue()
        self.setup_logging()

        # Initialize Agent (without auto-connecting to device)
        try:
            self.agent = Agent(auto_connect=False)
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Agent: {e}")
            messagebox.showerror("Error", f"Cannot initialize Agent:\n{str(e)}")
            self.agent = None

        # Initialize thread manager and task runner
        thread_manager = get_thread_manager()
        self.task_runner = BackgroundTaskRunner(thread_manager)

        # Apply modern styling
        self.setup_styles()

        # Setup UI
        self.setup_ui()

        # Check initial device status
        self.check_device()

        # Start optimized log polling
        self.start_log_polling()

        logger.info("GUI initialized")

    def setup_logging(self):
        """Setup optimized logging."""
        # Use optimized queue handler with buffering
        queue_handler = OptimizedQueueHandler(self.log_queue, buffer_size=25, flush_interval=0.3)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%H:%M:%S')
        queue_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(queue_handler)

    def setup_styles(self):
        """Setup modern styling."""
        style = ttk.Style()

        # Use modern theme
        style.theme_use('alt')

        # Configure colors and button styles
        style.configure('Accent.TButton', font=('', 11, 'bold'))
        style.configure('TButton', font=('', 10))
        style.configure('TLabel', font=('', 10))
        style.configure('TEntry', font=('', 10))

    def setup_ui(self):
        """Setup main interface."""

        # === HEADER ===
        header_frame = ttk.Frame(self, relief='raised', borderwidth=1)
        header_frame.pack(fill='x', padx=5, pady=5)

        # Left side - Title
        left_header = ttk.Frame(header_frame)
        left_header.pack(side='left', padx=10, pady=5)

        title_label = ttk.Label(
            left_header,
            text="Auto C-Peach",
            font=('', 18, 'bold')
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            left_header,
            text="Game Automation Tool",
            font=('', 9)
        )
        subtitle_label.pack()

        # Right side - Device status
        right_header = ttk.Frame(header_frame)
        right_header.pack(side='right', padx=10, pady=5)

        self.device_status_var = tk.StringVar(value="Not Connected")
        status_label = ttk.Label(
            right_header,
            textvariable=self.device_status_var,
            font=('', 11, 'bold')
        )
        status_label.pack()

        # Button container for horizontal alignment
        button_frame = ttk.Frame(right_header)
        button_frame.pack(pady=2)

        self.connect_button = ttk.Button(
            button_frame,
            text="Connect Device",
            command=self.connect_device,
            width=18,
            style='Accent.TButton'
        )
        self.connect_button.pack(side='left', padx=(0, 5), ipady=5)

        ttk.Button(
            button_frame,
            text="Refresh",
            command=self.check_device,
            width=15
        ).pack(side='left', ipady=5)

        # === MAIN CONTENT ===
        content_frame = ttk.Frame(self)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill='both', expand=True, side='top')

        # Tab 1: Festival Automation
        if self.agent:
            self.festival_tab = FestivalTab(self.notebook, self.agent)
            self.notebook.add(self.festival_tab, text="Festival Automation")

        # Tab 2: Gacha Automation
        if self.agent:
            self.gacha_tab = GachaTab(self.notebook, self.agent)
            self.notebook.add(self.gacha_tab, text="Gacha Automation")

        # Tab 3: Hopping Automation
        if self.agent:
            self.hopping_tab = HoppingTab(self.notebook, self.agent)
            self.notebook.add(self.hopping_tab, text="Hopping Automation")

        # Tab 4: Settings
        settings_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(settings_tab, text="Settings")
        self.setup_settings_tab(settings_tab)

        # === OPTIMIZED LOG VIEWER ===
        self.log_viewer = OptimizedLogViewer(
            content_frame,
            self.log_queue,
            max_lines=int(self.max_log_lines_var.get()),
            poll_interval=200
        )

        # === FOOTER ===
        footer_frame = ttk.Frame(self, relief='sunken', borderwidth=1)
        footer_frame.pack(fill='x', side='bottom')

        ttk.Label(
            footer_frame,
            text="© 2025 Auto C-Peach | Version 1.0",
            font=('', 8)
        ).pack(side='left', padx=10, pady=3)

        self.footer_status_var = tk.StringVar(value="Ready")
        ttk.Label(
            footer_frame,
            textvariable=self.footer_status_var,
            font=('', 8)
        ).pack(side='right', padx=10, pady=3)

    def setup_settings_tab(self, parent):
        """Setup Settings tab."""

        # General Settings Section
        general_frame = ttk.LabelFrame(parent, text="General Settings", padding=15)
        general_frame.pack(fill='x', pady=10)

        # Log level
        log_frame = ttk.Frame(general_frame)
        log_frame.pack(fill='x', pady=5)

        ttk.Label(log_frame, text="Log Level:", font=('', 10)).pack(side='left', padx=5)

        self.log_level_var = tk.StringVar(value="INFO")
        log_combo = ttk.Combobox(
            log_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state='readonly',
            width=15
        )
        log_combo.pack(side='left', padx=5)

        ttk.Button(
            log_frame,
            text="Apply",
            command=self.apply_log_level,
            width=12
        ).pack(side='left', padx=5, ipady=5)

        # Theme (placeholder)
        theme_frame = ttk.Frame(general_frame)
        theme_frame.pack(fill='x', pady=5)

        ttk.Label(theme_frame, text="Theme:", font=('', 10)).pack(side='left', padx=5)

        self.theme_var = tk.StringVar(value="Light")
        theme_combo = ttk.Combobox(
            theme_frame,
            textvariable=self.theme_var,
            values=["Light", "Dark"],
            state='readonly',
            width=15
        )
        theme_combo.pack(side='left', padx=5)

        ttk.Label(theme_frame, text="(Coming soon)", font=('', 8)).pack(side='left', padx=5)

        # Performance Settings Section
        perf_frame = ttk.LabelFrame(parent, text="Performance Settings", padding=15)
        perf_frame.pack(fill='x', pady=10)

        # Max log lines
        self.max_log_lines_var = tk.StringVar(value="1000")
        ttk.Label(perf_frame, text="Max Log Lines:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(perf_frame, textvariable=self.max_log_lines_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Button(perf_frame, text="Apply", command=self.apply_performance_settings, width=8).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(perf_frame, text="(Reduce for better performance)", font=('', 8)).grid(row=1, column=0, columnspan=3, sticky='w', padx=5)

        # Poll interval
        ttk.Label(perf_frame, text="Log Poll Interval (ms):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.poll_interval_var = tk.StringVar(value="200")
        ttk.Entry(perf_frame, textvariable=self.poll_interval_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(perf_frame, text="(50-1000ms)", font=('', 8)).grid(row=2, column=2, sticky='w', padx=5, pady=2)

        perf_frame.columnconfigure(1, weight=1)

    def apply_log_level(self):
        """Apply log level."""
        level = self.log_level_var.get()
        logging.getLogger().setLevel(level)
        logger.info(f"Log level changed to: {level}")
        messagebox.showinfo("Success", f"Log level set to: {level}")

    def apply_performance_settings(self):
        """Apply performance settings to log viewer."""
        try:
            max_lines = int(self.max_log_lines_var.get())
            poll_interval = int(self.poll_interval_var.get())

            # Validate ranges
            max_lines = max(100, min(10000, max_lines))
            poll_interval = max(50, min(1000, poll_interval))

            # Apply to log viewer
            self.log_viewer.set_max_lines(max_lines)
            self.log_viewer.set_poll_interval(poll_interval)

            # Update variables with validated values
            self.max_log_lines_var.set(str(max_lines))
            self.poll_interval_var.set(str(poll_interval))

            logger.info(f"Performance settings applied: max_lines={max_lines}, poll_interval={poll_interval}ms")
            messagebox.showinfo("Success", f"Performance settings applied!\nMax lines: {max_lines}\nPoll interval: {poll_interval}ms")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid values:\n{str(e)}")

    def connect_device(self):
        """Connect to device."""
        if not self.agent:
            messagebox.showerror("Error", "Agent not initialized!")
            return

        self.device_status_var.set("Connecting...")
        self.footer_status_var.set("Connecting to device...")

        # Disable button temporarily
        self.connect_button.config(state='disabled')

        def on_complete(success):
            self.after(0, lambda: self._connect_finished(success))

        def on_error(error):
            logger.error(f"Connection error: {error}")
            self.after(0, lambda: self._connect_finished(False, str(error)))

        # Run connection using task runner
        self.task_runner.run_task(
            "device_connection",
            self.agent.connect_device_with_retry,
            on_complete=on_complete,
            on_error=on_error
        )

    def _connect_finished(self, success: bool, error_msg: str = ""):
        """Callback when connection completes."""
        if success:
            # Disable connect button when successfully connected
            self.connect_button.config(state='disabled')
            self.device_status_var.set("Device Connected")
            self.footer_status_var.set("Device ready")
            logger.info("Device connected successfully")
            messagebox.showinfo("Success", "Device connected successfully!")
        else:
            # Re-enable button on failure
            self.connect_button.config(state='normal')
            self.device_status_var.set("Connection Failed")
            self.footer_status_var.set("Connection failed")
            msg = "Failed to connect to device!"
            if error_msg:
                msg += f"\n\nError: {error_msg}"
            messagebox.showerror("Connection Failed", msg)

    def check_device(self):
        """Check device connection."""
        if not self.agent:
            self.device_status_var.set("Agent Error")
            self.footer_status_var.set("Agent initialization failed")
            self.connect_button.config(state='disabled')
            return

        if self.agent.is_device_connected():
            self.device_status_var.set("Device Connected")
            self.footer_status_var.set("Device ready")
            # Disable connect button when device is already connected
            self.connect_button.config(state='disabled')
            logger.info("Device connected")
        else:
            self.device_status_var.set("Not Connected")
            self.footer_status_var.set("Device not connected")
            # Enable connect button when device is not connected
            self.connect_button.config(state='normal')
            logger.warning("Device not connected")

    def start_log_polling(self):
        """Start optimized log polling."""
        self.log_viewer.start_polling()

    def clear_logs(self):
        """Clear logs."""
        self.log_viewer.clear_logs()

    def save_logs(self):
        """Save logs to file."""
        self.log_viewer.save_logs()


def main():
    """Entry point."""
    app = AutoCPeachGUI()
    try:
        app.mainloop()
    finally:
        # Cleanup thread manager on exit
        shutdown_thread_manager()
        logger.info("Application shutdown complete")


if __name__ == '__main__':
    main()
