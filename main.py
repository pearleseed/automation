"""
Main GUI Application
"""

import logging
import queue
import tkinter as tk
from tkinter import messagebox, ttk

from core.agent import Agent
from core.utils import get_logger
from gui.tabs.festival_tab import FestivalTab
from gui.tabs.gacha_tab import GachaTab
from gui.tabs.hopping_tab import HoppingTab
from gui.utils.logging_utils import LogViewer, QueueHandler
from gui.utils.thread_utils import get_thread_manager, shutdown_thread_manager

logger = get_logger(__name__)


class AutoCPeachGUI(tk.Tk):
    """Main GUI Application for Auto C-Peach automation tool.

    This is the main application window that provides tabs for Festival, Gacha,
    and Hopping automations, along with device management, logging, and settings.
    """

    def __init__(self):
        super().__init__()

        self.title("Auto C-Peach - Game Automation Tool")
        self.geometry("1400x900")
        self.minsize(1200, 800)

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

        # Initialize thread manager
        self.thread_manager = get_thread_manager()

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
        """Configure logging with buffered queue handler."""
        queue_handler = QueueHandler(self.log_queue, buffer_size=25, flush_interval=0.3)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        queue_handler.setFormatter(formatter)
        logging.getLogger().addHandler(queue_handler)

    def setup_styles(self):
        """Configure UI styling with modern theme."""
        style = ttk.Style()
        style.theme_use("clam")  # More modern than "alt"

        # Accent button - primary actions
        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#1976d2",
            foreground="white",
            borderwidth=0,
            focuscolor="none",
        )
        style.map(
            "Accent.TButton", background=[("active", "#1565c0"), ("pressed", "#0d47a1")]
        )

        # Regular buttons
        style.configure("TButton", font=("Segoe UI", 9), padding=5)

        # Labels
        style.configure("TLabel", font=("Segoe UI", 9))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

        # Entry fields
        style.configure("TEntry", font=("Segoe UI", 9), padding=5)

        # Notebook tabs
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=[15, 8])

    def setup_ui(self):
        """Create main user interface with header, tabs, logs, and footer.

        Sets up the complete UI layout including device status, automation tabs,
        activity log viewer, and status bar.
        """
        # Header with gradient-like appearance
        header_frame = ttk.Frame(self, relief="flat")
        header_frame.pack(fill="x", padx=0, pady=0)

        # Add colored bar at top
        top_bar = tk.Frame(header_frame, bg="#1976d2", height=3)
        top_bar.pack(fill="x")

        header_content = ttk.Frame(header_frame, padding=8)
        header_content.pack(fill="x")

        left_header = ttk.Frame(header_content)
        left_header.pack(side="left")
        ttk.Label(left_header, text="Auto C-Peach", font=("Segoe UI", 14, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            left_header,
            text="Game Automation Tool",
            font=("Segoe UI", 8),
            foreground="#666",
        ).pack(anchor="w", pady=(1, 0))

        right_header = ttk.Frame(header_content)
        right_header.pack(side="right")

        # Device status
        status_container = ttk.Frame(right_header)
        status_container.pack(pady=(0, 3))

        self.device_status_var = tk.StringVar(value="Not Connected")
        self.device_status_label = ttk.Label(
            status_container,
            textvariable=self.device_status_var,
            font=("Segoe UI", 9, "bold"),
            foreground="#d32f2f",
        )
        self.device_status_label.pack()

        button_frame = ttk.Frame(right_header)
        button_frame.pack()

        self.connect_button = ttk.Button(
            button_frame,
            text="Connect Device",
            command=self.connect_device,
            width=16,
            style="Accent.TButton",
        )
        self.connect_button.pack(side="left", padx=(0, 3), ipady=4)

        ttk.Button(
            button_frame, text="Refresh", command=self.check_device, width=10
        ).pack(side="left", ipady=4)

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Main content
        content_frame = ttk.Frame(self)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create PanedWindow for resizable split between tabs and logs
        # User can drag the divider to adjust space between tabs and activity log
        paned = ttk.PanedWindow(content_frame, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True)

        # Top pane: Automation tabs
        tabs_frame = ttk.Frame(paned)
        paned.add(tabs_frame, weight=3)  # Give more weight to tabs

        self.notebook = ttk.Notebook(tabs_frame)
        self.notebook.pack(fill="both", expand=True)

        if self.agent:
            self.festival_tab = FestivalTab(self.notebook, self.agent)
            self.notebook.add(self.festival_tab, text="Festival Automation")

            self.gacha_tab = GachaTab(self.notebook, self.agent)
            self.notebook.add(self.gacha_tab, text="Gacha Automation")

            self.hopping_tab = HoppingTab(self.notebook, self.agent)
            self.notebook.add(self.hopping_tab, text="Hopping Automation")

        settings_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(settings_tab, text="Settings")
        self.setup_settings_tab(settings_tab)

        # Bottom pane: Log viewer
        log_frame = ttk.Frame(paned)
        paned.add(log_frame, weight=1)  # Give less weight to logs

        self.log_viewer = LogViewer(
            log_frame,
            self.log_queue,
            max_lines=int(self.max_log_lines_var.get()),
            poll_interval=200,
        )

        # Footer
        footer_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        footer_frame.pack(fill="x", side="bottom")
        ttk.Label(
            footer_frame, text="© 2025 Auto C-Peach | Version 1.0", font=("", 8)
        ).pack(side="left", padx=10, pady=3)
        self.footer_status_var = tk.StringVar(value="Ready")
        ttk.Label(footer_frame, textvariable=self.footer_status_var, font=("", 8)).pack(
            side="right", padx=10, pady=3
        )

    def setup_settings_tab(self, parent):
        """Create settings tab interface."""
        general_frame = ttk.LabelFrame(parent, text="General Settings", padding=15)
        general_frame.pack(fill="x", pady=10)

        log_frame = ttk.Frame(general_frame)
        log_frame.pack(fill="x", pady=5)
        ttk.Label(log_frame, text="Log Level:", font=("", 10)).pack(side="left", padx=5)

        self.log_level_var = tk.StringVar(value="INFO")
        log_combo = ttk.Combobox(
            log_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=15,
        )
        log_combo.pack(side="left", padx=5)
        ttk.Button(
            log_frame, text="Apply", command=self.apply_log_level, width=12
        ).pack(side="left", padx=5, ipady=5)

        perf_frame = ttk.LabelFrame(parent, text="Performance Settings", padding=15)
        perf_frame.pack(fill="x", pady=10)

        self.max_log_lines_var = tk.StringVar(value="1000")
        ttk.Label(perf_frame, text="Max Log Lines:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(perf_frame, textvariable=self.max_log_lines_var, width=10).grid(
            row=0, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Button(
            perf_frame, text="Apply", command=self.apply_performance_settings, width=8
        ).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(
            perf_frame, text="(Reduce for better performance)", font=("", 8)
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=5)

        ttk.Label(perf_frame, text="Log Poll Interval (ms):").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        self.poll_interval_var = tk.StringVar(value="200")
        ttk.Entry(perf_frame, textvariable=self.poll_interval_var, width=10).grid(
            row=2, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Label(perf_frame, text="(50-1000ms)", font=("", 8)).grid(
            row=2, column=2, sticky="w", padx=5, pady=2
        )
        perf_frame.columnconfigure(1, weight=1)

    def apply_log_level(self):
        """Change application log level."""
        level = self.log_level_var.get()
        logging.getLogger().setLevel(level)
        logger.info(f"Log level changed to: {level}")
        messagebox.showinfo("Success", f"Log level set to: {level}")

    def apply_performance_settings(self):
        """Update performance settings."""
        try:
            max_lines = max(100, min(10000, int(self.max_log_lines_var.get())))
            poll_interval = max(50, min(1000, int(self.poll_interval_var.get())))

            self.log_viewer.set_max_lines(max_lines)
            self.log_viewer.set_poll_interval(poll_interval)

            self.max_log_lines_var.set(str(max_lines))
            self.poll_interval_var.set(str(poll_interval))

            logger.info(
                f"Performance settings applied: max_lines={max_lines}, poll_interval={poll_interval}ms"
            )
            messagebox.showinfo(
                "Success",
                f"Performance settings applied!\nMax lines: {max_lines}\nPoll interval: {poll_interval}ms",
            )
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid values:\n{str(e)}")

    def connect_device(self):
        """Connect to device with retry in background thread.

        Initiates device connection in a separate thread to avoid blocking the UI.
        Updates device status and shows result dialog upon completion.
        """
        if not self.agent:
            messagebox.showerror("Error", "Agent not initialized!")
            return

        # Store agent reference for use in nested function
        agent = self.agent

        self.device_status_var.set("Connecting...")
        self.device_status_label.config(foreground="#1976d2")
        self.footer_status_var.set("Connecting to device...")
        self.connect_button.config(state="disabled")

        def connect_task():
            try:
                success = agent.connect_device_with_retry()
                self.after(0, lambda: self._connect_finished(success))
                return success
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.after(0, lambda: self._connect_finished(False, str(e)))
                raise

        thread = get_thread_manager().submit_task("device_connection", connect_task)
        if not thread:
            self._connect_finished(False, "Failed to start connection thread")

    def _connect_finished(self, success: bool, error_msg: str = ""):
        """Handle device connection result."""
        try:
            if success:
                self.connect_button.config(state="disabled")
                self.device_status_var.set("Device Connected")
                self.device_status_label.config(foreground="#2e7d32")
                self.footer_status_var.set("Device ready")
                logger.info("Device connected successfully")
                messagebox.showinfo("Success", "Device connected successfully!")
            else:
                self.connect_button.config(state="normal")
                self.device_status_var.set("Connection Failed")
                self.device_status_label.config(foreground="#d32f2f")
                self.footer_status_var.set("Connection failed")
                msg = "Failed to connect to device!"
                if error_msg:
                    msg += f"\n\nError: {error_msg}"
                messagebox.showerror("Connection Failed", msg)
        except Exception as err:
            logger.error(f"Error in _connect_finished: {err}")

    def check_device(self):
        """Check current device connection status."""
        if not self.agent:
            self.device_status_var.set("Agent Error")
            self.footer_status_var.set("Agent initialization failed")
            self.connect_button.config(state="disabled")
            return

        if self.agent.is_device_connected():
            self.device_status_var.set("Device Connected")
            self.device_status_label.config(foreground="#2e7d32")
            self.footer_status_var.set("Device ready")
            self.connect_button.config(state="disabled")
            logger.info("Device connected")
        else:
            self.device_status_var.set("Not Connected")
            self.device_status_label.config(foreground="#d32f2f")
            self.footer_status_var.set("Device not connected")
            self.connect_button.config(state="normal")
            logger.warning("Device not connected")

    def start_log_polling(self):
        """Start log polling."""
        self.log_viewer.start_polling()


def main():
    """Application entry point."""
    app = AutoCPeachGUI()
    try:
        app.mainloop()
    finally:
        shutdown_thread_manager()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
