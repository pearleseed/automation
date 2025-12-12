"""
Common UI Utilities for GUI Components
"""

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, Optional

from core.utils import get_logger

logger = get_logger(__name__)


class UIUtils:
    """Utility class for common UI operations and dialogs.

    This class provides static methods for file/directory browsing, input validation,
    configuration management, and common UI operations used across the application.
    """

    @staticmethod
    def browse_file(
        parent: tk.Widget, title: str, filetypes: list, initial_dir: str = ""
    ) -> Optional[str]:
        """Open file browsing dialog with error handling.

        Args:
            parent: Parent widget for the dialog.
            title: Dialog window title.
            filetypes: List of file type filters (e.g., [("CSV files", "*.csv")]).
            initial_dir: Initial directory to open (default: current directory).

        Returns:
            Optional[str]: Selected file path, or None if cancelled or error occurred.
        """
        try:
            filename = filedialog.askopenfilename(
                parent=parent,
                title=title,
                filetypes=filetypes,
                initialdir=initial_dir or None,
            )
            if filename:
                logger.info(f"Selected file: {filename}")
                return filename
        except Exception as e:
            logger.error(f"File dialog error: {e}")
            messagebox.showerror("Error", f"Cannot open file dialog:\n{str(e)}")
        return None

    @staticmethod
    def browse_directory(
        parent: tk.Widget, title: str, initial_dir: str = ""
    ) -> Optional[str]:
        """Common directory browsing dialog with error handling."""
        try:
            directory = filedialog.askdirectory(
                parent=parent, title=title, initialdir=initial_dir or None
            )
            if directory:
                logger.info(f"Selected directory: {directory}")
                return directory
        except Exception as e:
            logger.error(f"Directory dialog error: {e}")
            messagebox.showerror("Error", f"Cannot open directory dialog:\n{str(e)}")
        return None

    @staticmethod
    def save_file(
        parent: tk.Widget,
        title: str,
        defaultextension: str = "",
        filetypes: Optional[list] = None,
    ) -> Optional[str]:
        """Common save file dialog with error handling."""
        if filetypes is None:
            filetypes = [("All files", "*.*")]

        try:
            filename = filedialog.asksaveasfilename(
                parent=parent,
                title=title,
                defaultextension=defaultextension,
                filetypes=filetypes,
            )
            if filename:
                logger.info(f"Save file selected: {filename}")
                return filename
        except Exception as e:
            logger.error(f"Save file dialog error: {e}")
            messagebox.showerror("Error", f"Cannot open save dialog:\n{str(e)}")
        return None

    @staticmethod
    def validate_numeric_input(
        value: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        default: float = 0.0,
    ) -> float:
        """Validate numeric input with bounds checking."""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                logger.warning(
                    f"Value {num} below minimum {min_val}, using minimum value"
                )
                return min_val
            if max_val is not None and num > max_val:
                logger.warning(
                    f"Value {num} above maximum {max_val}, using maximum value"
                )
                return max_val
            return num
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value '{value}', using default {default}")
            return default

    @staticmethod
    def validate_integer_input(
        value: str,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        default: int = 0,
    ) -> int:
        """Validate integer input with bounds checking."""
        try:
            num = int(float(value))  # Handle float strings like "1.0"
            if min_val is not None and num < min_val:
                logger.warning(
                    f"Value {num} below minimum {min_val}, using minimum value"
                )
                return min_val
            if max_val is not None and num > max_val:
                logger.warning(
                    f"Value {num} above maximum {max_val}, using maximum value"
                )
                return max_val
            return num
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value '{value}', using default {default}")
            return default

    @staticmethod
    def save_config_to_file(
        parent: tk.Widget,
        config: Dict[str, Any],
        title: str = "Save config",
        defaultextension: str = ".json",
    ) -> bool:
        """Save configuration to JSON file with error handling."""
        filename = UIUtils.save_file(
            parent,
            title,
            defaultextension,
            [("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return False

        try:
            with open(filename, "w", encoding="utf-8-sig") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Config saved to:\n{filename}")
            logger.info(f"Config saved: {filename}")
            return True
        except Exception as e:
            logger.error(f"Cannot save config: {e}")
            messagebox.showerror("Error", f"Cannot save config:\n{str(e)}")
            return False

    @staticmethod
    def load_config_from_file(
        parent: tk.Widget, title: str = "Load config"
    ) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file with error handling."""
        filename = UIUtils.browse_file(
            parent, title, [("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return None

        try:
            with open(filename, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
            messagebox.showinfo("Success", "Config loaded successfully!")
            logger.info(f"Config loaded: {filename}")
            return config
        except Exception as e:
            logger.error(f"Cannot load config: {e}")
            messagebox.showerror("Error", f"Cannot load config:\n{str(e)}")
            return None

    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """Ensure directory exists, create if necessary."""
        if not directory:
            return False

        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"Cannot create directory {directory}: {e}")
            return False

    @staticmethod
    def open_directory_explorer(directory: str) -> bool:
        """Open directory in Windows Explorer."""
        if not UIUtils.ensure_directory_exists(directory):
            return False

        try:
            import subprocess

            subprocess.call(["explorer", os.path.abspath(directory)])
            logger.info(f"Opened directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"Cannot open directory {directory}: {e}")
            return False

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def calculate_eta(processed: int, total: int, elapsed_time: float) -> str:
        """Calculate estimated time of arrival."""
        if processed <= 0 or total <= 0 or processed >= total:
            return "--:--:--"

        try:
            remaining = elapsed_time / processed * (total - processed)
            return UIUtils.format_time(remaining)
        except (ZeroDivisionError, OverflowError):
            return "--:--:--"

    @staticmethod
    def show_info(parent: tk.Widget, title: str, message: str):
        """Show info message dialog."""
        messagebox.showinfo(title, message, parent=parent)

    @staticmethod
    def show_warning(parent: tk.Widget, title: str, message: str):
        """Show warning message dialog."""
        messagebox.showwarning(title, message, parent=parent)

    @staticmethod
    def show_error(parent: tk.Widget, title: str, message: str):
        """Show error message dialog."""
        messagebox.showerror(title, message, parent=parent)


# ==================== TOOLTIPS ====================

class ToolTip:
    """Hover tooltip for widgets with configurable delay."""
    def __init__(self, widget: tk.Widget, text: str, delay: int = 500):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip_window = None
        self.scheduled = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, e=None):
        self._cancel()
        self.scheduled = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self.scheduled:
            self.widget.after_cancel(self.scheduled)
            self.scheduled = None

    def _show(self, e=None):
        if self.tip_window: return
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_attributes("-topmost", True)
        frame = tk.Frame(self.tip_window, bg="#333", borderwidth=1, relief="solid")
        frame.pack()
        tk.Label(frame, text=self.text, bg="#333", fg="#fff", font=("Segoe UI", 9), wraplength=250, padx=8, pady=5).pack()
        self.tip_window.wm_geometry(f"+{x}+{y}")

    def _hide(self, e=None):
        self._cancel()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class TooltipManager:
    """Centralized tooltip management with predefined tips."""
    TIPS = {
        "start_button": "Start automation (Ctrl+Enter)",
        "stop_button": "Stop automation (Ctrl+Q, ESC, F9)",
        "pause_button": "Pause/Resume (Ctrl+P)",
        "browse_file": "Select data file (CSV/JSON)",
        "preview_data": "Preview file contents",
        "force_new_session": "Ignore saved progress, start fresh",
    }
    _tooltips: Dict[str, ToolTip] = {}

    @classmethod
    def add_tooltip(cls, widget: tk.Widget, key_or_text: str, delay: int = 500):
        """Add tooltip to widget using predefined key or custom text."""
        text = cls.TIPS.get(key_or_text, key_or_text)
        cls._tooltips[str(id(widget))] = ToolTip(widget, text, delay)
