"""
Quick Actions Panel Component
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict


class QuickActionsPanel(ttk.Frame):
    """Panel containing quick action buttons for common operations.

    This panel provides quick access to device checking, screenshots, OCR testing,
    opening results, copying logs, and clearing cache.
    """

    def __init__(self, parent, callbacks: Dict[str, Any]):
        super().__init__(parent, relief="flat", borderwidth=0)
        self.callbacks = callbacks

        ttk.Label(self, text="Quick Actions", font=("Segoe UI", 10, "bold")).pack(
            pady=(3, 5)
        )

        # Buttons grid - compact
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="both", expand=True, padx=3, pady=3)

        # Row 1
        ttk.Button(
            btn_frame,
            text="Check Device",
            command=lambda: self._call("check_device"),
            width=15,
        ).grid(row=0, column=0, sticky="ew", padx=2, pady=2, ipady=5)

        ttk.Button(
            btn_frame,
            text="Screenshot",
            command=lambda: self._call("screenshot"),
            width=15,
        ).grid(row=0, column=1, sticky="ew", padx=2, pady=2, ipady=5)

        # Row 2
        ttk.Button(
            btn_frame, text="OCR Test", command=lambda: self._call("ocr_test"), width=15
        ).grid(row=1, column=0, sticky="ew", padx=2, pady=2, ipady=5)

        ttk.Button(
            btn_frame,
            text="Open Results",
            command=lambda: self._call("open_output"),
            width=15,
        ).grid(row=1, column=1, sticky="ew", padx=2, pady=2, ipady=5)

        # Row 3
        ttk.Button(
            btn_frame,
            text="Copy Logs",
            command=lambda: self._call("copy_logs"),
            width=15,
        ).grid(row=2, column=0, sticky="ew", padx=2, pady=2, ipady=5)

        ttk.Button(
            btn_frame,
            text="Clear Cache",
            command=lambda: self._call("clear_cache"),
            width=15,
        ).grid(row=2, column=1, sticky="ew", padx=2, pady=2, ipady=5)

        # Configure grid weights
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def _call(self, action: str):
        """Execute callback action if available."""
        if action in self.callbacks:
            self.callbacks[action]()
