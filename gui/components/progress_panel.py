"""
Progress Panel Component
"""

import time
import tkinter as tk
from tkinter import ttk


class ProgressPanel(ttk.Frame):
    """Panel displaying progress and statistics for automation tasks.

    This panel shows a progress bar, success/failure counts, elapsed time,
    and estimated time of arrival (ETA) for running automations.
    """

    def __init__(self, parent):
        super().__init__(parent, relief="flat", borderwidth=0)

        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", pady=(3, 5))
        ttk.Label(
            title_frame, text="Progress & Statistics", font=("Segoe UI", 10, "bold")
        ).pack()

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress.pack(fill="x", padx=8, pady=3)

        # Progress label
        self.progress_label = ttk.Label(self, text="Ready", font=("Segoe UI", 8))
        self.progress_label.pack(pady=1)

        # Stats frame with compact layout
        stats_frame = ttk.Frame(self)
        stats_frame.pack(fill="x", padx=8, pady=3)

        # Row 1: Counts
        count_frame = ttk.Frame(stats_frame)
        count_frame.pack(fill="x", pady=1)

        # Total
        ttk.Label(count_frame, text="Total:", font=("Segoe UI", 8)).pack(side="left")
        self.total_label = ttk.Label(
            count_frame, text="0", font=("Segoe UI", 9, "bold")
        )
        self.total_label.pack(side="left", padx=(3, 10))

        # Success
        ttk.Label(count_frame, text="Success:", font=("Segoe UI", 8)).pack(side="left")
        self.ok_label = ttk.Label(
            count_frame, text="0", font=("Segoe UI", 9, "bold"), foreground="#2e7d32"
        )
        self.ok_label.pack(side="left", padx=(3, 10))

        # Failed
        ttk.Label(count_frame, text="Failed:", font=("Segoe UI", 8)).pack(side="left")
        self.ng_label = ttk.Label(
            count_frame, text="0", font=("Segoe UI", 9, "bold"), foreground="#d32f2f"
        )
        self.ng_label.pack(side="left", padx=3)

        # Row 2: Time info
        time_frame = ttk.Frame(stats_frame)
        time_frame.pack(fill="x", pady=1)

        # Time elapsed
        ttk.Label(time_frame, text="Elapsed:", font=("Segoe UI", 8)).pack(side="left")
        self.time_label = ttk.Label(
            time_frame, text="00:00:00", font=("Segoe UI", 9, "bold")
        )
        self.time_label.pack(side="left", padx=(3, 10))

        # ETA
        ttk.Label(time_frame, text="ETA:", font=("Segoe UI", 8)).pack(side="left")
        self.eta_label = ttk.Label(
            time_frame, text="--:--:--", font=("Segoe UI", 9, "bold")
        )
        self.eta_label.pack(side="left", padx=3)

        # Initialize counters
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.total = 0
        self.ok_count = 0
        self.ng_count = 0
        self.start_time = None
        self.update_display()

    def start(self, total: int):
        """Start progress tracking."""
        self.total = total
        self.ok_count = 0
        self.ng_count = 0
        self.start_time = time.time()
        self.update_display()

    def update_result(self, ok: bool):
        """Update progress with new result."""
        self.ok_count += ok
        self.ng_count += not ok

        processed = self.ok_count + self.ng_count
        # Update every 5 items, or if this is the last item
        if processed % 5 == 0 or processed >= self.total:
            self.update_display()

    def update_display(self):
        """Update progress display."""
        self.total_label.config(text=str(self.total))
        self.ok_label.config(text=str(self.ok_count))
        self.ng_label.config(text=str(self.ng_count))

        if self.total > 0:
            processed = self.ok_count + self.ng_count
            progress = (processed / self.total) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{processed}/{self.total}")
        else:
            self.progress_var.set(0)
            self.progress_label.config(text="Ready")

        if self.start_time:
            elapsed = time.time() - self.start_time
            self.time_label.config(text=self._format_time(elapsed))

            processed = self.ok_count + self.ng_count
            if 0 < processed < self.total:
                remaining = elapsed / processed * (self.total - processed)
                self.eta_label.config(text=self._format_time(remaining))
            elif processed >= self.total:
                self.eta_label.config(text="Done")

    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
