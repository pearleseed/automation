"""
Progress Components - Status tracking for automation.
"""

import time
import tkinter as tk
from collections import deque
from tkinter import ttk
from typing import Optional


# ==================== PROGRESS PANEL ====================

class ProgressPanel(ttk.Frame):
    """Panel displaying progress and statistics for automation tasks.

    Features:
    - Progress bar with percentage
    - Success/failure/skip counts
    - Elapsed time and ETA
    - Current item display
    - Average time per item
    """

    def __init__(self, parent):
        super().__init__(parent, relief="flat", borderwidth=0)
        
        self.item_times: deque = deque(maxlen=20)  # For average calculation
        self.current_item_start: Optional[float] = None

        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", pady=(3, 5))
        ttk.Label(
            title_frame, text="Progress & Statistics", font=("Segoe UI", 10, "bold")
        ).pack()

        # Progress bar with percentage
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill="x", padx=8, pady=3)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress.pack(fill="x", side="left", expand=True)
        
        self.percent_label = ttk.Label(
            progress_frame, text="0%", font=("Segoe UI", 9, "bold"), width=5
        )
        self.percent_label.pack(side="right", padx=(5, 0))

        # Progress label (current item)
        self.progress_label = ttk.Label(self, text="Ready", font=("Segoe UI", 8))
        self.progress_label.pack(pady=1)
        
        # Current item display
        current_frame = ttk.LabelFrame(self, text="Current", padding=3)
        current_frame.pack(fill="x", padx=8, pady=3)
        
        self.current_item_label = ttk.Label(
            current_frame, text="--", font=("Segoe UI", 9),
            foreground="#1976D2", wraplength=220
        )
        self.current_item_label.pack(anchor="w")

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
        self.total_label.pack(side="left", padx=(3, 8))

        # Success
        ttk.Label(count_frame, text="✓", font=("Segoe UI", 8), foreground="#2e7d32").pack(side="left")
        self.ok_label = ttk.Label(
            count_frame, text="0", font=("Segoe UI", 9, "bold"), foreground="#2e7d32"
        )
        self.ok_label.pack(side="left", padx=(2, 8))

        # Failed
        ttk.Label(count_frame, text="✗", font=("Segoe UI", 8), foreground="#d32f2f").pack(side="left")
        self.ng_label = ttk.Label(
            count_frame, text="0", font=("Segoe UI", 9, "bold"), foreground="#d32f2f"
        )
        self.ng_label.pack(side="left", padx=(2, 8))
        
        # Skipped
        ttk.Label(count_frame, text="⊘", font=("Segoe UI", 8), foreground="#666").pack(side="left")
        self.skip_label = ttk.Label(
            count_frame, text="0", font=("Segoe UI", 9, "bold"), foreground="#666"
        )
        self.skip_label.pack(side="left", padx=(2, 0))

        # Row 2: Time info
        time_frame = ttk.Frame(stats_frame)
        time_frame.pack(fill="x", pady=1)

        # Time elapsed
        ttk.Label(time_frame, text="Elapsed:", font=("Segoe UI", 8)).pack(side="left")
        self.time_label = ttk.Label(
            time_frame, text="00:00:00", font=("Segoe UI", 9, "bold")
        )
        self.time_label.pack(side="left", padx=(3, 8))

        # ETA
        ttk.Label(time_frame, text="ETA:", font=("Segoe UI", 8)).pack(side="left")
        self.eta_label = ttk.Label(
            time_frame, text="--:--:--", font=("Segoe UI", 9, "bold")
        )
        self.eta_label.pack(side="left", padx=(3, 8))
        
        # Avg time
        ttk.Label(time_frame, text="Avg:", font=("Segoe UI", 8)).pack(side="left")
        self.avg_label = ttk.Label(
            time_frame, text="--", font=("Segoe UI", 9, "bold")
        )
        self.avg_label.pack(side="left", padx=(3, 0))

        # Initialize counters
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.total = 0
        self.ok_count = 0
        self.ng_count = 0
        self.skip_count = 0
        self.start_time = None
        self.item_times.clear()
        self.current_item_start = None
        self.update_display()
        self.current_item_label.config(text="--")

    def start(self, total: int):
        """Start progress tracking."""
        self.total = total
        self.ok_count = 0
        self.ng_count = 0
        self.skip_count = 0
        self.start_time = time.time()
        self.item_times.clear()
        self.update_display()

    def start_item(self, item_name: str = ""):
        """Mark start of processing an item."""
        self.current_item_start = time.time()
        self.current_item_label.config(text=item_name or "Processing...")

    def update_result(self, ok: bool, item_name: str = "", skip: bool = False):
        """Update progress with new result."""
        # Calculate item time
        if self.current_item_start:
            item_time = time.time() - self.current_item_start
            self.item_times.append(item_time)
            self.current_item_start = None
        
        # Update counts
        if skip:
            self.skip_count += 1
        elif ok:
            self.ok_count += 1
        else:
            self.ng_count += 1
        
        # Update display
        self.update_display()

    def update_display(self):
        """Update progress display."""
        self.total_label.config(text=str(self.total))
        self.ok_label.config(text=str(self.ok_count))
        self.ng_label.config(text=str(self.ng_count))
        self.skip_label.config(text=str(self.skip_count))

        processed = self.ok_count + self.ng_count + self.skip_count
        
        if self.total > 0:
            progress = (processed / self.total) * 100
            self.progress_var.set(progress)
            self.percent_label.config(text=f"{progress:.0f}%")
            self.progress_label.config(text=f"{processed}/{self.total}")
        else:
            self.progress_var.set(0)
            self.percent_label.config(text="0%")
            self.progress_label.config(text="Ready")

        if self.start_time:
            elapsed = time.time() - self.start_time
            self.time_label.config(text=self._format_time(elapsed))

            if 0 < processed < self.total:
                remaining = elapsed / processed * (self.total - processed)
                self.eta_label.config(text=self._format_time(remaining))
            elif processed >= self.total:
                self.eta_label.config(text="Done")
                self.current_item_label.config(text="Completed!")
        
        # Update average time
        if self.item_times:
            avg_time = sum(self.item_times) / len(self.item_times)
            self.avg_label.config(text=f"{avg_time:.1f}s")

    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
