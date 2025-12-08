"""
Logging System for GUI.

Provides buffered logging with bounded queues to prevent memory issues
during long-running automation sessions.
"""

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Optional

from core.utils import get_logger

logger = get_logger(__name__)

# Maximum queue size to prevent unbounded memory growth
MAX_LOG_QUEUE_SIZE = 10000


class QueueHandler(logging.Handler):
    """Logging handler with buffering, batch processing, and bounded queue.

    This handler buffers log records and flushes them in batches to improve
    performance. Uses a bounded queue to prevent memory issues during
    long-running sessions.

    Attributes:
        log_queue: Queue to send log entries to.
        buffer_size: Number of entries to buffer before flushing.
        flush_interval: Maximum time between flushes in seconds.
        dropped_count: Number of log entries dropped due to full queue.
    """

    def __init__(
        self,
        log_queue: queue.Queue,
        buffer_size: int = 50,
        flush_interval: float = 0.5,
    ):
        super().__init__()
        self.log_queue = log_queue
        self.buffer: list = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.buffer_lock = threading.Lock()
        self.dropped_count = 0
        self._drop_warning_threshold = 100

    def emit(self, record):
        """Buffer log records and flush in batches."""
        log_entry = self.format(record)

        with self.buffer_lock:
            self.buffer.append(log_entry)

            # Flush if buffer is full or enough time has passed
            current_time = time.time()
            if (
                len(self.buffer) >= self.buffer_size
                or current_time - self.last_flush >= self.flush_interval
            ):
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered log entries to queue with overflow handling."""
        if not self.buffer:
            return

        # Join multiple entries for efficiency
        batch_entry = "\n".join(self.buffer)
        try:
            self.log_queue.put_nowait(batch_entry)
        except queue.Full:
            # Queue is full - drop oldest entries and track
            self.dropped_count += len(self.buffer)

            # Log warning periodically about dropped entries
            if self.dropped_count >= self._drop_warning_threshold:
                # Try to put a warning message
                try:
                    warning_msg = f"[WARNING] Log queue full - {self.dropped_count} entries dropped"
                    self.log_queue.put_nowait(warning_msg)
                    self._drop_warning_threshold = self.dropped_count + 100
                except queue.Full:
                    pass

        self.buffer.clear()
        self.last_flush = time.time()

    def flush(self):
        """Force flush remaining buffer."""
        with self.buffer_lock:
            self._flush_buffer()


class LogViewer:
    """Log viewer with adaptive polling and performance monitoring.

    This widget displays application logs in real-time with auto-scrolling,
    line limiting, and batch processing for optimal performance.
    """

    def __init__(self, parent, log_queue, max_lines=1000, poll_interval=200):
        self.parent = parent
        self.log_queue = log_queue
        self.max_lines = max_lines
        self.poll_interval = poll_interval
        self.auto_scroll = True
        self.is_polling = False
        self.poll_job_id: Optional[str] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the log viewer UI."""
        # Log container with border
        log_container = ttk.LabelFrame(self.parent, text="Activity Logs", padding=5)
        log_container.pack(fill="both", expand=True)

        # Log header with controls
        log_header = ttk.Frame(log_container)
        log_header.pack(fill="x", pady=(0, 3))

        # Control buttons
        ttk.Button(log_header, text="Clear", command=self.clear_logs, width=10).pack(
            side="left", padx=2, ipady=3
        )

        ttk.Button(log_header, text="Save", command=self.save_logs, width=10).pack(
            side="left", padx=2, ipady=3
        )

        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            log_header, text="Auto-scroll", variable=self.auto_scroll_var
        ).pack(side="right", padx=5)

        # Info label
        ttk.Label(
            log_header,
            text="Drag the divider above to resize",
            font=("", 8),
            foreground="#666",
        ).pack(side="right", padx=10)

        # Log text widget with compact height
        log_frame = ttk.Frame(log_container)
        log_frame.pack(fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,  # Reduced from 12 to 8 for better space distribution
            wrap=tk.WORD,
            font=("Courier", 9),
            state="normal",  # Allow editing for selection/copy
        )
        self.log_text.pack(fill="both", expand=True)

        # Bind auto-scroll variable
        self.auto_scroll_var.trace_add("write", self._on_auto_scroll_changed)

    def _on_auto_scroll_changed(self, *args):
        """Handle auto-scroll setting change."""
        self.auto_scroll = self.auto_scroll_var.get()

    def start_polling(self):
        """Start optimized log polling."""
        if not self.is_polling:
            self.is_polling = True
            self._poll_logs()

    def stop_polling(self) -> None:
        """Stop log polling."""
        self.is_polling = False
        if self.poll_job_id:
            self.parent.after_cancel(self.poll_job_id)
            self.poll_job_id = None

    def _poll_logs(self):
        """Poll and display log entries."""
        if not self.is_polling:
            return

        batch_entries = []
        while len(batch_entries) < 50:
            try:
                entry = self.log_queue.get_nowait()
                batch_entries.append(entry)
            except queue.Empty:
                break

        if batch_entries:
            self.log_text.insert("end", "\n".join(batch_entries) + "\n")
            self._limit_log_lines()

            if self.auto_scroll:
                self.log_text.see("end")

        if self.is_polling:
            self.poll_job_id = self.parent.after(self.poll_interval, self._poll_logs)

    def _limit_log_lines(self):
        """Keep log within maximum line limit."""
        content = self.log_text.get("1.0", "end-1c")
        lines = content.split("\n")

        if len(lines) > self.max_lines:
            keep_lines = lines[-int(self.max_lines * 0.5) :]
            self.log_text.delete("1.0", "end")
            self.log_text.insert("1.0", "\n".join(keep_lines) + "\n")

    def clear_logs(self):
        """Clear all logs."""
        self.log_text.delete("1.0", "end")
        logger.info("Logs cleared by user")

    def save_logs(self):
        """Save logs to file."""
        from gui.utils.ui_utils import UIUtils

        filename = UIUtils.save_file(
            self.parent,
            "Save logs",
            ".txt",
            [("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filename:
            try:
                content = self.log_text.get("1.0", "end")
                with open(filename, "w", encoding="utf-8-sig") as f:
                    f.write(content)
                UIUtils.show_info(self.parent, "Success", f"Logs saved to:\n{filename}")
            except Exception as e:
                UIUtils.show_error(self.parent, "Error", f"Cannot save logs:\n{str(e)}")

    def set_max_lines(self, max_lines: int):
        """Set maximum number of log lines."""
        self.max_lines = max_lines
        logger.info(f"Max log lines set to: {max_lines}")

    def set_poll_interval(self, interval_ms: int):
        """Set polling interval in milliseconds."""
        self.poll_interval = max(
            50, min(2000, interval_ms)
        )  # Clamp between 50ms and 2s
        logger.info(f"Poll interval set to: {self.poll_interval}ms")

    def destroy(self):
        """Cleanup when destroying the viewer."""
        self.stop_polling()
