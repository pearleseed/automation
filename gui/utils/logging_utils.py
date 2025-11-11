"""
Optimized Logging System for GUI
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
import queue
import threading
import time
from typing import Optional
from core.utils import get_logger

logger = get_logger(__name__)


class OptimizedQueueHandler(logging.Handler):
    """Optimized logging handler with buffering and batch processing."""

    def __init__(self, log_queue, buffer_size=50, flush_interval=0.5):
        super().__init__()
        self.log_queue = log_queue
        self.buffer = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.lock = threading.Lock()

    def emit(self, record):
        """Buffer log records and flush in batches."""
        log_entry = self.format(record)

        with self.lock:
            self.buffer.append(log_entry)

            # Flush if buffer is full or enough time has passed
            current_time = time.time()
            if (len(self.buffer) >= self.buffer_size or
                current_time - self.last_flush >= self.flush_interval):
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered log entries to queue."""
        if self.buffer:
            # Join multiple entries for efficiency
            batch_entry = '\n'.join(self.buffer)
            try:
                self.log_queue.put_nowait(batch_entry)
            except queue.Full:
                # If queue is full, put individual entries
                for entry in self.buffer:
                    try:
                        self.log_queue.put_nowait(entry)
                    except queue.Full:
                        break  # Stop trying if queue remains full

            self.buffer.clear()
            self.last_flush = time.time()

    def flush(self):
        """Force flush remaining buffer."""
        with self.lock:
            self._flush_buffer()


class OptimizedLogViewer:
    """Optimized log viewer with adaptive polling and performance monitoring."""

    def __init__(self, parent, log_queue, max_lines=1000, poll_interval=200):
        self.parent = parent
        self.log_queue = log_queue
        self.max_lines = max_lines
        self.poll_interval = poll_interval  # milliseconds
        self.auto_scroll = True
        self.is_polling = False
        self.poll_job_id: Optional[str] = None

        # Performance tracking
        self.last_poll_time = 0
        self.poll_count = 0
        self.adaptive_mode = True

        self._setup_ui()

    def _setup_ui(self):
        """Setup the log viewer UI."""
        # Log container
        log_container = ttk.Frame(self.parent)
        log_container.pack(fill='both', expand=True, pady=(5, 0))

        # Log header with controls
        log_header = ttk.Frame(log_container)
        log_header.pack(fill='x')

        ttk.Label(log_header, text="📋 Activity Logs", font=('', 10, 'bold')).pack(side='left', padx=5)

        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            log_header,
            text="Auto-scroll",
            variable=self.auto_scroll_var
        ).pack(side='right', padx=5)

        # Performance indicator
        self.perf_label = ttk.Label(log_header, text="", font=('', 8), foreground='gray')
        self.perf_label.pack(side='right', padx=10)

        # Control buttons
        ttk.Button(
            log_header,
            text="Clear",
            command=self.clear_logs,
            width=10
        ).pack(side='right', padx=5, ipady=4)

        ttk.Button(
            log_header,
            text="💾 Save",
            command=self.save_logs,
            width=10
        ).pack(side='right', padx=5, ipady=4)

        # Log text widget
        log_frame = ttk.Frame(log_container)
        log_frame.pack(fill='both', expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            wrap=tk.WORD,
            font=('Courier', 9),
            state='normal'  # Allow editing for selection/copy
        )
        self.log_text.pack(fill='both', expand=True)

        # Bind auto-scroll variable
        self.auto_scroll_var.trace_add('write', self._on_auto_scroll_changed)

    def _on_auto_scroll_changed(self, *args):
        """Handle auto-scroll setting change."""
        self.auto_scroll = self.auto_scroll_var.get()

    def start_polling(self):
        """Start optimized log polling."""
        if not self.is_polling:
            self.is_polling = True
            self._poll_logs()

    def stop_polling(self):
        """Stop log polling."""
        self.is_polling = False
        if self.poll_job_id:
            self.parent.after_cancel(self.poll_job_id)
            self.poll_job_id = None

    def _poll_logs(self):
        """Optimized log polling with adaptive frequency."""
        if not self.is_polling:
            return

        start_time = time.time()
        updated = False
        batch_entries = []

        # Process all available log entries
        while True:
            try:
                entry = self.log_queue.get_nowait()
                batch_entries.append(entry)
            except queue.Empty:
                break

        # Insert batch entries efficiently
        if batch_entries:
            self.log_text.insert('end', '\n'.join(batch_entries) + '\n')
            updated = True

            # Limit log lines for performance
            self._limit_log_lines()

        # Adaptive polling frequency
        if self.adaptive_mode:
            poll_time = time.time() - start_time
            self.poll_count += 1

            # Adjust polling frequency based on activity
            if batch_entries:
                # High activity - poll more frequently
                next_interval = max(50, self.poll_interval // 2)
            else:
                # Low activity - poll less frequently
                next_interval = min(1000, self.poll_interval * 2)

            self.poll_interval = next_interval

            # Update performance indicator occasionally
            if self.poll_count % 10 == 0:
                self._update_performance_indicator(poll_time)

        # Auto scroll if enabled and we had updates
        if updated and self.auto_scroll:
            self.log_text.see('end')

        # Schedule next poll
        if self.is_polling:
            self.poll_job_id = self.parent.after(self.poll_interval, self._poll_logs)

    def _limit_log_lines(self):
        """Limit the number of log lines for performance."""
        try:
            # Get total lines
            content = self.log_text.get('1.0', 'end-1c')
            lines = content.split('\n')

            if len(lines) > self.max_lines:
                # Keep only the most recent lines
                keep_lines = lines[-(self.max_lines // 2):]
                self.log_text.delete('1.0', 'end')
                self.log_text.insert('1.0', '\n'.join(keep_lines) + '\n')
                logger.info(f"Trimmed logs to {len(keep_lines)} lines for performance")
        except Exception as e:
            logger.warning(f"Error limiting log lines: {e}")

    def _update_performance_indicator(self, poll_time: float):
        """Update performance indicator."""
        try:
            # Calculate polling frequency
            freq = 1000 / self.poll_interval  # polls per second
            self.perf_label.config(
                text=".1f"
            )
        except Exception as e:
            logger.warning(f"Error updating performance indicator: {e}")

    def clear_logs(self):
        """Clear all logs."""
        self.log_text.delete('1.0', 'end')
        logger.info("Logs cleared by user")

    def save_logs(self):
        """Save logs to file."""
        from gui.utils.ui_utils import UIUtils

        filename = UIUtils.save_file(self.parent, "Save logs", ".txt",
                                   [("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            try:
                content = self.log_text.get('1.0', 'end')
                with open(filename, 'w', encoding='utf-8') as f:
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
        self.poll_interval = max(50, min(2000, interval_ms))  # Clamp between 50ms and 2s
        logger.info(f"Poll interval set to: {self.poll_interval}ms")

    def destroy(self):
        """Cleanup when destroying the viewer."""
        self.stop_polling()
