"""Agent Module - Device interaction and OCR.

This module provides a unified interface for device interaction through Airtest
and OCR operations with thread-safe implementations and proper resource management.
"""

import threading
import time
from typing import List, Optional, Tuple, Union

import cv2  # type: ignore[import-untyped]
import numpy as np
from airtest.core.api import connect_device, swipe, touch
from airtest.core.error import AirtestError

import core.oneocr_optimized as oneocr

from .config import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_DEVICE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OCR_TIMEOUT,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TOUCH_TIMES,
)
from .utils import get_logger

# ==================== OCR ENGINE ENHANCEMENT ====================


class EnhancedOcrEngine(oneocr.OcrEngine):
    """Enhanced OCR engine with direct NumPy array processing for better performance.

    This class extends the OcrEngine to process images directly from NumPy arrays,
    eliminating the overhead of encoding/decoding operations while maintaining thread safety.

    Attributes:
        timeout: Maximum time in seconds for OCR operations (default: 5.0).
    """

    def __init__(self, timeout: float = DEFAULT_OCR_TIMEOUT):
        """Initialize enhanced OCR engine.

        Args:
            timeout: Maximum time in seconds for OCR operations.
        """
        super().__init__()
        self.timeout = timeout

    def recognize(self, image_array: np.ndarray) -> dict:
        """Process image from NumPy array efficiently without encode/decode overhead.

        Args:
            image_array: Input image as NumPy array (height, width, channels).

        Returns:
            dict: OCR recognition result containing detected text and bounding boxes.
                Returns empty result with error message if image size is invalid.

        Note:
            Thread-safe: uses internal lock for concurrent access.
        """
        if not self._initialized:
            return self._error_result("OCR engine not initialized")

        # Validate input
        if image_array is None or image_array.size == 0:
            return self._error_result("Empty or invalid image array")

        # Validate image size
        height, width = image_array.shape[:2]
        error = self._validate_image_size(width, height)
        if error:
            return self._error_result(error)

        # Convert to BGRA format
        channels = image_array.shape[2] if len(image_array.shape) == 3 else 1
        if channels == 1:
            img_bgra = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGRA)
        elif channels == 3:
            img_bgra = cv2.cvtColor(image_array, cv2.COLOR_BGR2BGRA)
        elif channels == 4:
            # Already BGRA or RGBA - assume BGRA
            img_bgra = image_array
        else:
            return self._error_result(f"Unsupported channel count: {channels}")

        # Process with thread safety
        with self._lock:
            return self._process_image(
                cols=img_bgra.shape[1],
                rows=img_bgra.shape[0],
                step=img_bgra.shape[1] * 4,
                data=img_bgra.ctypes.data,
            )


# ==================== AGENT CLASS ====================


class Agent:
    """Agent for device interaction via Airtest and OCR processing.

    This class provides a unified interface for interacting with devices through Airtest
    and performing OCR operations. It handles device connection, screenshot capture,
    and text recognition with automatic retry capabilities.

    Thread Safety:
        - Device operations are protected by _device_lock
        - OCR operations are protected by the OCR engine's internal lock
        - State variables use atomic operations where possible
    """

    def __init__(
        self,
        device_url: Optional[str] = None,
        enable_retry: bool = True,
        auto_connect: bool = True,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
    ):
        """Initialize Agent with device connection and OCR engine.

        Args:
            device_url: Device connection URL for Airtest. If None, uses DEFAULT_DEVICE_URL.
            enable_retry: Enable automatic retry on connection failure.
            auto_connect: Automatically connect to device on initialization.
            connection_timeout: Timeout for connection attempts in seconds.

        Raises:
            RuntimeError: If agent initialization or device connection fails.
        """
        self.logger = get_logger(__name__)
        self.device = None
        self.ocr_engine: Optional[EnhancedOcrEngine] = None
        self._device_verified = False
        self.device_url = device_url or DEFAULT_DEVICE_URL
        self.connection_timeout = connection_timeout

        # Thread safety locks
        self._device_lock = threading.RLock()
        self._state_lock = threading.Lock()

        try:
            if auto_connect:
                if enable_retry:
                    if not self.connect_device_with_retry(self.device_url):
                        raise RuntimeError(
                            f"Cannot connect to device: {self.device_url}"
                        )
                else:
                    with self._device_lock:
                        self.device = connect_device(self.device_url)
                    self.logger.info(f"Connected to device: {self.device_url}")
                    if not self._verify_device():
                        self.logger.warning("Device connected but verification failed")
                    self._init_ocr_engine()

        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def _init_ocr_engine(self) -> None:
        """Initialize OCR engine lazily when device is connected.

        This method is called when the device connection is established,
        ensuring OCR resources are only loaded when actually needed.
        Thread-safe: uses state lock to prevent double initialization.
        """
        with self._state_lock:
            if self.ocr_engine is not None:
                return
            try:
                self.ocr_engine = EnhancedOcrEngine(timeout=DEFAULT_OCR_TIMEOUT)
                self.logger.info("OCR engine initialized")
            except Exception as e:
                self.logger.warning(f"OCR engine initialization failed: {e}")

    def connect_device_with_retry(
        self,
        device_url: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> bool:
        """Connect to device with automatic retry on failure.

        Args:
            device_url: Device connection URL for Airtest. If None, uses instance URL.
            max_retries: Maximum number of retry attempts (default: 3).
            retry_delay: Delay between retries in seconds (default: 1.0).

        Returns:
            bool: True if connection successful and verified, False otherwise.

        Thread Safety:
            Uses device lock to prevent concurrent connection attempts.
        """
        url = device_url or self.device_url

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Connecting to device (attempt {attempt + 1}/{max_retries})..."
                )

                with self._device_lock:
                    self.device = connect_device(url)

                if self._verify_device():
                    self.logger.info("Device connected and verified")
                    self._init_ocr_engine()
                    return True

                if attempt < max_retries - 1:
                    self.logger.warning("Verification failed, retrying...")
                    time.sleep(retry_delay)

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Connection failed: {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Connection failed: {e}")

        self.logger.error(f"Failed to connect after {max_retries} attempts")
        return False

    def _verify_device(self) -> bool:
        """Verify device is working properly by checking snapshot capability.

        Returns:
            bool: True if device is functional and can take screenshots, False otherwise.

        Thread Safety:
            Uses state lock for _device_verified flag.
        """
        try:
            with self._device_lock:
                if self.device and hasattr(self.device, "snapshot"):
                    with self._state_lock:
                        self._device_verified = True
                    return True
            return False
        except (AirtestError, Exception):
            return False

    def is_device_connected(self) -> bool:
        """Check if device is connected and functional.

        Returns:
            bool: True if device is connected and verified, False otherwise.

        Thread Safety:
            Uses locks for thread-safe state access.
        """
        with self._device_lock:
            if not self.device:
                with self._state_lock:
                    self._device_verified = False
                return False

        with self._state_lock:
            if self._device_verified:
                return True

        return self._verify_device()

    def snapshot(self) -> Optional[np.ndarray]:
        """Take a screenshot of the connected device.

        Returns:
            Screenshot as numpy array (BGR format) or None if failed.

        Thread Safety:
            Uses device lock for thread-safe snapshot.
        """
        with self._device_lock:
            if not self.device:
                self.logger.error("Cannot take snapshot: No device connected")
                return None

            try:
                return self.device.snapshot()
            except Exception as e:
                self.logger.error(f"Snapshot failed: {e}")
                return None

    def snapshot_region(
        self, region: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Capture screenshot of specific region.

        Args:
            region: Region coordinates as (x1, y1, x2, y2) in pixels.

        Returns:
            Optional[np.ndarray]: Cropped screenshot as NumPy array, or None if failed.
        """
        x1, y1, x2, y2 = region
        if x1 >= x2 or y1 >= y2:
            self.logger.error(f"Invalid region: {region}")
            return None

        full_screenshot = self.snapshot()
        if full_screenshot is None:
            return None

        return full_screenshot[y1:y2, x1:x2]

    def ocr(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[dict]:
        """Perform OCR on full screen or specified region.

        Args:
            region: Optional region coordinates as (x1, y1, x2, y2). If None, performs OCR on full screen.

        Returns:
            Optional[dict]: OCR result dictionary containing text and bounding boxes, or None if failed.
        """
        if not self.ocr_engine:
            self.logger.error("OCR engine not initialized")
            return None

        image_array = self.snapshot_region(region) if region else self.snapshot()
        if image_array is None:
            self.logger.error("Failed to get screenshot for OCR")
            return None

        try:
            return self.ocr_engine.recognize(image_array)
        except Exception as e:
            self.logger.error(f"OCR recognition failed: {e}")
            return None

    def safe_touch(
        self,
        pos: Union[Tuple[float, float], List[float]],
        times: int = DEFAULT_TOUCH_TIMES,
    ) -> bool:
        """Safely touch screen at position with error handling.

        Args:
            pos: Touch position as (x, y) tuple or list of two floats.
            times: Number of times to touch (default: 1).

        Returns:
            bool: True if touch successful, False otherwise.

        Thread Safety:
            Uses device lock for thread-safe touch operation.
        """
        # Validate and convert coordinates
        try:
            if isinstance(pos, (list, tuple)):
                if len(pos) != 2:
                    self.logger.error(f"Invalid coordinates length: {pos}")
                    return False
                x, y = float(pos[0]), float(pos[1])
                # Basic bounds validation
                if x < 0 or y < 0:
                    self.logger.error(f"Negative coordinates: ({x}, {y})")
                    return False
                pos = (x, y)
            else:
                self.logger.error(f"Invalid coordinates type: {type(pos)}")
                return False
        except (TypeError, ValueError) as e:
            self.logger.error(f"Invalid coordinates: {pos}, error: {e}")
            return False

        with self._device_lock:
            try:
                touch(pos, times=times)
                return True
            except Exception as e:
                self.logger.error(f"Touch failed at {pos}: {e}")
                with self._state_lock:
                    self._device_verified = False
                return False

    def safe_swipe(
        self,
        v1: Union[Tuple[float, float], List[float]],
        v2: Union[Tuple[float, float], List[float]],
        vector: Optional[Union[Tuple[float, float], List[float]]] = None,
        duration: float = 0.5,
        steps: int = 5,
        fingers: int = 1,
    ) -> bool:
        """Safely swipe from v1 to v2 or along vector.

        Args:
            v1: Start point (x, y).
            v2: End point (x, y).
            vector: Vector (x, y) to swipe along (alternative to v2).
            duration: Duration of swipe in seconds.
            steps: Number of steps for swipe interpolation.
            fingers: Number of fingers to use.

        Returns:
            bool: True if swipe successful, False otherwise.

        Thread Safety:
            Uses device lock for thread-safe swipe operation.
        """
        with self._device_lock:
            try:
                swipe(
                    v1,
                    v2=v2,
                    vector=vector,
                    duration=duration,
                    steps=steps,
                    fingers=fingers,
                )
                return True
            except Exception as e:
                self.logger.error(f"Swipe failed: {e}")
                with self._state_lock:
                    self._device_verified = False
                return False

    def cleanup(self) -> None:
        """Clean up resources held by the agent.

        Should be called when the agent is no longer needed to release
        device connections and OCR engine resources.
        """
        with self._device_lock:
            self.device = None

        with self._state_lock:
            self._device_verified = False
            if self.ocr_engine is not None:
                # OCR engine cleanup if needed
                self.ocr_engine = None

        self.logger.info("Agent resources cleaned up")
