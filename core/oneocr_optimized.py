import copy
import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char,
    c_char_p,
    c_float,
    c_int32,
    c_int64,
    c_ubyte,
)
from threading import Lock
from typing import Any, Dict, Optional, Union

from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "oneocr")
MODEL_NAME = "oneocr.onemodel"
DLL_NAME = "oneocr.dll"
MODEL_KEY = b'kj)TGtrK>f]b[Piow.gU+nC@s""""""4'

# Image size constraints
MIN_IMAGE_SIZE = 50
MAX_IMAGE_SIZE = 10000

c_int64_p = POINTER(c_int64)
c_float_p = POINTER(c_float)
c_ubyte_p = POINTER(c_ubyte)


class ImageStructure(Structure):
    """Image data structure for OCR processing.

    Attributes:
        type: Image type identifier.
        width: Image width in pixels.
        height: Image height in pixels.
        step_size: Number of bytes per row.
        data_ptr: Pointer to image data buffer.
    """

    _fields_ = [
        ("type", c_int32),
        ("width", c_int32),
        ("height", c_int32),
        ("_reserved", c_int32),
        ("step_size", c_int64),
        ("data_ptr", c_ubyte_p),
    ]


class BoundingBox(Structure):
    """Text bounding box coordinates.

    Attributes:
        x1, y1: First corner coordinates.
        x2, y2: Second corner coordinates.
        x3, y3: Third corner coordinates.
        x4, y4: Fourth corner coordinates.
    """

    _fields_ = [
        ("x1", c_float),
        ("y1", c_float),
        ("x2", c_float),
        ("y2", c_float),
        ("x3", c_float),
        ("y3", c_float),
        ("x4", c_float),
        ("y4", c_float),
    ]


BoundingBox_p = POINTER(BoundingBox)

DLL_FUNCTIONS = [
    ("CreateOcrInitOptions", [c_int64_p], c_int64),
    ("OcrInitOptionsSetUseModelDelayLoad", [c_int64, c_char], c_int64),
    ("CreateOcrPipeline", [c_char_p, c_char_p, c_int64, c_int64_p], c_int64),
    ("CreateOcrProcessOptions", [c_int64_p], c_int64),
    ("OcrProcessOptionsSetMaxRecognitionLineCount", [c_int64, c_int64], c_int64),
    ("RunOcrPipeline", [c_int64, POINTER(ImageStructure), c_int64, c_int64_p], c_int64),
    ("GetImageAngle", [c_int64, c_float_p], c_int64),
    ("GetOcrLineCount", [c_int64, c_int64_p], c_int64),
    ("GetOcrLine", [c_int64, c_int64, c_int64_p], c_int64),
    ("GetOcrLineContent", [c_int64, POINTER(c_char_p)], c_int64),
    ("GetOcrLineBoundingBox", [c_int64, POINTER(BoundingBox_p)], c_int64),
    ("GetOcrLineWordCount", [c_int64, c_int64_p], c_int64),
    ("GetOcrWord", [c_int64, c_int64, c_int64_p], c_int64),
    ("GetOcrWordContent", [c_int64, POINTER(c_char_p)], c_int64),
    ("GetOcrWordBoundingBox", [c_int64, POINTER(BoundingBox_p)], c_int64),
    ("GetOcrWordConfidence", [c_int64, c_float_p], c_int64),
    ("ReleaseOcrResult", [c_int64], None),
    ("ReleaseOcrInitOptions", [c_int64], None),
    ("ReleaseOcrPipeline", [c_int64], None),
    ("ReleaseOcrProcessOptions", [c_int64], None),
]


def bind_dll_functions(dll, functions):
    """Dynamically bind function specifications to DLL methods.

    Args:
        dll: The loaded DLL object.
        functions: List of tuples containing (name, argtypes, restype).

    Raises:
        RuntimeError: If a required DLL function is missing.
    """
    for name, argtypes, restype in functions:
        try:
            func = getattr(dll, name)
            func.argtypes = argtypes
            func.restype = restype
        except AttributeError as e:
            raise RuntimeError(f"Missing DLL function: {name}") from e


# Initialize DLL once at module level
try:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if hasattr(kernel32, "SetDllDirectoryW"):
        kernel32.SetDllDirectoryW(CONFIG_DIR)

    dll_path = os.path.join(CONFIG_DIR, DLL_NAME)
    ocr_dll = ctypes.WinDLL(dll_path)
    bind_dll_functions(ocr_dll, DLL_FUNCTIONS)
except (OSError, RuntimeError) as e:
    logger.error(f"DLL initialization failed: {e}")
    sys.exit(f"DLL initialization failed: {e}")


@contextmanager
def suppress_output():
    """Suppress stdout and stderr output.

    Context manager that redirects stdout and stderr to devnull,
    restoring them after the context exits.

    Yields:
        None
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(original_stdout)
        os.close(original_stderr)
        os.close(devnull)


class OcrEngine:
    """Thread-safe OCR engine with proper resource management.

    This class provides a high-performance OCR engine that uses OneOCR DLL
    for text recognition. It supports both PIL Image and OpenCV image formats,
    with thread-safe operations and automatic resource cleanup.

    Attributes:
        init_options: OCR initialization options handle.
        pipeline: OCR pipeline handle.
        process_options: OCR processing options handle.
    """

    _EMPTY_RESULT = {"text": "", "text_angle": None, "lines": []}

    def __init__(self):
        self._lock = Lock()  # Thread safety
        self._initialized = False
        self.init_options = None
        self.pipeline = None
        self.process_options = None

        # Pre-allocate reusable C types to avoid repeated allocations
        self._c_int64_pool = c_int64()
        self._c_float_pool = c_float()
        self._c_char_p_pool = c_char_p()
        self._bbox_ptr_pool = BoundingBox_p()
        self._cv2_module = None

        # Cache for small image conversion (avoid repeated allocations)
        self._small_image_threshold = 1_000_000
        self._bgra_buffer = None

        try:
            self._initialize()
            self._initialized = True
        except Exception as e:
            logger.error(f"OCR engine initialization failed: {e}")
            self.cleanup()
            raise

    def _initialize(self):
        """Initialize OCR resources.

        Creates initialization options, pipeline, and process options
        required for OCR operations.
        """
        self.init_options = self._create_init_options()
        self.pipeline = self._create_pipeline()
        self.process_options = self._create_process_options()

    def cleanup(self):
        """Release all OCR resources.

        Explicitly releases process options, pipeline, and initialization
        options to free memory and DLL resources.
        """
        if self.process_options:
            ocr_dll.ReleaseOcrProcessOptions(self.process_options)
            self.process_options = None
        if self.pipeline:
            ocr_dll.ReleaseOcrPipeline(self.pipeline)
            self.pipeline = None
        if self.init_options:
            ocr_dll.ReleaseOcrInitOptions(self.init_options)
            self.init_options = None
        self._initialized = False

    def __del__(self):
        """Destructor for fallback resource cleanup.

        Attempts to clean up resources if not already done.
        Suppresses all exceptions to avoid issues during garbage collection.
        """
        try:
            self.cleanup()
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def _create_init_options(self) -> c_int64:
        """Create OCR initialization options.

        Returns:
            Handle to the created initialization options.

        Raises:
            RuntimeError: If initialization options creation fails.
        """
        init_options = c_int64()
        self._check_dll_result(
            ocr_dll.CreateOcrInitOptions(byref(init_options)),
            "Init options creation failed",
        )

        self._check_dll_result(
            ocr_dll.OcrInitOptionsSetUseModelDelayLoad(init_options, 0),
            "Model loading config failed",
        )
        return init_options

    def _create_pipeline(self) -> c_int64:
        """Create OCR processing pipeline.

        Returns:
            Handle to the created OCR pipeline.

        Raises:
            RuntimeError: If pipeline creation fails.
        """
        model_path = os.path.join(CONFIG_DIR, MODEL_NAME)
        model_buf = ctypes.create_string_buffer(model_path.encode())
        key_buf = ctypes.create_string_buffer(MODEL_KEY)

        pipeline = c_int64()
        with suppress_output():
            self._check_dll_result(
                ocr_dll.CreateOcrPipeline(
                    model_buf, key_buf, self.init_options, byref(pipeline)
                ),
                "Pipeline creation failed",
            )
        return pipeline

    def _create_process_options(self) -> c_int64:
        """Create OCR process options.

        Returns:
            Handle to the created process options.

        Raises:
            RuntimeError: If process options creation fails.
        """
        process_options = c_int64()
        self._check_dll_result(
            ocr_dll.CreateOcrProcessOptions(byref(process_options)),
            "Process options creation failed",
        )

        self._check_dll_result(
            ocr_dll.OcrProcessOptionsSetMaxRecognitionLineCount(process_options, 1000),
            "Line count config failed",
        )
        return process_options

    @staticmethod
    def _validate_image_size(width: int, height: int) -> Optional[str]:
        """Validate image dimensions against size constraints.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Error message if validation fails, None otherwise.
        """
        if width < MIN_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
            return f"Invalid width: {width} (must be {MIN_IMAGE_SIZE}-{MAX_IMAGE_SIZE})"
        if height < MIN_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            return (
                f"Invalid height: {height} (must be {MIN_IMAGE_SIZE}-{MAX_IMAGE_SIZE})"
            )
        return None

    def recognize_pil(self, image: Image.Image) -> Dict[str, Any]:
        """Perform OCR on a PIL Image object.

        Args:
            image: PIL Image object to process.

        Returns:
            Dictionary containing:
                - text: Recognized text with lines separated by newlines.
                - text_angle: Detected text angle in degrees, or None.
                - lines: List of line dictionaries with text, bounding boxes, and words.
                - error: Error message if processing failed (optional).
        """
        if not self._initialized:
            return self._error_result("OCR engine not initialized")

        width, height = image.width, image.height
        error = self._validate_image_size(width, height)
        if error:
            return self._error_result(error)

        # Avoid unnecessary conversion if already RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        step = width * 4
        rgba_bytes = image.tobytes()
        byte_size = len(rgba_bytes)

        # Optimized BGRA conversion for small images using in-place operations
        if byte_size < self._small_image_threshold:
            # Reuse buffer if possible
            if self._bgra_buffer is None or len(self._bgra_buffer) < byte_size:
                self._bgra_buffer = bytearray(byte_size)
            else:
                # Reuse existing buffer
                bgra_bytes = self._bgra_buffer[:byte_size]
                bgra_bytes[:] = rgba_bytes
                bgra_bytes[0::4], bgra_bytes[2::4] = bgra_bytes[2::4], bgra_bytes[0::4]
                data = bytes(bgra_bytes)

                with self._lock:
                    return self._process_image(
                        cols=width, rows=height, step=step, data=data
                    )

            bgra_bytes = self._bgra_buffer[:byte_size]
            bgra_bytes[:] = rgba_bytes
            bgra_bytes[0::4], bgra_bytes[2::4] = bgra_bytes[2::4], bgra_bytes[0::4]
            data = bytes(bgra_bytes)
        else:
            # For large images, use PIL split/merge (more memory efficient)
            b, g, r, a = image.split()
            bgra_image = Image.merge("RGBA", (b, g, r, a))
            data = bgra_image.tobytes()

        with self._lock:
            return self._process_image(cols=width, rows=height, step=step, data=data)

    def recognize_cv2(self, image_buffer) -> Dict[str, Any]:
        """Perform OCR on an OpenCV image buffer.

        Args:
            image_buffer: Numpy array containing encoded image data.

        Returns:
            Dictionary containing:
                - text: Recognized text with lines separated by newlines.
                - text_angle: Detected text angle in degrees, or None.
                - lines: List of line dictionaries with text, bounding boxes, and words.
                - error: Error message if processing failed (optional).
        """
        if not self._initialized:
            return self._error_result("OCR engine not initialized")

        if self._cv2_module is None:
            try:
                import cv2

                self._cv2_module = cv2
            except ImportError:
                return self._error_result("OpenCV not installed")

        cv2 = self._cv2_module

        img = cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED)
        if img is None:
            return self._error_result("Failed to decode image")

        height, width = img.shape[:2]
        error = self._validate_image_size(width, height)
        if error:
            return self._error_result(error)

        channels = img.shape[2] if img.ndim == 3 else 1
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        with self._lock:  # Thread safety
            return self._process_image(
                cols=width,
                rows=height,
                step=width * 4,
                data=img.ctypes.data,
            )

    def _process_image(
        self, cols: int, rows: int, step: int, data: Union[bytes, int]
    ) -> Dict[str, Any]:
        """Create image structure and perform OCR processing.

        Args:
            cols: Image width in pixels.
            rows: Image height in pixels.
            step: Number of bytes per row.
            data: Image data as bytes or pointer address (int).

        Returns:
            OCR result dictionary.
        """
        if isinstance(data, bytes):
            data_ptr = (c_ubyte * len(data)).from_buffer_copy(data)
        else:
            data_ptr = ctypes.cast(ctypes.c_void_p(data), c_ubyte_p)

        img_struct = ImageStructure(
            type=3,
            width=cols,
            height=rows,
            _reserved=0,
            step_size=step,
            data_ptr=data_ptr,
        )

        return self._perform_ocr(img_struct)

    def _perform_ocr(self, image_struct: ImageStructure) -> Dict[str, Any]:
        """Execute OCR pipeline and parse results.

        Args:
            image_struct: ImageStructure containing image data.

        Returns:
            Parsed OCR results dictionary.
        """
        ocr_result = c_int64()
        result_code = ocr_dll.RunOcrPipeline(
            self.pipeline,
            byref(image_struct),
            self.process_options,
            byref(ocr_result),
        )

        if result_code != 0:
            logger.warning(f"OCR pipeline failed with code: {result_code}")
            return copy.deepcopy(self._EMPTY_RESULT)

        try:
            parsed_result = self._parse_ocr_results(ocr_result)
        finally:
            ocr_dll.ReleaseOcrResult(ocr_result)

        return parsed_result

    def _parse_ocr_results(self, ocr_result: c_int64) -> Dict[str, Any]:
        """Extract and format OCR results from DLL.

        Args:
            ocr_result: Handle to OCR result from DLL.

        Returns:
            Formatted dictionary with text, angle, and line information.
        """
        if ocr_dll.GetOcrLineCount(ocr_result, byref(self._c_int64_pool)) != 0:
            return copy.deepcopy(self._EMPTY_RESULT)

        line_count = self._c_int64_pool.value

        if line_count == 0:
            return copy.deepcopy(self._EMPTY_RESULT)

        lines = self._get_lines(ocr_result, line_count)

        # Optimize text joining - avoid intermediate list if possible
        if line_count == 1:
            text = lines[0]["text"] or ""
        else:
            text_parts = []
            for line in lines:
                if line["text"]:
                    text_parts.append(line["text"])
            text = "\n".join(text_parts) if text_parts else ""

        return {
            "text": text,
            "text_angle": self._get_text_angle(ocr_result),
            "lines": lines,
        }

    def _get_text_angle(self, ocr_result: c_int64) -> Optional[float]:
        """Extract text angle from OCR result.

        Args:
            ocr_result: Handle to OCR result.

        Returns:
            Text angle in degrees, or None if unavailable.
        """
        if ocr_dll.GetImageAngle(ocr_result, byref(self._c_float_pool)) != 0:
            return None
        return self._c_float_pool.value

    def _get_lines(self, ocr_result: c_int64, line_count: int) -> list:
        """Extract individual text lines from OCR result.

        Args:
            ocr_result: Handle to OCR result.
            line_count: Number of lines to extract.

        Returns:
            List of line dictionaries.
        """
        # Pre-allocate list for better performance
        lines = [None] * line_count
        for idx in range(line_count):
            lines[idx] = self._process_line(ocr_result, idx)
        return lines

    def _process_line(self, ocr_result: c_int64, line_index: int) -> Dict[str, Any]:
        """Process a single text line.

        Args:
            ocr_result: Handle to OCR result.
            line_index: Index of the line to process.

        Returns:
            Dictionary with text, bounding_rect, and words.
        """
        line_handle = c_int64()
        if ocr_dll.GetOcrLine(ocr_result, line_index, byref(line_handle)) != 0:
            return {"text": None, "bounding_rect": None, "words": []}

        return {
            "text": self._get_text(line_handle, ocr_dll.GetOcrLineContent),
            "bounding_rect": self._get_bounding_box(
                line_handle, ocr_dll.GetOcrLineBoundingBox
            ),
            "words": self._get_words(line_handle),
        }

    def _get_words(self, line_handle: c_int64) -> list:
        """Extract words from a text line.

        Args:
            line_handle: Handle to the text line.

        Returns:
            List of word dictionaries.
        """
        if ocr_dll.GetOcrLineWordCount(line_handle, byref(self._c_int64_pool)) != 0:
            return []

        word_count = self._c_int64_pool.value
        if word_count == 0:
            return []

        # Pre-allocate list for better performance
        words = [None] * word_count
        for idx in range(word_count):
            words[idx] = self._process_word(line_handle, idx)
        return words

    def _process_word(self, line_handle: c_int64, word_index: int) -> Dict[str, Any]:
        """Process an individual word.

        Args:
            line_handle: Handle to the parent text line.
            word_index: Index of the word to process.

        Returns:
            Dictionary with text, bounding_rect, and confidence.
        """
        word_handle = c_int64()
        if ocr_dll.GetOcrWord(line_handle, word_index, byref(word_handle)) != 0:
            return {"text": None, "bounding_rect": None, "confidence": None}

        return {
            "text": self._get_text(word_handle, ocr_dll.GetOcrWordContent),
            "bounding_rect": self._get_bounding_box(
                word_handle, ocr_dll.GetOcrWordBoundingBox
            ),
            "confidence": self._get_word_confidence(word_handle),
        }

    def _get_text(self, handle: c_int64, text_function) -> Optional[str]:
        """Extract text content from a handle.

        Args:
            handle: Handle to text object (line or word).
            text_function: DLL function to extract text.

        Returns:
            Decoded text string, or None if unavailable.
        """
        if (
            text_function(handle, byref(self._c_char_p_pool)) == 0
            and self._c_char_p_pool.value
        ):
            return self._c_char_p_pool.value.decode("utf-8", errors="ignore")
        return None

    def _get_bounding_box(
        self, handle: c_int64, bbox_function
    ) -> Optional[Dict[str, float]]:
        """Extract bounding box coordinates from a handle.

        Args:
            handle: Handle to text object (line or word).
            bbox_function: DLL function to extract bounding box.

        Returns:
            Dictionary with x1-x4, y1-y4 coordinates, or None if unavailable.
        """
        if (
            bbox_function(handle, byref(self._bbox_ptr_pool)) != 0
            or not self._bbox_ptr_pool
        ):
            return None

        bbox = self._bbox_ptr_pool.contents
        # Direct assignment is faster than dict literal for small dicts
        return {
            "x1": bbox.x1,
            "y1": bbox.y1,
            "x2": bbox.x2,
            "y2": bbox.y2,
            "x3": bbox.x3,
            "y3": bbox.y3,
            "x4": bbox.x4,
            "y4": bbox.y4,
        }

    def _get_word_confidence(self, word_handle: c_int64) -> Optional[float]:
        """Extract confidence score from a word handle.

        Args:
            word_handle: Handle to the word object.

        Returns:
            Confidence score (0.0-1.0), or None if unavailable.
        """
        if ocr_dll.GetOcrWordConfidence(word_handle, byref(self._c_float_pool)) == 0:
            return self._c_float_pool.value
        return None

    @staticmethod
    def _check_dll_result(result_code: int, error_message: str):
        """Check DLL function result code and raise exception on error.

        Args:
            result_code: Return code from DLL function.
            error_message: Error message to include in exception.

        Raises:
            RuntimeError: If result_code is non-zero.
        """
        if result_code != 0:
            raise RuntimeError(f"{error_message} (Code: {result_code})")

    @classmethod
    def _error_result(cls, error_msg: str) -> Dict[str, Any]:
        """Create an error result dictionary.

        Args:
            error_msg: Error message to include.

        Returns:
            Dictionary with empty OCR results and error message.
        """
        result = copy.deepcopy(cls._EMPTY_RESULT)
        result["error"] = error_msg
        logger.error(error_msg)
        return result


def serve(host: str = "0.0.0.0", port: int = 8001, workers: int = 1):
    """Initialize and run the OCR web service.

    Creates a FastAPI application that exposes OCR functionality via HTTP POST.
    The service accepts image data in the request body and returns OCR results
    as JSON.

    Args:
        host: Host address to bind the server. Defaults to "0.0.0.0".
        port: Port number to listen on. Defaults to 8001.
        workers: Number of worker processes. Defaults to 1.

    Example:
        >>> serve(host="127.0.0.1", port=8080)

    Note:
        The service accepts POST requests with image data in the body.
        Supported formats: JPEG, PNG, BMP, and other PIL-compatible formats.
    """
    import json
    from io import BytesIO

    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="OneOCR Service",
        version="1.0.0",
        description="High-performance OCR service using OneOCR engine",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ocr_processor = OcrEngine()

    @app.post("/ocr", response_class=JSONResponse)
    async def process_image(request: Request):
        """Process image and return OCR results.

        Args:
            request: FastAPI request containing image data in body.

        Returns:
            JSON response with OCR results.

        Raises:
            HTTPException: If image data is empty or processing fails.
        """
        try:
            image_data = await request.body()
            if not image_data:
                raise HTTPException(status_code=400, detail="Empty image data")

            image = Image.open(BytesIO(image_data))
            result = ocr_processor.recognize_pil(image)

            if "error" in result:
                raise HTTPException(status_code=422, detail=result["error"])

            return JSONResponse(
                content=result,
                media_type="application/json; charset=utf-8",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OCR processing error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint.

        Returns:
            Status information about the service.
        """
        return {"status": "healthy", "service": "OneOCR"}

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup resources on server shutdown."""
        ocr_processor.cleanup()
        logger.info("OCR service shutdown complete")

    logger.info(f"Starting OCR service on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")


if __name__ == "__main__":
    serve()
