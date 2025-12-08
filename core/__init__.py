"""Core package for automation framework."""

from .detector import (
    YOLO_AVAILABLE,
    DetectionResult,
    OCRTextProcessor,
    TemplateMatcher,
    TextProcessor,
)

__all__ = [
    "DetectionResult",
    "TemplateMatcher",
    "TextProcessor",
    "OCRTextProcessor",
    "YOLO_AVAILABLE",
]
