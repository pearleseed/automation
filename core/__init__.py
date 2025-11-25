"""Core package for automation framework."""

from .detector import (
    DetectionResult,
    TemplateMatcher,
    TextProcessor,
    OCRTextProcessor,
    YOLO_AVAILABLE,
)

__all__ = [
    "DetectionResult",
    "TemplateMatcher",
    "TextProcessor",
    "OCRTextProcessor",
    "YOLO_AVAILABLE",
]
