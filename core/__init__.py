"""Core package for automation framework."""

from .detector import DetectionResult, OCRTextProcessor, TemplateMatcher, TextProcessor

__all__ = [
    "DetectionResult",
    "TemplateMatcher",
    "TextProcessor",
    "OCRTextProcessor",
]
