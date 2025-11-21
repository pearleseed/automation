"""
Item Detector Module - YOLO and Template Matching for automated detection.

This module provides three main components:
1. YOLODetector - AI-based object detection with YOLO
2. TemplateMatcher - Template-based detection with OpenCV
3. OCRTextProcessor - Advanced OCR text processing and validation

"""

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .agent import Agent
from .utils import get_logger

logger = get_logger(__name__)

# Conditional YOLO import
try:
    import torch
    from ultralytics import YOLO  # type: ignore

    YOLO_AVAILABLE = True
    logger.info("YOLO available")
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None  # type: ignore
    torch = None  # type: ignore
    logger.warning("YOLO not available. Install: pip install ultralytics torch")


# ==================== DATA CLASSES ====================


@dataclass
class DetectionResult:
    """Result from YOLO or Template detection."""

    item: str
    quantity: int = 0
    x: int = 0
    y: int = 0
    x2: int = 0
    y2: int = 0
    center_x: int = 0
    center_y: int = 0
    confidence: float = 0.0
    ocr_text: str = ""

    def __post_init__(self):
        """Calculate center if not provided."""
        if self.center_x == 0 and self.center_y == 0:
            self.center_x = (self.x + self.x2) // 2
            self.center_y = (self.y + self.y2) // 2


@dataclass
class ExtractionResult:
    """Result from field extraction."""

    value: Any
    raw_text: str
    confidence: float
    success: bool
    error_message: str = ""


@dataclass
class ValidationResult:
    """Result from field validation."""

    field: str
    status: str  # 'match', 'mismatch', 'error', 'missing'
    extracted: Any
    expected: Any
    ocr_text: str
    message: str
    confidence: float = 0.0


@dataclass
class ValidationSummary:
    """Summary of validation results."""

    total: int
    matched: int
    mismatched: int
    missing: int
    errors: int
    match_rate: float
    status: str  # 'pass' or 'fail'


# ==================== TEXT PROCESSING UTILITIES ====================


class TextProcessor:
    """Unified text processing utilities for OCR text normalization and comparison.

    This class provides static methods for cleaning, normalizing, and comparing text
    extracted from OCR operations. All methods are cached for performance.
    """

    # Pre-compiled patterns for performance
    _NUMBER_PATTERN = re.compile(r"\d+")
    _CLEAN_PUNCTUATION = str.maketrans("", "", ",.")

    @staticmethod
    @lru_cache(maxsize=1024)
    def normalize_text(
        text: str, remove_spaces: bool = True, lowercase: bool = True
    ) -> str:
        """
        Normalize text for comparison with caching.

        Args:
            text: Text to normalize
            remove_spaces: Remove all spaces
            lowercase: Convert to lowercase

        Returns:
            str: Normalized text
        """
        if not text:
            return ""

        result = text.strip()

        if lowercase:
            result = result.lower()

        if remove_spaces:
            result = result.replace(" ", "").replace("\u3000", "")

        # Remove common punctuation (optimized with translate)
        result = result.translate(TextProcessor._CLEAN_PUNCTUATION)

        return result

    @staticmethod
    def clean_ocr_artifacts(text: str) -> str:
        """
        Clean common OCR artifacts.

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Common OCR mistakes
        replacements = {
            "o": "0",
            "O": "0",  # Letter O to zero
            "l": "1",
            "I": "1",  # Letter I/l to one
        }

        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)

        return result.strip()

    @staticmethod
    def extract_numbers(
        text: str, clean_chars: Optional[List[str]] = None
    ) -> List[int]:
        """
        Extract all numbers from text.

        Args:
            text: Text to extract from
            clean_chars: Characters to remove before extraction

        Returns:
            List[int]: List of extracted numbers
        """
        if not text:
            return []

        # Clean text (optimized)
        cleaned = text.strip()
        if clean_chars:
            trans_table = str.maketrans("", "", "".join(clean_chars))
            cleaned = cleaned.translate(trans_table)
        else:
            cleaned = cleaned.replace(",", "").replace(" ", "")

        # Extract numbers (use cached pattern)
        numbers = TextProcessor._NUMBER_PATTERN.findall(cleaned)
        try:
            return [int(n) for n in numbers]
        except ValueError:
            return []

    @staticmethod
    def get_number_at_position(
        text: str, position: int = 0, clean_chars: Optional[List[str]] = None
    ) -> Optional[int]:
        """
        Get number at specific position in text.

        Args:
            text: Text to extract from
            position: Position in numbers list (0=first, -1=last)
            clean_chars: Characters to remove before extraction

        Returns:
            Optional[int]: Number at position or None
        """
        numbers = TextProcessor.extract_numbers(text, clean_chars)
        if not numbers:
            return None

        try:
            idx = position if position >= 0 else len(numbers) + position
            return numbers[idx]
        except IndexError:
            return None

    @staticmethod
    @lru_cache(maxsize=512)
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (cached).

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity ratio (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0

        return SequenceMatcher(None, text1, text2).ratio()

    @staticmethod
    def fuzzy_match(text: str, template: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy match text against template.

        Args:
            text: Text to match
            template: Template to match against
            threshold: Similarity threshold

        Returns:
            bool: True if match
        """
        if not text or not template:
            return False

        # Normalize both
        text_norm = TextProcessor.normalize_text(text)
        template_norm = TextProcessor.normalize_text(template)

        # Exact match
        if text_norm == template_norm:
            return True

        # Substring match
        if text_norm in template_norm or template_norm in text_norm:
            return True

        # Similarity match
        if len(text_norm) == 0 or len(template_norm) == 0:
            return False

        similarity = TextProcessor.calculate_similarity(text_norm, template_norm)
        return similarity >= threshold


# ==================== FIELD EXTRACTORS (STRATEGY PATTERN) ====================


class FieldExtractor(Protocol):
    """Protocol for field extractors."""

    def extract(self, text: str) -> ExtractionResult:
        """Extract field value from text."""
        ...


class NumberExtractor:
    """Extract number from text."""

    def __init__(self, position: int = 0, clean_chars: Optional[List[str]] = None):
        self.position, self.clean_chars = position, clean_chars

    def extract(self, text: str) -> ExtractionResult:
        value = TextProcessor.get_number_at_position(
            text, self.position, self.clean_chars
        )
        return ExtractionResult(
            value, text, 0.9 if value is not None else 0.0, value is not None
        )


class RankExtractor:
    """Extract rank letter from text."""

    RANK_PATTERN = re.compile(r"\b(SSS+|SSS|SS|S|A|B|C|D|E|F)\b")

    def extract(self, text: str) -> ExtractionResult:
        if not text:
            return ExtractionResult(None, text, 0.0, False)
        match = self.RANK_PATTERN.search(text.strip().upper())
        return (
            ExtractionResult(match.group(1), text, 1.0, True)
            if match
            else ExtractionResult(None, text, 0.0, False)
        )


class MoneyExtractor:
    """Extract money/currency from text."""

    NUMBER_PATTERN = re.compile(r"\d+")
    CLEAN_CHARS = str.maketrans("", "", ",× xX ")

    def extract(self, text: str) -> ExtractionResult:
        if not text:
            return ExtractionResult(None, text, 0.0, False)
        numbers = self.NUMBER_PATTERN.findall(text.strip().translate(self.CLEAN_CHARS))
        try:
            return (
                ExtractionResult(int("".join(numbers)), text, 0.9, True)
                if numbers
                else ExtractionResult(None, text, 0.0, False)
            )
        except ValueError:
            return ExtractionResult(None, text, 0.0, False)


class ItemQuantityExtractor:
    """Extract item name and quantity."""

    PATTERN = re.compile(r"(.+?)\s*[xX×]\s*(\d+)")
    FALLBACK_PATTERN = re.compile(r"\s*[xX×]?\s*\d+\s*$")
    NUMBER_PATTERN = re.compile(r"\d+")

    def extract(self, text: str) -> ExtractionResult:
        if not text:
            return ExtractionResult((None, None), text, 0.0, False)
        text = text.strip()

        # Try pattern match
        if match := self.PATTERN.search(text):
            try:
                return ExtractionResult(
                    (match.group(1).strip(), int(match.group(2))), text, 0.9, True
                )
            except ValueError:
                pass

        # Fallback: extract numbers at end
        if numbers := self.NUMBER_PATTERN.findall(text):
            try:
                return ExtractionResult(
                    (self.FALLBACK_PATTERN.sub("", text).strip(), int(numbers[-1])),
                    text,
                    0.7,
                    True,
                )
            except ValueError:
                pass

        return ExtractionResult((text, None), text, 0.5, True)


class DropRangeExtractor:
    """Extract drop range (e.g., '3 ~ 4')."""

    PATTERN = re.compile(r"(\d+)\s*[~～\-]\s*(\d+)")
    NUMBER_PATTERN = re.compile(r"\d+")

    def extract(self, text: str) -> ExtractionResult:
        if not text:
            return ExtractionResult(None, text, 0.0, False)
        text = text.strip()

        # Try range pattern
        if match := self.PATTERN.search(text):
            try:
                return ExtractionResult(
                    (int(match.group(1)), int(match.group(2))), text, 0.9, True
                )
            except ValueError:
                pass

        # Single number (range = same number)
        if numbers := self.NUMBER_PATTERN.findall(text):
            try:
                val = int(numbers[0])
                return ExtractionResult((val, val), text, 0.8, True)
            except ValueError:
                pass

        return ExtractionResult(None, text, 0.0, False)


# ==================== OCR TEXT PROCESSOR ====================


class OCRTextProcessor:
    """Advanced OCR text processor with strategy pattern for field extraction.

    This class uses specialized extractors to parse different types of fields
    (numbers, ranks, money, items, etc.) from OCR text and validate them against
    expected values.
    """

    # Field extractors registry
    EXTRACTORS: Dict[str, FieldExtractor] = {
        "勝利点数": NumberExtractor(position=0),
        "推奨ランク": RankExtractor(),
        "Sランクボーダー": NumberExtractor(position=-1),
        "消費FP": NumberExtractor(position=0),
        "獲得ザックマネー": MoneyExtractor(),
        "Ｚマネー": MoneyExtractor(),
        "獲得EXP-Ace": NumberExtractor(position=-1),
        "獲得EXP-NonAce": NumberExtractor(position=-1),
        "エース": NumberExtractor(position=-1),
        "非エース": NumberExtractor(position=-1),
        "獲得アイテム": ItemQuantityExtractor(),
        "drop_range": DropRangeExtractor(),
    }

    # Cached extractor instances for internal use
    _drop_range_extractor = DropRangeExtractor()

    @classmethod
    def extract_field(cls, field_name: str, text: str) -> ExtractionResult:
        """Extract field value using appropriate extractor."""
        extractor = cls.EXTRACTORS.get(field_name)
        return (
            extractor.extract(text)
            if extractor
            else ExtractionResult(text, text, 0.5, bool(text))
        )

    @staticmethod
    def validate_field(
        field_name: str, ocr_text: str, expected_value: Any
    ) -> ValidationResult:
        """Validate OCR field against expected value using appropriate extractor.

        Args:
            field_name: Name of the field to validate (e.g., '勝利点数', '推奨ランク').
            ocr_text: Raw OCR text extracted from the field.
            expected_value: Expected value to compare against.

        Returns:
            ValidationResult: Validation result containing status, extracted/expected values,
                            confidence, and detailed message.
        """
        try:
            # Extract value using extractor
            extraction = OCRTextProcessor.extract_field(field_name, ocr_text)

            if not extraction.success:
                return ValidationResult(
                    field_name,
                    "error",
                    None,
                    expected_value,
                    ocr_text,
                    f"Failed to extract: {extraction.error_message}",
                    0.0,
                )

            extracted_value = extraction.value

            # Validate based on field type
            if "報酬" in field_name or "クリア" in field_name:
                match = TextProcessor.fuzzy_match(ocr_text, str(expected_value))
                return ValidationResult(
                    field_name,
                    "match" if match else "mismatch",
                    ocr_text,
                    expected_value,
                    ocr_text,
                    f"Template match: {match}",
                    extraction.confidence,
                )

            if "コイン" in field_name or "ドロップ" in field_name:
                drop_range = OCRTextProcessor._drop_range_extractor.extract(
                    str(expected_value)
                )
                if (
                    drop_range.success
                    and isinstance(drop_range.value, tuple)
                    and isinstance(extracted_value, int)
                ):
                    min_val, max_val = drop_range.value
                    in_range = min_val <= extracted_value <= max_val
                    return ValidationResult(
                        field_name,
                        "match" if in_range else "mismatch",
                        extracted_value,
                        expected_value,
                        ocr_text,
                        f"Drop: {extracted_value} in range [{min_val}, {max_val}] = {bool(in_range)}",
                        extraction.confidence,
                    )

            # Direct comparison - normalize types
            if isinstance(extracted_value, (int, float)):
                try:
                    match = extracted_value == type(extracted_value)(expected_value)
                except (ValueError, TypeError):
                    match = str(extracted_value) == str(expected_value)
            else:
                match = str(extracted_value) == str(expected_value)

            return ValidationResult(
                field_name,
                "match" if match else "mismatch",
                extracted_value,
                expected_value,
                ocr_text,
                f"Comparison: {extracted_value} == {expected_value} = {match}",
                extraction.confidence,
            )

        except Exception as e:
            return ValidationResult(
                field_name,
                "error",
                None,
                expected_value,
                ocr_text,
                f"Validation error: {str(e)}",
                0.0,
            )

    normalize_text_for_comparison = staticmethod(TextProcessor.normalize_text)
    compare_with_template = staticmethod(TextProcessor.fuzzy_match)


# ==================== YOLO DETECTOR ====================


class YOLODetector:
    """YOLO-based item detector with OCR quantity extraction.

    This class uses YOLO object detection to identify items in screenshots
    and extracts quantities using OCR on regions adjacent to detected items.
    """

    def __init__(
        self,
        agent: Agent,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        device: str = "cpu",
    ):
        """Initialize YOLO detector with model and configuration.

        Args:
            agent: Agent instance for OCR operations.
            model_path: Path to YOLO model file (default: 'yolo11n.pt').
            confidence: Detection confidence threshold 0.0-1.0 (default: 0.25).
            device: Device for inference - 'cpu', 'cuda', 'mps', or 'auto' (default: 'cpu').

        Raises:
            RuntimeError: If YOLO is not available or model loading fails.
        """
        self.agent = agent
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model: Optional[Any] = None

        if not YOLO_AVAILABLE:
            logger.error("YOLO not available")
            raise RuntimeError(
                "YOLO not available. Install: pip install ultralytics torch"
            )

        self._init_model()

    def _init_model(self) -> None:
        """Load YOLO model and configure device."""
        try:
            if YOLO is None:
                raise RuntimeError("YOLO not available")

            logger.info(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)

            if self.model is None:
                raise RuntimeError("Failed to load YOLO model")

            if self.device == "auto" and torch is not None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = "mps"
                else:
                    self.device = "cpu"

            logger.info(f"YOLO model loaded on device: {self.device}")

        except Exception as e:
            logger.error(f"YOLO initialization failed: {e}")
            raise

    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> List[DetectionResult]:
        """Detect items in image using YOLO with OCR quantity extraction.

        Args:
            image: Input image as NumPy array (BGR format).
            conf: Confidence threshold (default: uses instance confidence).
            iou: IoU threshold for NMS (default: 0.45).
            imgsz: Input image size for model (default: 640).

        Returns:
            List[DetectionResult]: List of detected items with bounding boxes, confidence, and quantities.
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            return []

        if conf is None:
            conf = self.confidence

        try:
            # Run detection
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

            found_items = []

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    item_name = result.names[class_id]

                    # Extract quantity via OCR
                    quantity, ocr_text = self._extract_quantity(image, (x1, y1, x2, y2))

                    # Use dataclass
                    found_items.append(
                        DetectionResult(
                            item=item_name,
                            quantity=quantity,
                            x=x1,
                            y=y1,
                            x2=x2,
                            y2=y2,
                            confidence=confidence,
                            ocr_text=ocr_text,
                        )
                    )

            logger.info(f"🎯 YOLO detected {len(found_items)} items")
            return found_items

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

    def _extract_quantity(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        offset_x: int = 30,
        offset_y: int = 0,
        roi_width: int = 80,
        roi_height: int = 30,
    ) -> Tuple[int, str]:
        """Extract item quantity using OCR (reuses TextProcessor)."""
        x1, y1, x2, y2 = bbox
        img_h, img_w = image.shape[:2]

        quantity_x1, quantity_y1 = max(0, x2 + offset_x), max(0, y2 + offset_y)
        quantity_x2, quantity_y2 = min(img_w, quantity_x1 + roi_width), min(
            img_h, quantity_y1 + roi_height
        )

        if quantity_x1 >= quantity_x2 or quantity_y1 >= quantity_y2:
            return 0, ""
        quantity_roi = image[quantity_y1:quantity_y2, quantity_x1:quantity_x2]
        if quantity_roi.size == 0:
            return 0, ""

        try:
            if self.agent.ocr_engine is None:
                return 0, ""
            ocr_text = self.agent.ocr_engine.recognize(quantity_roi).get("text", "")
            return self._parse_quantity_text(ocr_text), ocr_text
        except Exception as e:
            logger.debug(f"Quantity OCR error: {e}")
            return 0, ""

    @staticmethod
    def _parse_quantity_text(text: str) -> int:
        """Parse quantity from text (uses TextProcessor)."""
        if not text:
            return 0
        text = text.strip().lstrip("xX").strip()
        numbers = TextProcessor.extract_numbers(text)
        return numbers[0] if numbers else 0


# ==================== TEMPLATE MATCHER ====================


class TemplateMatcher:
    """Template-based item detector."""

    def __init__(
        self,
        templates_dir: str = "templates",
        threshold: float = 0.85,
        method: str = "TM_CCOEFF_NORMED",
    ):
        """Initialize template matcher."""
        self.templates_dir = templates_dir
        self.threshold = threshold
        self.method = method
        self.templates = self._load_templates()

        logger.info(f"TemplateMatcher initialized with {len(self.templates)} templates")

    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Load template images from directory."""
        if not os.path.isdir(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return {}

        templates = {}
        supported_formats = [".png", ".jpg", ".jpeg"]

        for filename in os.listdir(self.templates_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.templates_dir, filename)

                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        templates[name] = img
                        logger.debug(f"Loaded template: {name}")
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {e}")

        return templates

    def detect(
        self, image: np.ndarray, threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """Detect items using template matching with OpenCV.

        Args:
            image: Input image as NumPy array (BGR format).
            threshold: Match threshold 0.0-1.0 (default: uses instance threshold).

        Returns:
            List[DetectionResult]: List of detected items with bounding boxes and positions.
        """
        if threshold is None:
            threshold = self.threshold

        try:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.error(f"Grayscale conversion error: {e}")
            return []

        found_items = []

        for template_name, template in self.templates.items():
            matches = self._find_matches(image_gray, template, threshold)

            for x, y in matches:
                h, w = template.shape
                # Use dataclass
                found_items.append(
                    DetectionResult(
                        item=template_name, quantity=0, x=x, y=y, x2=x + w, y2=y + h
                    )
                )

        # Optimized duplicate removal
        unique_items = self._remove_duplicates_optimized(found_items, min_distance=10)
        logger.info(f"Template matching found {len(unique_items)} items")
        return unique_items

    def _find_matches(
        self, image_gray: np.ndarray, template: np.ndarray, threshold: float
    ) -> List[Tuple[int, int]]:
        """Find template match positions."""
        try:
            method = getattr(cv2, self.method)
            res = cv2.matchTemplate(image_gray, template, method)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                res = 1 - res

            locs = np.where(res >= threshold)
            matches = list(zip(*locs[::-1]))

            return matches

        except Exception as e:
            logger.error(f"Template matching error: {e}")
            return []

    @staticmethod
    def _remove_duplicates_optimized(
        items: List[DetectionResult], min_distance: int
    ) -> List[DetectionResult]:
        """
        Remove duplicate detections (optimized algorithm).
        Groups by item name first, then uses spatial locality.
        """
        if not items:
            return []

        # Group by item name first
        by_item: Dict[str, List[DetectionResult]] = {}
        for item in items:
            by_item.setdefault(item.item, []).append(item)

        unique_items: List[DetectionResult] = []
        CHECK_WINDOW = 10  # Only check recent items
        min_dist_sq = min_distance * min_distance  # Use squared distance to avoid sqrt

        for item_list in by_item.values():
            # Sort by position for spatial locality
            sorted_items = sorted(item_list, key=lambda i: (i.x, i.y))

            for item in sorted_items:
                # Check only against recent additions (use squared distance)
                is_duplicate = any(
                    existing.item == item.item
                    and (existing.x - item.x) ** 2 + (existing.y - item.y) ** 2
                    < min_dist_sq
                    for existing in unique_items[-CHECK_WINDOW:]
                )
                if not is_duplicate:
                    unique_items.append(item)

        return unique_items
