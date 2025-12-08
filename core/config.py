"""Configuration Module - System-wide configuration management.

This module provides centralized configuration management with environment
variable support, validation, and type safety.
"""

import os
from typing import Any, Dict

from .utils import get_logger

logger = get_logger(__name__)


# ==================== ENVIRONMENT VARIABLE HELPERS ====================


def get_env_str(key: str, default: str) -> str:
    """Get string value from environment variable."""
    return os.environ.get(key, default)


def get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable with validation."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {key}: {value}, using default: {default}")
        return default


def get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable with validation."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {key}: {value}, using default: {default}")
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


# ==================== DEFAULT PATHS ====================
# Paths can be overridden via environment variables
DEFAULT_PATHS: Dict[str, str] = {
    "templates": get_env_str("AUTOCP_TEMPLATES_PATH", "./templates/jp"),
    "results": get_env_str("AUTOCP_RESULTS_PATH", "./result"),
    "snapshots": get_env_str("AUTOCP_SNAPSHOTS_PATH", "./result/snapshots"),
    "logs": get_env_str("AUTOCP_LOGS_PATH", "./logs"),
}

# ==================== DEFAULT VALUES ====================
DEFAULT_MAX_RETRIES: int = get_env_int("AUTOCP_MAX_RETRIES", 3)
DEFAULT_RETRY_DELAY: float = get_env_float("AUTOCP_RETRY_DELAY", 1.0)
DEFAULT_TOUCH_TIMES: int = get_env_int("AUTOCP_TOUCH_TIMES", 1)

# ==================== TIMEOUT SETTINGS ====================
DEFAULT_OPERATION_TIMEOUT: float = get_env_float("AUTOCP_OPERATION_TIMEOUT", 30.0)
DEFAULT_CONNECTION_TIMEOUT: float = get_env_float("AUTOCP_CONNECTION_TIMEOUT", 10.0)
DEFAULT_OCR_TIMEOUT: float = get_env_float("AUTOCP_OCR_TIMEOUT", 5.0)

# ==================== DEVICE SETTINGS ====================
DEFAULT_DEVICE_URL: str = get_env_str(
    "AUTOCP_DEVICE_URL", "Windows:///?title_re=DOAX VenusVacation.*"
)

# ==================== ROI CONFIGURATION ====================
# Format: [x1, x2, y1, y2] - pixel coordinates
FESTIVALS_ROI_CONFIG: Dict[str, list] = {
    "フェス名": [784, 1296, 247, 759],  # Festival name/rank area
    "フェスランク": [392, 904, 41, 86],  # Festival rank level
    "勝利点数": [1012, 1240, 41, 86],  # Victory points
    "推奨ランク": [1050, 1255, 108, 163],  # Recommended rank
    "Sランクボーダー": [1050, 1255, 170, 225],  # S rank border points
    "初回クリア報酬": [1050, 1255, 233, 330],  # First clear reward
    "Sランク報酬": [1050, 1255, 343, 440],  # S rank reward
    "消費FP": [1050, 1255, 400, 500],  # FP consumed
    "獲得ザックマネー": [392, 904, 20, 60],  # Zack money earned
    "獲得アイテム": [392, 904, 70, 170],  # Items earned
    "獲得EXP-Ace": [950, 1040, 115, 155],  # EXP for Ace units
    "獲得EXP-NonAce": [950, 1040, 215, 255],  # EXP for non-Ace units
    "エース": [1085, 1175, 85, 175],  # Venus memory for Ace
    "非エース": [1085, 1175, 185, 275],  # Venus memory for non-Ace
}

GACHA_ROI_CONFIG: Dict[str, list] = {
    "banner": [400, 900, 300, 400],  # Banner area
}

HOPPING_ROI_CONFIG: Dict[str, list] = {
    "コース名": [50, 400, 20, 80],  # Course name
    "アイテム名": [450, 800, 100, 160],  # Item name
    "獲得数": [800, 950, 100, 160],  # Acquired quantity
}


# ==================== DETECTOR CONFIGURATION ====================
DETECTOR_CONFIG: Dict[str, Any] = {
    "yolo": {
        "model_path": "yolo11n.pt",  # Model: n/s/m/l/x
        "confidence": 0.25,  # Confidence threshold (0.0-1.0)
        "iou": 0.45,  # IoU threshold for NMS
        "imgsz": 640,  # Input image size
        "device": "cpu",  # Device: cpu/cuda/mps/auto
    },
    "template": {
        "templates_dir": DEFAULT_PATHS["templates"],  # Templates directory
        "threshold": 0.85,  # Match threshold (0.0-1.0)
        "method": "TM_CCOEFF_NORMED",  # Matching method
        "min_distance": 15,  # Min distance for duplicates
    },
    "quantity_extraction": {  # OCR region for quantity
        "offset_x": 30,  # Offset X from bbox right
        "offset_y": 0,  # Offset Y from bbox bottom
        "roi_width": 80,  # OCR region width
        "roi_height": 30,  # OCR region height
    },
}

# ==================== AUTOMATION CONFIGURATIONS ====================
FESTIVAL_CONFIG: Dict[str, Any] = {
    # Paths
    "templates_path": DEFAULT_PATHS["templates"],
    "snapshot_dir": f"{DEFAULT_PATHS['results']}/festival/snapshots",
    "results_dir": f"{DEFAULT_PATHS['results']}/festival/results",
    # Timing & Retry
    "wait_after_touch": 1.0,
    "max_step_retries": 5,
    "retry_delay": 1.0,
    # Fuzzy matching (0.9+=strict, 0.7-0.8=balanced, 0.5-0.6=lenient)
    "fuzzy_matching": {"enabled": True, "threshold": 0.7},
    # Detector (yolo/template/auto)
    "use_detector": True,
    "detector_type": "template",
    "yolo_config": {"model_path": "yolo11n.pt", "confidence": 0.25, "device": "cpu"},
    "template_config": {
        "templates_dir": DEFAULT_PATHS["templates"],
        "threshold": 0.85,
        "method": "TM_CCOEFF_NORMED",
    },
    # ROI groups
    "festivals_rois": ["フェス名", "フェスランク"],
    "pre_battle_rois": [
        "勝利点数",
        "推奨ランク",
        "消費FP",
        "Sランクボーダー",
        "初回クリア報酬",
        "Sランク報酬",
    ],
    "post_battle_rois": [
        "獲得ザックマネー",
        "獲得アイテム",
        "獲得EXP-Ace",
        "獲得EXP-NonAce",
        "エース",
        "非エース",
    ],
}

GACHA_CONFIG: Dict[str, Any] = {
    # Paths
    "templates_path": DEFAULT_PATHS["templates"],
    "snapshot_dir": f"{DEFAULT_PATHS['results']}/gacha/snapshots",
    "results_dir": f"{DEFAULT_PATHS['results']}/gacha/results",
    # Timing & Pull settings
    "wait_after_touch": 1.0,
    "wait_after_pull": 2.0,
    "max_pulls": 10,
    "pull_type": "single",  # single or multi
    # Detector
    "use_detector": False,
    "detector_type": "auto",
}

HOPPING_CONFIG: Dict[str, Any] = {
    # Paths
    "templates_path": DEFAULT_PATHS["templates"],
    "snapshot_dir": f"{DEFAULT_PATHS['results']}/hopping/snapshots",
    "results_dir": f"{DEFAULT_PATHS['results']}/hopping/results",
    # Timing & Retry
    "wait_after_touch": 1.0,
    "max_step_retries": 5,
    "retry_delay": 1.0,
    # Fuzzy matching (0.9+=strict, 0.7-0.8=balanced, 0.5-0.6=lenient)
    "fuzzy_matching": {"enabled": True, "threshold": 0.7},
    # Detector (yolo/template/auto)
    "use_detector": False,
    "detector_type": "template",
    "yolo_config": {"model_path": "yolo11n.pt", "confidence": 0.25, "device": "cpu"},
    "template_config": {
        "templates_dir": DEFAULT_PATHS["templates"],
        "threshold": 0.85,
        "method": "TM_CCOEFF_NORMED",
    },
    # ROI groups for verification
    "verification_rois": ["アイテム名", "獲得数"],
}


# ==================== UTILITY FUNCTIONS ====================
def get_festival_config() -> Dict[str, Any]:
    """Get Festival Automation configuration"""
    return FESTIVAL_CONFIG.copy()


def get_gacha_config() -> Dict[str, Any]:
    """Get Gacha Automation configuration"""
    return GACHA_CONFIG.copy()


def get_hopping_config() -> Dict[str, Any]:
    """Get Hopping Automation configuration"""
    return HOPPING_CONFIG.copy()


def get_detector_config(detector_type: str = "yolo") -> Dict[str, Any]:
    """Get Detector configuration (yolo or template)"""
    if detector_type not in ["yolo", "template"]:
        logger.warning(f"Unknown detector type: {detector_type}, using 'yolo'")
        detector_type = "yolo"
    config = DETECTOR_CONFIG[detector_type].copy()
    if detector_type == "yolo":
        config["quantity_extraction"] = DETECTOR_CONFIG["quantity_extraction"].copy()
    return config


def merge_config(
    base_config: Dict[str, Any], custom_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge custom config into base config"""
    result = base_config.copy()
    for key, value in custom_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result
