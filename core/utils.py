"""
Shared utility functions for the Auto C-Peach system.
"""

import os
import logging

logger = logging.getLogger(__name__)


# ==================== FILE & PATH UTILITIES ====================

def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if not present.

    Args:
        path (str): Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance by module name.

    Args:
        name (str): Module name (usually __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

