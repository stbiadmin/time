"""
Module: logging_config.py
=========================

Purpose: Configure structured logging for the ML demonstration pipeline.

Proper logging is essential for debugging and monitoring ML pipelines.
         We use Python's built-in logging with custom formatting that includes
         timestamps, log levels, and module names for easy tracing.

Business Context:
    In production ML systems, logs help diagnose issues during training,
    data processing, and inference. This setup mirrors enterprise practices.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "outputs/logs"
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.

    We set up both console output (for immediate feedback during
             development) and file logging (for post-hoc analysis).

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific log file name
        log_dir: Directory for log files

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(log_level="DEBUG")
        >>> logger.info("Starting training pipeline")
    """
    # ============ CREATE LOGGER ============
    logger = logging.getLogger("time_series_ml")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # ============ DEFINE FORMAT ============
    # NOTE: This format includes all essential information for debugging:
    #        - Timestamp for temporal ordering
    #        - Level for severity filtering
    #        - Module name for source identification
    #        - Message for the actual log content
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ============ CONSOLE HANDLER ============
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    # ============ FILE HANDLER (OPTIONAL) ============
    if log_file or log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            # Generate timestamped log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"pipeline_{timestamp}.log"

        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path / log_file}")

    return logger


def get_logger(name: str = "time_series_ml") -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name (default: main pipeline logger)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============ CONVENIENCE FUNCTIONS ============

def log_section(logger: logging.Logger, title: str, char: str = "=") -> None:
    """
    Log a visual section separator for better readability.

    These separators improve readability
             by clearly delineating different pipeline stages.

    Args:
        logger: Logger instance
        title: Section title
        char: Character to use for the separator line
    """
    separator = char * 60
    logger.info(separator)
    logger.info(f"  {title}")
    logger.info(separator)


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = "") -> None:
    """
    Log a dictionary of metrics in a formatted way.

    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for all metric names
    """
    for name, value in metrics.items():
        metric_name = f"{prefix}{name}" if prefix else name
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")
