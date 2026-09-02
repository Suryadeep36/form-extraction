"""
Centralized configuration for the document extraction pipeline.

All tunables that affect the geometric/structure layers live here and are
driven by environment variables so they never need to be hardcoded across
the codebase.

Naming convention matches the existing project (snake_case module attributes,
UPPER_SNAKE env vars).
"""

import os


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


_load_env()


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Enable/disable document-boundary detection + perspective correction.
PERSPECTIVE_CORRECTION_ENABLED = _env_bool(
    "PERSPECTIVE_CORRECTION_ENABLED", default=True
)

# Minimum corner-detection confidence (0..1) below which the original image
# is used unchanged.
PERSPECTIVE_MIN_CONFIDENCE = _env_float("PERSPECTIVE_MIN_CONFIDENCE", default=0.55)

# ---------------------------------------------------------------------------
# Table model (pretrained, optional)
# ---------------------------------------------------------------------------

# Master switch for the pretrained table detection / structure pipeline.
TABLE_MODEL_ENABLED = _env_bool("TABLE_MODEL_ENABLED", default=True)

# Device selection. Values: "auto", "gpu", "cpu", or a raw paddle device
# string such as "gpu:0".
TABLE_MODEL_DEVICE = os.getenv("TABLE_MODEL_DEVICE", "auto")

# Minimum score accepted from the model table detector.
TABLE_DETECTION_THRESHOLD = _env_float("TABLE_DETECTION_THRESHOLD", default=0.5)

# Whether the model stack requires official-weights download on first use.
TABLE_MODEL_DOWNLOAD_ALLOWED = _env_bool("TABLE_MODEL_DOWNLOAD_ALLOWED", default=True)

# Model names used by the Paddle table stack. Keep them configurable so the
# service can be pointed at alternative pretrained checkpoints without code
# changes.
TABLE_DETECTION_MODEL = os.getenv("TABLE_DETECTION_MODEL", "PicoDet_layout_1x_table")
TABLE_STRUCTURE_MODEL_WIRED = os.getenv(
    "TABLE_STRUCTURE_MODEL_WIRED", "SLANeXt_wired"
)
TABLE_STRUCTURE_MODEL_WIRELESS = os.getenv(
    "TABLE_STRUCTURE_MODEL_WIRELESS", "SLANeXt_wireless"
)

# ---------------------------------------------------------------------------
# CV table detection
# ---------------------------------------------------------------------------

# Minimum IoU between two candidate tables to be considered duplicates / the
# same physical table during fusion.
TABLE_FUSION_IOU_THRESHOLD = _env_float("TABLE_FUSION_IOU_THRESHOLD", default=0.5)

# Minimum confidence for a CV-only table to be kept when the model stack is
# unavailable or silent.
TABLE_CV_MIN_CONFIDENCE = _env_float("TABLE_CV_MIN_CONFIDENCE", default=0.35)

MIN_TABLE_AREA_RATIO = _env_float("MIN_TABLE_AREA_RATIO", default=0.015)
MIN_TABLE_WIDTH_RATIO = _env_float("MIN_TABLE_WIDTH_RATIO", default=0.18)

# Structure recognition cell-grid expansion tolerance (pixels).
TABLE_CELL_PAD = _env_int("TABLE_CELL_PAD", default=2)

# ---------------------------------------------------------------------------
# Line detection
# ---------------------------------------------------------------------------

FIELD_LINE_MIN_WIDTH_RATIO = _env_float("FIELD_LINE_MIN_WIDTH_RATIO", default=0.10)
TABLE_LINE_MIN_WIDTH_RATIO = _env_float("TABLE_LINE_MIN_WIDTH_RATIO", default=0.15)
TABLE_LINE_MIN_HEIGHT_RATIO = _env_float("TABLE_LINE_MIN_HEIGHT_RATIO", default=0.10)

# ---------------------------------------------------------------------------
# Form fields
# ---------------------------------------------------------------------------

# Minimum OCR overlap ratio required to treat a text element as the value of
# an input region.
FIELD_VALUE_OVERLAP_RATIO = _env_float("FIELD_VALUE_OVERLAP_RATIO", default=0.30)

# Maximum gap (as a fraction of the input region width) between a label and
# the start of an input region for them to be associated.
FIELD_LABEL_MAX_GAP_RATIO = _env_float("FIELD_LABEL_MAX_GAP_RATIO", default=2.0)

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

# Ask PaddleOCR for word-level boxes so compound-element splitting can use
# real token coordinates instead of proportional guesses.
OCR_USE_WORD_BOXES = _env_bool("OCR_USE_WORD_BOXES", default=True)

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

# PostgreSQL connection string for storing uploaded documents + extraction
# output. When unset, document storage is disabled and /extract-document
# simply returns the live response without saving anything.
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Directory (relative to the repo root) holding the uploaded document files.
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# ---------------------------------------------------------------------------
# Observability / debugging
# ---------------------------------------------------------------------------

DEBUG_VISION = _env_bool("DEBUG_VISION", default=False)
DEBUG_VISION_DIR = os.getenv("DEBUG_VISION_DIR", "debug")
DEBUG_LOG = _env_bool("DEBUG_LOG", default=False)


def log_debug(message: str) -> None:
    """Print a debug-level message only when DEBUG_LOG is enabled."""
    if DEBUG_LOG:
        print(f"[DEBUG] {message}")