"""
OCR-anchored checkbox / selection-mark detection.

Low-level pixel detectors alone are unreliable on real scanned forms: printed
box outlines are thin and fragmented, handwritten X/check marks look like text
glyphs, and boxes sit right at the edge of their label.  We therefore anchor
detection to the OCR output, which is reliable, and sample the expected
checkbox column immediately left of each option row.

Detection is deliberately conservative: a candidate is only kept when the
sampled window shows a real box ring or an unambiguous X/check mark.  Plain
space, single stray glyphs and ordinary text are rejected, so we do not spam
the downstream structure resolver with hundreds of bogus checkboxes.  On
degraded scans this will under-detect (never over-detect), which is the safer
direction: the semantic/LLM layer can still reason about option lists.
"""

import re
import cv2
import numpy as np

from util.image_utils import _load_gray, _threshold_gray


_MARK_LEAD = re.compile(r'^[Xx✓✔✗✘■▪□_•·\-\u2013\u2014]\s*', re.UNICODE)


def _classify_window(bin_img, x1, y1, x2, y2):
    """Return (edge_dark, inner_dark) for the window, or None if too small."""
    h, w = bin_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    win = bin_img[y1:y2, x1:x2]
    inner = win[1:-1, 1:-1]
    if win.shape[0] >= 2 and win.shape[1] >= 2:
        edge = np.concatenate([win[0, :], win[-1, :], win[:, 0], win[:, -1]])
    else:
        edge = win
    edge_dark = float(np.mean(edge > 0))
    inner_dark = float(np.mean(inner > 0)) if inner.size else 0.0
    return edge_dark, inner_dark


def _mark_kind(bin_img, bbox):
    """Guess interior mark: 'X' | 'tick' | 'filled' | 'ink' | 'empty'."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = bin_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return "empty"
    win = bin_img[y1:y2, x1:x2]
    inner = win[1:-1, 1:-1]
    if inner.size == 0:
        return "empty"
    density = float(np.mean(inner > 0))
    if density >= 0.45:
        return "filled"
    if density < 0.06:
        return "empty"
    diag = np.concatenate([np.diag(inner), np.diag(np.fliplr(inner))])
    diag_dark = float(np.mean(diag > 0)) if diag.size else 0.0
    body_dark = density
    if diag_dark > 0.35 and diag_dark > body_dark * 0.65:
        return "X"
    return "tick" if density >= 0.12 else "ink"


def _looks_like_box_window(edge_dark, inner_dark, box_w, box_h):
    """
    Decide whether a sampled window genuinely contains a checkbox:
      - a printed box is dark on its rim (all / most sides) with comparably
        little interior ink, OR
      - it contains a clear handwritten mark (X / tick / filled), which shows
        up as stronger interior ink than its rim.
    """
    if edge_dark >= 0.28:          # strong rim => a (possibly empty) box
        return True
    if inner_dark >= 0.18:          # lots of interior ink => a mark
        return True
    return False


def _neighbor_text_overlap(bin_img, win, min_ink_cols=2):
    """Reject windows whose content is really the right neighbor's text by
    checking whether dark extends to the right edge uninterrupted (i.e. the
    window's right edge touches real text, not a box gap)."""
    x1, y1, x2, y2 = win
    h, w = bin_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    col = bin_img[y1:y2, x2 - 1]
    return float(np.mean(col > 0)) > 0.4


def detect_checkboxes_visual(image, ocr_elements=None, options=None):
    """
    Conservative OCR-anchored checkbox detection.

    Returns a list of candidates with keys:
        id, bbox, center, width, height, type, state, confidence,
        mark_type, element_id, word, associated_text
    """
    gray = _load_gray(image)
    bin_img = _threshold_gray(gray)

    ocr_elements = ocr_elements or []
    candidates = []
    seen = set()

    for element in ocr_elements:
        if element.get("type") != "text":
            continue
        elem_id = element["id"]
        text = element["text"]
        # Skip elements that are plain value continuations ("... : value")
        # rather than option labels: their "left column" is another field and
        # not a checkbox row.
        if text.strip().startswith(":") and len(text.strip()) > 1:
            continue
        x1, y1, x2, y2 = [float(v) for v in element["bbox"]]
        text_h = max((y2 - y1), 10)
        box_w = min(max(int(text_h * 1.0), 12), 42)
        pad = max(2, int(text_h * 0.16))

        # Checkbox column immediately left of the option text.
        win = (int(x1 - box_w - pad), int(y1 - pad), int(x1 - pad), int(y2 + pad))
        r = _classify_window(bin_img, *win)
        if r is None:
            continue
        edge_dark, inner_dark = r
        if not _looks_like_box_window(edge_dark, inner_dark, box_w, win[3] - win[1]):
            continue
        # Avoid clipping the label's own first glyph as a "mark": if the right
        # edge is solidly dark the window is part of the text, not a box gap.
        if _neighbor_text_overlap(bin_img, win) and edge_dark < 0.28 and inner_dark < 0.6:
            continue

        cx1, cy1, cx2, cy2 = win
        cb_bbox = [cx1, cy1, cx2, cy2]
        center = [(cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0]
        key = (elem_id, int(center[0] // 6), int(center[1] // 6))
        if key in seen:
            continue
        seen.add(key)

        mark = _mark_kind(bin_img, cb_bbox) if inner_dark >= 0.12 else "empty"
        # Only an unambiguous cross / solid mark counts as reliably checked.
        # Generic "tick"/ink from neighboring-text bleed is treated as
        # uncertain rather than a confident checked to avoid false positives.
        if mark in ("X", "filled"):
            state = "checked"
            conf = 0.68 if edge_dark >= 0.28 else 0.55
        elif edge_dark >= 0.28:
            state = "unchecked"
            conf = min(0.9, 0.4 + edge_dark)
        else:
            state = "uncertain"
            mark = "ink"
            conf = 0.4

        candidates.append({
            "id": "",
            "bbox": cb_bbox,
            "center": [round(center[0], 2), round(center[1], 2)],
            "width": cx2 - cx1,
            "height": cy2 - cy1,
            "type": "checkbox",
            "state": state,
            "confidence": round(conf, 3),
            "mark_type": mark,
            "element_id": elem_id,
            "word": None,
            "associated_text": text,
        })

    candidates.sort(key=lambda c: (c["center"][1], c["center"][0]))
    for i, c in enumerate(candidates):
        c["id"] = f"c{i:03d}"
    return candidates


def strip_leading_option_mark(text, has_box=False):
    """
    Remove a selection mark that OCR glued/stuck onto the front of an option
    label (e.g. "XPartially Boatable" -> "Partially Boatable").  Only strips a
    single mark glyph when (a) it is the very first character, and (b) either a
    box was detected for this element (`has_box`) or the glyph is clearly a
    checkbox mark symbol.  A legitimate leading "X" that spaces out a label
    (e.g. coordinate-axis "X Latitude North") is left untouched.
    """
    if not text:
        return text
    m = _MARK_LEAD.match(text)
    if not m:
        return text, False
    glyph = m.group(0).strip()
    rest = text[m.end():].strip()
    if not rest:
        return text, False
    # Symbolic marks are almost never a label's first real word.
    if glyph in {"✓", "✔", "✗", "✘", "■", "▪", "□", "_", "•", "·"}:
        return rest, True
    if glyph.lower() == "x":
        # "XPartially Boatable" (glued, no space between X and word) is a mark.
        if text.startswith("X") and len(text) > 1 and text[1:2].isalnum() and not text[1:2].isspace():
            if (has_box or text[1].isupper() or _looks_like_mark_context(text)):
                return rest, True
        return text, False
    return text, False


def _looks_like_mark_context(text):
    """Heuristics: a glued leading 'X' followed immediately by another
    uppercase letter is more likely a mark (matches 'XPartially Boatable')."""
    return len(text) > 1 and text[0].lower() == "x" and text[1].isalnum()


def leading_mark_width_px(element, bin_img=None):
    """Return the pixel width of a leading mark glyph, if present, else 0."""
    text = element.get("text") or ""
    if element.get("words"):
        first = next((w for w in element["words"] if w.get("text", "").strip()), None)
        if first:
            return first["bbox"][2] - first["bbox"][0]
    return 0


def checkbox_from_leading_mark(element, image=None):
    """
    Build a checkbox candidate purely from an OCR-leading selection mark.

    Many forms mark a choice by writing an X / tick / underscore against the
    label, and the OCR engine glues it onto the front of the option text
    (e.g. t028 "XPartially Boatable", t019 "_NO IfNO...").  The leading glyph
    is a reliable `checked` signal on its own.

    Returns a candidate dict (same schema as detect_checkboxes_visual) or None
    when the element has no leading mark.  The element's text is NOT modified
    here (the caller strips it separately so the box state and the cleaned
    label stay consistent).
    """
    text = element.get("text") or ""
    m = _MARK_LEAD.match(text)
    if not m:
        return None
    glyph = m.group(0).strip()
    if not glyph:
        return None
    # A leading "X" separated by a space ("X Latitude North") is a label's own
    # first token, not a mark -> do not treat as checked.
    if glyph.lower() == "x":
        if len(text) > 1 and text[1:2].isspace():
            return None
    bbox = element.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h = max((y2 - y1), 12)
    wmark = max(int(h * 0.6), 8)
    cb_bbox = [int(x1 - wmark), int(y1), int(x1), int(y2)]
    center = [(cb_bbox[0] + cb_bbox[2]) / 2.0, (cb_bbox[1] + cb_bbox[3]) / 2.0]
    return {
        "id": "",
        "bbox": cb_bbox,
        "center": [round(center[0], 2), round(center[1], 2)],
        "width": cb_bbox[2] - cb_bbox[0],
        "height": cb_bbox[3] - cb_bbox[1],
        "type": "checkbox",
        "state": "checked",
        "confidence": 0.7,
        "mark_type": "X" if glyph.lower() == "x" else "tick",
        "element_id": element["id"],
        "word": None,
        "associated_text": text[m.end():].strip(),
        "source": "ocr_mark",
    }


def merge_checkbox_candidates(pixel_candidates, mark_candidates):
    """
    Merge pixel- and OCR-mark-derived checkbox candidates, keeping at most one
    per element.  Prefer the OCR-mark candidate when both agree it is real (the
    OCR-mark signal is the more reliable `checked` indicator on degraded
    scans).  New pixel-only candidates that do not overlap an OCR-mark
    candidate are kept as-is.
    """
    by_element = {}
    for c in mark_candidates:
        if c.get("element_id"):
            by_element[c["element_id"]] = c
    for c in pixel_candidates:
        eid = c.get("element_id")
        if eid and eid in by_element:
            continue  # OCR-mark candidate already covers this option
        if eid:
            by_element[eid] = c
        else:
            by_element[f"un{len(by_element)}"] = c

    merged = list(by_element.values())
    merged.sort(key=lambda c: (c["center"][1], c["center"][0]))
    for i, c in enumerate(merged):
        c["id"] = f"c{i:03d}"
    return merged
