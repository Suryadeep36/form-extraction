"""
Debug / observability helpers.

Visual QA artifacts are written to the directory configured by
DEBUG_VISION_DIR. Everything here is optional and driven purely by config;
production callers should keep DEBUG_VISION disabled.
"""

import json
import os

import cv2

import util.config as config


def _debug_dir():
    os.makedirs(config.DEBUG_VISION_DIR, exist_ok=True)
    return config.DEBUG_VISION_DIR


def draw_document_overlay(image, doc_rep, max_width=1400):
    """
    Draw a human-readable overlay of the geometric layers:

      * table bboxes and cell grids
      * checkbox rectangles
      * input regions (colored by kind)
      * OCR element boxes (green) with a small label bleed

    Returns a new BGR ndarray; the input image is not modified.
    """
    out = image.copy()
    h, w = out.shape[:2]
    scale = max_width / w if max_width and w > max_width else 1.0
    if scale != 1.0:
        out = cv2.resize(out, (int(w * scale), int(h * scale)))

    def _pts(bbox):
        x1, y1, x2, y2 = [float(v) * scale for v in bbox]
        return [(int(x1), int(y1)), (int(x2), int(y2))]

    # Tables first (they are backgrounds).
    for table in doc_rep.get("tables", []):
        p1, p2 = _pts(table["bbox"])
        cv2.rectangle(out, p1, p2, (255, 0, 255), 2)
        for cell in table.get("cells", []):
            cv2.rectangle(out, *_pts(cell["bbox"]), (180, 60, 220), 1)

    for checkbox in doc_rep.get("checkboxes", []):
        p1, p2 = _pts(checkbox["bbox"])
        cv2.rectangle(out, p1, p2, (255, 170, 0), 2)

    region_colors = {
        "underline": (255, 0, 0),
        "box": (0, 200, 255),
        "blank": (0, 255, 200),
        "checkbox": (255, 170, 0),
    }
    fields = {f.get("id"): f for f in doc_rep.get("fields", [])}
    for region in doc_rep.get("input_regions", []):
        color = region_colors.get(region.get("kind"), (200, 200, 0))
        p1, p2 = _pts(region["bbox"])
        cv2.rectangle(out, p1, p2, color, 2)
        if region.get("id") in fields:
            cv2.putText(
                out,
                str(region["id"]),
                (p1[0], max(12, p1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    for element in doc_rep.get("elements", []):
        p1, p2 = _pts(element["bbox"])
        cv2.rectangle(out, p1, p2, (0, 200, 0), 1)

    for region in doc_rep.get("regions", []):
        p1, p2 = _pts(region["bbox"])
        cv2.rectangle(out, p1, p2, (255, 255, 0), 1)

    return out


def write_debug_document(image, doc_rep, name):
    """Persist the annotated overlay + a JSON summary of a document rep."""
    if not config.DEBUG_VISION:
        return None

    d = _debug_dir()
    stem = os.path.splitext(os.path.basename(name))[0] or "document"

    overlay = draw_document_overlay(image, doc_rep)
    overlay_path = os.path.join(d, f"{stem}_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)

    summary = {
        "elements": len(doc_rep.get("elements", [])),
        "input_regions": len(doc_rep.get("input_regions", [])),
        "fields": len(doc_rep.get("fields", [])),
        "tables": [
            {
                "id": t["id"],
                "bbox": t["bbox"],
                "n_cells": t["n_cells"],
                "structure_source": t.get("structure_source"),
            }
            for t in doc_rep.get("tables", [])
        ],
        "checkboxes": len(doc_rep.get("checkboxes", [])),
        "regions": len(doc_rep.get("regions", [])),
        "preprocessing": doc_rep.get("preprocessing"),
    }
    json_path = os.path.join(d, f"{stem}_vision.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return overlay_path