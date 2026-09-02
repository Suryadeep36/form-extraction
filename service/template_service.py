"""
Two-pass template registration + filled-form extraction.

Pass 1 - register_template(empty_form_image):
    Preprocess + analyze the EMPTY form with the existing pipeline, turn its
    blank value regions into a reusable template, and persist both the template
    JSON and the warped reference image.

Pass 2 - extract_filled(template, filled_form_image):
    Preprocess + analyze the FILLED form, warp it into the template coordinate
    space via Homography alignment, then recover each field's value.  Checkbox
    regions are decided by OpenCV pixel density, everything else by OCR text
    that lands inside the pre-registered value region.

Both passes reuse the existing document pipeline (`analyze_document`), so
templates are truly dynamic: any form the pipeline can parse works, with no
hardcoded labels or positions.  The only coupling to the empty form is the
layout itself, which is exactly what we want for a template.
"""

import os
import json
import re
import uuid
import base64
import shutil

import cv2
import numpy as np

from service.document_service import (
    analyze_document,
    preprocess_document,
    build_document_representation,
)
from util.alignment_utils import estimate_transform
from util.form_utils import _repair_date_range
from util.template_utils import (
    build_template_fields,
    build_template_tables,
    find_label_anchor,
)
from util.config import UPLOAD_DIR

TEMPLATES_DIR = os.path.join(UPLOAD_DIR, "templates")


# ---------------------------------------------------------------------------
# Persistence (file-based, no DB dependency)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Convert numpy scalars/arrays to plain JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.ndarray,)):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def _templates_dir():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    return TEMPLATES_DIR


def _template_path(template_id):
    return os.path.join(_templates_dir(), f"{template_id}.json")


def _reference_path(template_id):
    return os.path.join(_templates_dir(), f"{template_id}_ref.jpg")


def save_template(template):
    """Persist a template dict + its reference image. Returns template_id."""
    template_id = template.get("id") or str(uuid.uuid4())
    template["id"] = template_id
    ref_image = template.pop("reference_image", None)
    with open(_template_path(template_id), "w") as f:
        meta = {k: v for k, v in template.items() if k != "reference_image"}
        json.dump(meta, f, indent=2)
    if ref_image is not None:
        ok, buf = cv2.imencode(".jpg", ref_image)
        if ok:
            with open(_reference_path(template_id), "wb") as f:
                f.write(buf.tobytes())
    return template_id


def load_template(template_id, with_reference=False):
    path = _template_path(template_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        template = json.load(f)
    if with_reference:
        ref_path = _reference_path(template_id)
        if os.path.exists(ref_path):
            template["reference_image"] = cv2.imread(ref_path)
    return template


def get_template_image(template_id):
    """
    Raw bytes of a template's reference (preprocessed blank-form) image, or
    None when the template does not exist / has no stored image.
    """
    if not os.path.exists(_template_path(template_id)):
        return None
    ref_path = _reference_path(template_id)
    if not os.path.exists(ref_path):
        return None
    with open(ref_path, "rb") as f:
        return f.read()


def list_templates():
    if not os.path.isdir(_templates_dir()):
        return []
    out = []
    for fname in sorted(os.listdir(_templates_dir())):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_templates_dir(), fname)) as f:
                t = json.load(f)
            out.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "source_filename": t.get("source_filename"),
                "field_count": len(t.get("fields", [])),
                "table_count": len(t.get("tables", [])),
                "created_at": t.get("created_at"),
            })
        except Exception as e:
            print(f"[TEMPLATE] skip {fname}: {e}")
    return out


# ---------------------------------------------------------------------------
# Pass 1 - registration
# ---------------------------------------------------------------------------

def register_template(image_path, name=None):
    """
    Take an EMPTY form image and build a template from its blank structure.

    Returns:
        {
            "template_id", "name", "source_filename", "fields": [...],
            "image_width", "image_height", "alignment_reference": data-url,
            "created_at",
        }
    """
    info = preprocess_document(image_path)
    doc_rep = build_document_representation(info["image"], image_path=image_path)
    w, h = doc_rep["image_width"], doc_rep["image_height"]
    fields = build_template_fields(doc_rep, w, h)
    tables = build_template_tables(doc_rep, w, h)

    template = {
        "id": str(uuid.uuid4()),
        "name": name or os.path.splitext(os.path.basename(image_path))[0],
        "source_filename": os.path.basename(image_path),
        "image_width": w,
        "image_height": h,
        "fields": fields,
        "tables": tables,
        "created_at": None,
        "reference_image": info["image"],
    }
    import datetime
    template["created_at"] = datetime.datetime.utcnow().isoformat()

    template_id = save_template(template)
    # Encode reference image as data URL for the API response (not stored in JSON).
    ok, buf = cv2.imencode(".jpg", info["image"])
    ref_url = None
    if ok:
        ref_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    template.pop("reference_image", None)
    template["template_id"] = template_id
    template["reference_image_data_url"] = ref_url
    print(f"[TEMPLATE] registered {template_id} with {len(fields)} fields")
    return template


# ---------------------------------------------------------------------------
# Pass 2 - extraction
# ---------------------------------------------------------------------------

def extract_filled(template, image_path):
    """
    Extract key/value pairs from a FILLED form using a registered template.

    Returns:
        {
            "template_id", "fields": [ {label, value, ...} ... ],
            "alignment": {...transform...},
            "warped_image_data_url": optional,
        }
    """
    info = preprocess_document(image_path)
    doc_rep = build_document_representation(info["image"], image_path=image_path)

    # --- Align filled -> template coordinate space (Homography) ----------
    elements = doc_rep["elements"]
    ref_image = template.get("reference_image")
    if ref_image is None:
        # No stored reference; fall back to identity alignment.
        warped = info["image"]
        transform = {"method": "identity", "matches": 0, "inlier_ratio": 0.0}
        aligned_elements = elements
    else:
        warped, transform = estimate_transform(ref_image, info["image"])
        if len(warped.shape) == 2:
            warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        else:
            warped_bgr = warped
        # Transform each element bbox through the homography into template space.
        H = transform.get("homography")
        aligned_elements = _warp_element_bbox(elements, H, doc_rep["image_width"])
        # Fall back to raw elements when alignment was identity.
        if transform["method"] == "identity":
            aligned_elements = elements

    # --- Extract each field's value ---------------------------------------
    w = template.get("image_width") or doc_rep["image_width"]
    h = template.get("image_height") or doc_rep["image_height"]
    # The template value bboxes are normalized to the TEMPLATE page; we work
    # in the warped image space, which equals the template space, so convert
    # normalized -> pixels using the template dims.
    norm_to_px = lambda vb: [vb[0]*w, vb[1]*h, vb[2]*w, vb[3]*h]

    # The elements are in the aligned space; build a workspace object with
    # raw pixel bboxes and the warped image for checkbox density.
    workspace = {
        "image": warped if transform.get("homography") is not None else info["image"],
        "element_px": aligned_elements,
        "checkboxes": doc_rep.get("checkboxes") or [],
    }

    extracted = []
    for field in template["fields"]:
        vb_px = norm_to_px(field["value_bbox"])
        # Make a copy of the field with pixel value_bbox so the helper works
        # with real coordinates.
        fcopy = dict(field)
        fcopy["value_bbox_px"] = vb_px
        result = _extract_one(fcopy, workspace, w, h)
        extracted.append(result)

    # --- Extract each table's cells --------------------------------------
    tables_out = _extract_template_tables(template, aligned_elements, w, h)

    # --- Response ----------------------------------------------------------
    out = {
        "template_id": template.get("id"),
        "template_name": template.get("name"),
        "fields": extracted,
        "tables": tables_out,
        "alignment": _json_safe(transform),
        "element_count": len(doc_rep["elements"]),
    }
    # Optionally include the warped image for debugging.
    try:
        ok, buf = cv2.imencode(".jpg", workspace["image"])
        if ok:
            out["warped_image_data_url"] = (
                "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            )
    except Exception as e:
        print(f"[TEMPLATE] warped encode failed: {e}")
    return out


def _warp_element_bbox(elements, H, width):
    """Transform each element's pixel bbox through homography H."""
    if H is None:
        return elements
    out = []
    for e in elements:
        bbox = e["bbox"]
        corners = np.float32([
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]],
        ]).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(corners, H).reshape(4, 2)
            xs, ys = warped[:, 0], warped[:, 1]
            new_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        except cv2.error:
            new_bbox = list(bbox)
        ne = dict(e)
        ne["bbox"] = new_bbox
        # Word-level boxes are stored in the same original OCR space as the
        # element bbox; they must be transformed through the same homography
        # or the compound-header/word assignment in `_assign_ocr_to_cells`
        # would resolve words against misaligned (un-warped) coordinates and
        # drop them into the wrong table row.
        if e.get("words"):
            warped_words = []
            for w in e["words"]:
                wb = w.get("bbox")
                if not wb or len(wb) != 4:
                    continue
                wc = np.float32([
                    [wb[0], wb[1]], [wb[2], wb[1]],
                    [wb[2], wb[3]], [wb[0], wb[3]],
                ]).reshape(-1, 1, 2)
                try:
                    ww = cv2.perspectiveTransform(wc, H).reshape(4, 2)
                    wxs, wys = ww[:, 0], ww[:, 1]
                    wbbox = [float(wxs.min()), float(wys.min()), float(wxs.max()), float(wys.max())]
                except cv2.error:
                    wbbox = list(wb)
                nw = dict(w)
                nw["bbox"] = wbbox
                warped_words.append(nw)
            if warped_words:
                ne["words"] = warped_words
        out.append(ne)
    return out


def _extract_one(field_copy, workspace, w, h):
    """Extract a single field respecting checkbox regions."""
    vb = field_copy["value_bbox_px"]
    cb_match = None
    best_iou = 0.0
    for cb in workspace["checkboxes"]:
        cb_bbox = cb["bbox"]
        ix1 = max(vb[0], cb_bbox[0]); iy1 = max(vb[1], cb_bbox[1])
        ix2 = min(vb[2], cb_bbox[2]); iy2 = min(vb[3], cb_bbox[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = (vb[2]-vb[0])*(vb[3]-vb[1]) + (cb_bbox[2]-cb_bbox[0])*(cb_bbox[3]-cb_bbox[1]) - inter
        r = inter / union if union else 0.0
        if r > best_iou:
            best_iou, cb_match = r, cb
    if cb_match is not None and best_iou >= 0.15:
        from util.checkbox_utils import _mark_kind, _classify_window
        from util.image_utils import _threshold_gray
        bin_img = _threshold_gray(workspace["image"])
        mark = _mark_kind(bin_img, cb_match["bbox"])
        checked = mark in ("X", "filled", "tick")
        return {
            "label": field_copy["label"],
            "value": "True" if checked else "False",
            "value_type": "boolean",
            "confidence": cb_match.get("confidence"),
            "bbox": [round(x / w, 5) for x in vb],
            "source": "checkbox",
            "checked": checked,
            "mark_type": mark,
        }

    # OCR text extraction inside the value bbox.
    #
    # Sampling the raw detection region bleeds text from neighboring rows
    # (the region is full-width/tall while a value is short).  Instead anchor
    # the window to the matched LABEL on the filled form:
    #   x from label-right -> region-right
    #   y clipped tightly around the label's row band.
    # This keeps the value crisp and avoids catching the next row's label.
    anchor = find_label_anchor(field_copy["label"], workspace["element_px"])

    value_window = None
    if anchor is not None:
        lb = anchor["bbox"]
        label_h = max(lb[3] - lb[1], 12)
        row_cy = (lb[1] + lb[3]) / 2.0
        # Values sit on the label's own line; keep the vertical band tight to
        # the anchor row so an adjacent row (e.g. a title above a field) never
        # bleeds in.
        row_half = max(label_h * 0.65, 10)
        gap = max(label_h * 0.25, 4)
        x_left = lb[2] + gap
        # Cap the window width so a full-page underline cannot swallow text
        # belonging to independent right-hand content.
        max_width = max((vb[2] - vb[0]) * 0.55, 160)
        x_right = min(vb[2], x_left + max_width)
        x_left = min(max(x_left, vb[0]), x_right)
        value_window = [x_left, row_cy - row_half, x_right, row_cy + row_half]
    else:
        # No reliable label anchor: use the template region (with tight pad).
        value_window = [
            vb[0],
            vb[1] + (vb[3] - vb[1]) * 0.08,
            vb[2],
            vb[3] - (vb[3] - vb[1]) * 0.08,
        ]

    def _center_in_region(bbox, region):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]

    matched = []
    for e in workspace["element_px"]:
        bbox = e["bbox"]
        if not _center_in_region(bbox, value_window):
            continue
        # Exclude the anchor label itself and any text starting before the
        # anchor's right edge (a value cannot live left of its own label).
        if anchor is not None and (bbox[1] + bbox[3]) / 2.0 is not None:
            if e["id"] == anchor["id"]:
                continue
            emid_x = (bbox[0] + bbox[2]) / 2.0
            lb = anchor["bbox"]
            if emid_x < lb[0]:
                continue
        matched.append(e)
    if not matched:
        return {
            "label": field_copy["label"],
            "value": None,
            "value_type": None,
            "confidence": None,
            "bbox": [round(x / w, 5) for x in vb],
            "source": "blank",
        }
    matched.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    value = " ".join(e["text"].strip() for e in matched).strip()
    # The printed label often ends with ":" but OCR can glue it onto the start
    # of a value (or leave a lone colon breadcrumb); the colon belongs to the
    # label, never to the value.
    value = re.sub(r"^\s*:+\s*|\s*:+\s*$", "", value)
    value = _repair_date_range(value)
    # Drop a lone anchor breadcrumb such as ':' if it's the only thing.
    if value in (":", ":", ""):
        value = None
    conf = float(np.mean([e.get("confidence", 0.0) for e in matched]))
    return {
        "label": field_copy["label"],
        "value": value or None,
        "value_type": None,
        "confidence": round(conf, 4),
        "bbox": [round(x / w, 5) for x in vb],
        "source": "ocr",
    }


def _extract_template_tables(template, aligned_elements, w, h):
    """
    Refill each template table's cells using the aligned filled form.

    The template stored the empty form's cell grid (normalized bboxes + static
    header text). We project that grid onto the warped filled form and reuse
    the pipeline's deterministic OCR->cell mapping so every cell gets the text
    that landed inside it (printed headers first, then the filled-in values).

    Returns a list of:
        {
            "id", "bbox", "n_rows", "n_cols", "n_cells",
            "structure_source",
            "cells": [
                {"row", "column", "row_span", "column_span",
                 "bbox": normalized, "template_text": static text or None,
                 "text": value found on the filled form or None}
            ],
        }
    """
    from service.table_service import _assign_ocr_to_cells

    norm_to_px = lambda vb: [vb[0]*w, vb[1]*h, vb[2]*w, vb[3]*h]

    tables_out = []
    for table in template.get("tables", []):
        t = dict(table)
        tb_px = norm_to_px(table["bbox"]) if table.get("bbox") else None
        cells = []
        for i, cell in enumerate(table.get("cells", [])):
            nc = dict(cell)
            nc["id"] = f"cell_{i:03d}"
            nc["template_text"] = nc.get("text")
            nc["bbox"] = norm_to_px(cell["bbox"])
            cells.append(nc)

        if tb_px is not None and cells:
            try:
                _assign_ocr_to_cells(aligned_elements, cells, tb_px)
            except Exception as e:
                print(f"[TEMPLATE] table cell assignment failed: {e}")

        for cell in cells:
            cell["bbox"] = [round(v / w, 5) for v in cell["bbox"]]
        t["cells"] = cells
        tables_out.append(t)

    return tables_out