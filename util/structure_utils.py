import re

from util.geometry_utils import (
    _point_in_bbox,
    _union_bbox,
    _norm,
    _intersection_over_area,
)
from service.ocr_service import (
    group_into_rows
)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0

def _region_center(region):
    if "center" in region:
        return region["center"]
    bbox = region["bbox"]
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def _geometry_field_signature(field):
    ids = set(field.get("label_element_ids") or [])
    ids.update(field.get("value_element_ids") or [])
    return ids


def _geometry_field_map(doc_rep):
    return {f["id"]: f for f in doc_rep.get("fields", [])}


def _find_geometry_table(doc_rep, region_id):
    """
    Find a cv/model table (doc_rep["tables"]) that corresponds to a region by
    bbox overlap. Returns the table dict or None.
    """
    region = _region_map(doc_rep).get(region_id)
    if not region:
        return None
    rbbox = region["bbox"]
    rcenter = _region_center(region)

    best = None
    best_ioa = 0.0
    for table in doc_rep.get("tables", []):
        tbbox = table["bbox"]
        if not _point_in_bbox(rcenter, tbbox):
            continue
        inter = _intersection_over_area(rbbox, tbbox)
        if inter > best_ioa:
            best_ioa = inter
            best = table
    return best


def _rows_from_geometry_table(table, width, height, median_text_h=None):
    """
    Convert a cv/model table (cell grid) into the resolver's row structure.
    Cells of the same `row` band become one row, sorted by column; merged cells
    keep their row_span/column_span.

    Text-less leading rows are dropped (grid artifacts above a table's first
    content line); thin text-less trailing rows are dropped as well (phantom
    bands below the last grid line).
    """
    cells = table.get("cells") or []
    if not cells:
        return None

    rows = []
    current_row = None
    for cell in sorted(cells, key=lambda c: (c.get("row", 0), c.get("column", 0))):
        if current_row is None or cell.get("row") != current_row[0]:
            current_row = (cell.get("row", 0), [])
            rows.append(current_row)
        bbox = cell["bbox"]
        current_row[1].append(
            {
                "value": cell.get("text"),
                "element_ids": list(cell.get("ocr_element_ids") or []),
                "bbox": _norm(bbox, width, height),
                "bbox_pixels": bbox,
                "confidence": None,
                "row_span": cell.get("row_span", 1),
                "column_span": cell.get("column_span", 1),
                "cell_id": cell.get("id"),
            }
        )

    if rows and median_text_h:
        thin = 0.6 * median_text_h
        while rows and not _row_has_text(rows[0][1]):
            rows.pop(0)
        while (
            rows
            and not _row_has_text(rows[-1][1])
            and _row_height(rows[-1][1]) < thin
        ):
            rows.pop()

    return [row for _, row in rows]


def _row_has_text(row):
    return any((c.get("value") or "").strip() for c in row)


def _row_height(row):
    bboxes = [c.get("bbox_pixels") for c in row]
    if not bboxes:
        return 0.0
    return max(b[3] for b in bboxes) - min(b[1] for b in bboxes)


def _build_table_from_document(doc_rep, region_id, has_header):
    """
    Prefer the cv/model cell grid for a table; fall back to the geometric
    rows-from-OCR heuristic (_build_table).
    """
    table = _find_geometry_table(doc_rep, region_id)
    width = doc_rep["image_width"]
    height = doc_rep["image_height"]

    if table is not None:
        heights = [e.get("height") for e in doc_rep["elements"] if e.get("height")]
        median_text_h = _median(heights) if heights else None
        rows = _rows_from_geometry_table(table, width, height, median_text_h)
        if rows:
            headers = []
            rows_data = rows
            if has_header and len(rows) > 1:
                headers = [c["value"] for c in rows[0]]
                rows_data = rows[1:]
            return {
                "title": None,
                "headers": headers,
                "rows": rows_data,
                "region_id": region_id,
                "bbox": _norm(table["bbox"], width, height),
                "bbox_pixels": table["bbox"],
                "confidence": table.get("confidence"),
                "uncertain": False,
                "reason": None,
                "structure_source": table.get("structure_source"),
            }

    return _build_table(doc_rep, region_id, has_header)

def _infer_value_type(text):
    stripped = text.strip()

    if re.fullmatch(r"\d{1,4}[-/]\d{1,2}[-/]\d{2,4}", stripped):
        return "date"

    if re.fullmatch(r"\d{1,2}:\d{2}(\s?(AM|PM))?", stripped, re.IGNORECASE):
        return "time"

    if re.fullmatch(r"[+-]?\d[\d,]*([.,]\d+)?%?", stripped):
        return "number"

    if re.fullmatch(
        r"[+-]?\d[\d,]*([.,]\d+)?\s*(kg|cm|m|km|mi|mph|ml|l|hr|min|g|mg|USD|\$|°C|F)",
        stripped,
        re.IGNORECASE,
    ):
        return "measurement"

    return "text"


def _avg_confidence(confidences):
    values = [c for c in confidences if c is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _element_map(doc_rep):
    return {e["id"]: e for e in doc_rep["elements"]}


def _region_map(doc_rep):
    return {r["id"]: r for r in doc_rep["regions"]}


def _checkbox_map(doc_rep):
    return {c["id"]: c for c in doc_rep["checkboxes"]}


def _elements_in_bbox(doc_rep, bbox):
    return [e for e in doc_rep["elements"] if _point_in_bbox(e["center"], bbox)]


def _resolve_sections(llm_data, doc_rep):
    elements = _element_map(doc_rep)
    regions = _region_map(doc_rep)

    sections = []

    raw_sections = llm_data.get("sections") or []

    for section in raw_sections:

        name = str(section.get("name") or "").strip() or "Section"

        region_bboxes = []
        for rid in section.get("region_ids") or []:
            region = regions.get(rid)
            if region:
                region_bboxes.append(region["bbox"])

        section_element_ids = []
        for eid in section.get("element_ids") or []:
            if eid in elements and eid not in section_element_ids:
                section_element_ids.append(eid)

        # Elements whose center lies inside one of the section's regions
        # automatically belong to the section.
        for region in regions.values():
            if region["bbox"] in region_bboxes:
                for e in _elements_in_bbox(doc_rep, region["bbox"]):
                    if e["id"] not in section_element_ids:
                        section_element_ids.append(e["id"])

        sections.append(
            {
                "name": name,
                "region_ids": list(section.get("region_ids") or []),
                "element_ids": section_element_ids,
                "region_bboxes": region_bboxes,
            }
        )

    return sections


def _find_section_for_element(sections, element_id, elements):
    element = elements.get(element_id)
    if not element:
        return 0

    for index, section in enumerate(sections):
        for bbox in section["region_bboxes"]:
            if _point_in_bbox(element["center"], bbox):
                return index

    # Fall back to a section that explicitly lists the element.
    for index, section in enumerate(sections):
        if element_id in section["element_ids"]:
            return index

    return -1


def _build_table(doc_rep, region_id, has_header):
    region = _region_map(doc_rep).get(region_id)
    if not region:
        return None

    items = _elements_in_bbox(doc_rep, region["bbox"])
    if not items:
        return None

    rows = group_into_rows(items, y_tolerance=12)

    # Cluster columns by x-center gaps.
    row_cells = []

    for row in rows:
        cells = []
        current_cell = [row["items"][0]]

        for item in row["items"][1:]:
            prev = current_cell[-1]
            gap = item["center"][0] - (prev["bbox"][2] + prev["bbox"][0]) / 2
            if gap > 0 and gap > prev["width"] * 0.5:
                cells.append(current_cell)
                current_cell = [item]
            else:
                current_cell.append(item)

        cells.append(current_cell)
        row_cells.append(cells)

    width = doc_rep["image_width"]
    height = doc_rep["image_height"]

    normalized_cells = []

    for cells in row_cells:
        normalized_row = []
        for cell in cells:
            bbox = _union_bbox([e["bbox"] for e in cell])
            normalized_row.append(
                {
                    "value": " ".join(e["text"] for e in cell).strip(),
                    "element_ids": [e["id"] for e in cell],
                    "bbox": _norm(bbox, width, height),
                    "bbox_pixels": bbox,
                    "confidence": _avg_confidence([e["confidence"] for e in cell]),
                }
            )
        normalized_cells.append(normalized_row)

    if not normalized_cells:
        return None

    if has_header and len(normalized_cells) > 1:
        headers = [cell["value"] for cell in normalized_cells[0]]
        rows_data = normalized_cells[1:]
    else:
        headers = []
        rows_data = normalized_cells

    return {
        "title": None,
        "headers": headers,
        "rows": rows_data,
        "region_id": region_id,
        "bbox": _norm(region["bbox"], width, height),
        "bbox_pixels": region["bbox"],
        "confidence": None,
        "uncertain": False,
        "reason": None,
    }


def _split_inline_label_value(label_element):
    text = label_element["text"]
    match = re.split(r"\s*[:\uFF1A]\s*", text, maxsplit=1)
    if len(match) == 2 and match[1].strip():
        return match[0].strip(), match[1].strip()
    return text, None


def _split_option_instruction(label):
    """
    Split an option label whose trailing part is actually the question's
    instruction glued onto the option word.

    Recognised pattern: a leading option token(s) followed by an embedded
    instruction clause that starts with "If"/"if" (optionally glued, e.g.
    "IfNO") and contains an action verb:

        "NO IfNO, check one below"        -> ("NO", "IfNO, check one below")
        "YES If YES, check one below"     -> ("YES", "If YES, check one below")

    When no embedded instruction is found the whole string is the label and the
    instruction is None.
    """
    if not label:
        return label, None
    text = label.strip()
    # Instruction clause: "If..." (possibly glued to the option, e.g. IfNO) that
    # contains a directive verb. Confine the search to after the first token so
    # a legitimate leading "IF..." option (unlikely) is not harmed.
    m = re.search(
        r"(?i)\b(if[a-z0-9_-]{0,8}\b[^;]{2,})$",
        text,
    )
    if m:
        instr = m.group(1).strip()
        option = text[: m.start()].strip()
        # Require the instruction to read like a directive (contains an action
        # verb or the word "below" / punctuation), and keep at least one option
        # token left.
        if (
            option
            and instr
            and re.search(r"(?i)check|select|choose|skip|describe|explain|notify|below|go\s+to|provide", instr)
        ):
            return option, instr
    return text, None


def _build_coordinate_table(elements_by_id, section_element_ids, width, height):
    """
    Deterministically build a coordinate table (rows = measurement source such
    as Map / GPS, columns = latitude / longitude) from a coordinate section.

    Many hydro/physical forms record the site coordinates plus the method used
    to obtain them.  The section typically contains:
        - a "Coordinates" style heading,
        - axis headers (e.g. "X Latitude North", "X Longitude West") or raw
          coordinate values (DMS/DDM with a degree symbol),
        - one or more source labels (e.g. "Map", "GPS").

    Returns a table dict (same shape as other resolvers emit) or None when no
    coordinate data is found.
    """
    elems = [e for e in section_element_ids if e in elements_by_id]
    texts = {eid: (elements_by_id[eid].get("text") or "") for eid in elems}

    # Coordinate values: tokens containing a degree symbol or N/S/E/W letter.
    def _is_coord(v):
        return any(ch in v for ch in "°º") or bool(
            re.search(r"[0-9]+\s*[°º'\"]", v)
        )

    # Axis headers
    def _is_lat_hdr(v):
        return bool(re.search(r"\blatitude\b", v, re.I)) or (
            bool(re.search(r"\bnorth\b", v, re.I))
            and not re.search(r"west|east|south", v, re.I)
        )

    def _is_lon_hdr(v):
        return bool(re.search(r"\blongitude\b|\blong\b", v, re.I)) or bool(
            re.search(r"\bwest\b|\beast\b", v, re.I)
        )

    def _center_x(eid):
        b = elements_by_id[eid].get("bbox")
        return (b[0] + b[2]) / 2 if b else None

    def _center_y(eid):
        b = elements_by_id[eid].get("bbox")
        return (b[1] + b[3]) / 2 if b else None

    lat_hdr = lon_hdr = None
    sources = []
    coord_values = []  # (eid, text, x_center, y_center)
    coord_elem_ids = []

    for eid in elems:
        t = texts[eid].strip()
        low = t.lower()
        if _is_coord(t) and re.search(r"[0-9]", t):
            coord_elem_ids.append(eid)
            coord_values.append((eid, t, _center_x(eid), _center_y(eid)))
            continue
        if _is_lat_hdr(t):
            lat_hdr = (eid, t)
        elif _is_lon_hdr(t):
            lon_hdr = (eid, t)
        elif re.match(r"(?i)^map$|^gps$|^gps\b|utm|nav|dlg|survey", t):
            sources.append((eid, t))
            coord_elem_ids.append(eid)

    if not coord_elem_ids:
        return None

    # Classify each coordinate value into a latitude / longitude column.  Prefer
    # the axis-header column whose x lies closest (geometry), then hemisphere
    # letters, then pure numeric (latitudes in DMS stay under ~90, longitudes
    # can exceed 90 so a leading 1xx hints longitude).
    def _classify(eid, t, x):
        hdr_centers = []
        if lat_hdr is not None and _center_x(lat_hdr[0]) is not None:
            hdr_centers.append(("lat", _center_x(lat_hdr[0])))
        if lon_hdr is not None and _center_x(lon_hdr[0]) is not None:
            hdr_centers.append(("lon", _center_x(lon_hdr[0])))
        if x is not None and hdr_centers:
            nearest = min(hdr_centers, key=lambda hc: abs(hc[1] - x))
            return nearest[0]
        low = t.lower()
        if re.search(r"\b[nN]\b|north", low) and not re.search(r"west|east", low):
            return "lat"
        if re.search(r"[wW]\b|west|east|south", low) and not re.search(r"\bn\b|north", low):
            return "lon"
        m = re.search(r"(\d+)\s*[°º]", t)
        if m:
            deg = float(m.group(1))
            return "lat" if deg <= 90 else "lon"
        return "lat"

    per_source = {}
    for eid, t, x, y in coord_values:
        col = _classify(eid, t, x)
        if sources:
            row_key = min(sources, key=lambda s: abs((_center_y(s[0]) or 0) - (y or 0)))
        else:
            row_key = (None, "recorded")
        per_source.setdefault(row_key, {"lat": None, "lon": None})[col] = t

    rows = []
    for (src_id, src), vals in per_source.items():
        # Keep non-empty rows only (a source with no recorded values, e.g. an
        # unchecked alternative, is left out of the table).
        if vals["lat"] is None and vals["lon"] is None:
            continue
        rows.append(
            {
                "source": src,
                "latitude": vals["lat"] or "",
                "longitude": vals["lon"] or "",
            }
        )
    if not rows:
        rows.append(
            {
                "source": sources[0][1] if sources else "(recorded)",
                "latitude": "",
                "longitude": "",
            }
        )

    header = [
        {"label": "Source", "column_key": "source"},
        {"label": lat_hdr[1] if lat_hdr else "Latitude", "column_key": "latitude"},
        {"label": lon_hdr[1] if lon_hdr else "Longitude", "column_key": "longitude"},
    ]

    return {
        "id": "coord_table",
        "kind": "coordinate_table",
        "title": "Coordinates",
        "header": header,
        "rows": [
            [row["source"], row["latitude"], row["longitude"]]
            for row in rows
        ],
        "has_header": True,
        "source": "geometry",
        "element_ids": coord_elem_ids,
    }


_MULTISELECT_MARKERS = (
    r"\(x\s+all\s+that\s+apply\)",
    r"\(check\s+all\s+that\s+apply\)",
    r"\(mark\s+all\s+that\s+apply\)",
    r"\(select\s+all\s+that\s+apply\)",
    r"all\s+that\s+apply",
)

# Common option phrases appearing glued together on physical forms (the OCR
# often collapses them into a single run, e.g. "GPSlocal contactsignsroads").
# Ordered longest-first so multi-word options are matched before their parts.
_MULTISELECT_OPTION_VOCAB = (
    "local contact",
    "topographic map",
    "topo map",
    "local",
    "contact",
    "topo",
    "GPS",
    "signs",
    "roads",
    "road",
    "sign",
    "map",
    "radio",
    "telephone",
    "internet",
    "other",
    "n/a",
    "na",
)

_MULTISELECT_MARKER_RE = re.compile(
    "|".join(_MULTISELECT_MARKERS), re.IGNORECASE
)


def _split_multiselect_question(element_text, vocabulary=None):
    """
    Split a merged question element that says "(x) all that apply" into a
    multi-select question plus its individual options.

    Physical forms often print a multi-check instruction ("(x all that apply)")
    and several options on one line.  OCR commonly glues the options together
    (e.g. "GPSlocal contactsignsroads topo map").  This helper:

      * returns (None, []) when the element is not a multi-select prompt, so
        callers can confidently skip it, and
      * otherwise returns (question_text, [option_text, ...]) by removing the
        lead and scanning a vocabulary over the remainder.

    ``vocabulary`` is an ordered tuple of candidate option phrases; caller may
    supply the domain's known options.  When nothing can be segmented the
    options list is empty and the caller should leave the element as text.
    """
    if not element_text:
        return None, []
    text = element_text.strip()
    pm = _MULTISELECT_MARKER_RE.search(text)
    if not pm:
        return None, []

    vocab = vocabulary if vocabulary else _MULTISELECT_OPTION_VOCAB
    # Ordered longest-first so "local contact" wins over "local".
    vocab = tuple(sorted(vocab, key=len, reverse=True))

    # Question lead = text up to and including the first ':' (the marker lives
    # in the lead, e.g. "verified by (x all that apply):").
    m = pm.start()
    rest = text
    qmark = text.find(":")
    if qmark != -1:
        question = text[:qmark].strip()
        rest = text[qmark + 1 :]
    else:
        question = text[:m].strip()

    # Clean the marker-words out of the question lead so we don't duplicate
    # them into an option.
    options = []
    consumed = rest
    while consumed:
        consumed = consumed.lstrip(" :.,;-_")
        if not consumed:
            break
        matched = None
        for phrase in vocab:
            if consumed.lower().startswith(phrase.lower()):
                matched = phrase
                break
        if matched:
            options.append(matched)
            consumed = consumed[len(matched):]
        else:
            # Remove a single leading token then keep scanning; if no token
            # boundary exists we stop to avoid fabricating options.
            token = re.match(r"[a-zA-Z0-9/]+", consumed)
            if not token:
                break
            consumed = consumed[token.end():]
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for opt in options:
        k = opt.lower()
        if k not in seen:
            seen.add(k)
            unique.append(opt)
    return question, unique