"""
Synthetic-form / table generators for the geometry regression tests.

Everything is rendered with PIL (anti-aliased TrueType) and returned as BGR
ndarrays so the pipeline functions accept them directly.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def _font(size):
    return ImageFont.truetype(FONT_PATH, size)


def to_np_bgr(pil_image):
    rgb = np.asarray(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_leave_application(seed=0):
    """
    A synthetic leave-application form exercising the classic
    "Period From ____ To ____" underline layout and a short leave-days field.

    Layout (printed labels on one baseline, underlines under their value):
        LEAVE APPLICATION
        Employee Name    ______________________ (Rahul Sharma)
        Department       ______________________ (Engineering)
        Leave Period From ____01-06-2026___ To ____05-06-2026____
        Total Leave Days _____________________ (5)
    """
    W, H = 1100, 1100
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    font_label = _font(30)
    font_value = _font(32)
    font_title = _font(42)

    draw.text((W // 2 - 220, 50), "LEAVE APPLICATION", font=font_title, fill=(0, 0, 0))

    def field_row(label, value, y, underline_x, value_x=None, value_w=420, label_x=90):
        draw.text((label_x, y), label, font=font_label, fill=(0, 0, 0))
        line_y = y + 40
        draw.line([(underline_x, line_y), (underline_x + value_w, line_y)],
                  fill=(0, 0, 0), width=5)
        vx = value_x if value_x is not None else underline_x + 30
        draw.text((vx, y + 4), value, font=font_value, fill=(0, 0, 0))
        return line_y

    y = 180
    field_row("Employee Name", "Rahul Sharma", y, 320, value_x=340)
    field_row("Department", "Engineering", y + 100, 300, value_x=320)

    # Leave Period row: two underline fields sharing one baseline.
    row_y = y + 200
    draw.text((90, row_y), "Leave Period From", font=font_label, fill=(0, 0, 0))
    from_x = 500
    line_y = row_y + 40
    draw.line([(from_x, line_y), (from_x + 200, line_y)], fill=(0, 0, 0), width=5)
    draw.text((from_x + 35, row_y + 4), "01-06-2026", font=font_value, fill=(0, 0, 0))

    to_label_x = from_x + 280
    draw.text((to_label_x, row_y), "To", font=font_label, fill=(0, 0, 0))
    to_x = to_label_x + 60
    draw.line([(to_x, line_y), (to_x + 200, line_y)], fill=(0, 0, 0), width=5)
    draw.text((to_x + 35, row_y + 4), "05-06-2026", font=font_value, fill=(0, 0, 0))

    field_row("Total Leave Days", "5", y + 300, 350, value_x=370, value_w=320)

    return to_np_bgr(img)


def make_wired_table(cols=3, data_rows=3, header=("Item", "Qty", "Price"),
                     cell_texts=None, weak=False):
    """
    A bordered table with a header row + data rows.

    weak=True drops the interior vertical separators below the header so the
    body cells merge horizontally (weak-line scenario). The header always
    keeps its own separators.
    """
    img = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(img)

    cell_w = 260
    cell_h = 80
    top = 120
    left = 60
    header_h = 80
    total_w = cols * cell_w
    total_h = header_h + data_rows * cell_h
    bottom = top + total_h
    right = left + total_w

    line_w = 4

    # Horizontal grid lines.
    for row_i in range(data_rows + 1):
        y = top + row_i * cell_h
        draw.line([(left, y), (right, y)], fill=(0, 0, 0), width=line_w)
    draw.line([(left, bottom), (right, bottom)], fill=(0, 0, 0), width=line_w)

    # Vertical grid lines. Interior lines below the header can be dropped
    # (weak-line / merged-body scenario).
    for col_i in range(cols + 1):
        x = left + col_i * cell_w
        y_stop = bottom
        if weak and col_i != 0 and col_i != cols:
            y_stop = top + header_h
        draw.line([(x, top), (x, y_stop)], fill=(0, 0, 0), width=line_w)

    font = _font(34)

    if header:
        for c, text in enumerate(header):
            draw.text((left + c * cell_w + 60, top), text, font=font, fill=(0, 0, 0))

    for r in range(data_rows):
        y0 = top + header_h + r * cell_h
        for c in range(cols):
            text = " " if not cell_texts else cell_texts[r * cols + c]
            draw.text((left + c * cell_w + 40, y0 + 15), text, font=font, fill=(0, 0, 0))

    return to_np_bgr(img)


def default_table_texts(cols=3, data_rows=3):
    names = ["Widget", "Gadget", "Sprocket", "Cog", "Lever", "Pulley",
             "Bracket", "Spring", "Hinge"]
    rows = []
    for r in range(data_rows):
        for c in range(cols):
            if c == 0:
                rows.append(names[r])
            elif c == 1:
                rows.append(str(r + 1))
            else:
                rows.append(f"{float(r + 1) * 4:.2f}")
    return rows


def make_weak_header_table():
    """
    One merged header cell ("Details") spanning 3 columns over a 3-column body
    with a single row. Tests merged-cell structure recognition.
    """
    img = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(img)
    font = _font(34)

    left, top = 60, 120
    cell_w, cell_h = 260, 80
    cols, rows = 3, 1
    header_h = 80
    right = left + cols * cell_w
    bottom = top + header_h + rows * cell_h

    line_w = 4

    # Horizontal boundaries (header bottom + body rows + bottom border).
    for y in [top, top + header_h, bottom]:
        draw.line([(left, y), (right, y)], fill=(0, 0, 0), width=line_w)

    # Outer vertical edges full height; no interior verticals below the header.
    for x in [left, right]:
        draw.line([(x, top), (x, bottom)], fill=(0, 0, 0), width=line_w)

    # Header text (merged, centered across the full width).
    draw.text((left + 250, top), "Details", font=font, fill=(0, 0, 0))

    # Body row: fully wired columns.
    y0 = top + header_h
    for c in range(cols + 1):
        x = left + c * cell_w
        draw.line([(x, y0), (x, bottom)], fill=(0, 0, 0), width=line_w)
        if c < cols:
            draw.text(
                (left + c * cell_w + 20, y0 + 15),
                ["Item", "Qty", "Price"][c],
                font=font,
                fill=(0, 0, 0),
            )

    return to_np_bgr(img)