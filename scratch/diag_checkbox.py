"""
Checkbox detection diagnostic.

Run against the exact image that is failing, e.g.

    venv/bin/python scratch/diag_checkbox.py path/to/mobile_form.png

Reports, stage by stage, where checkbox detection dies, so we can tell apart
the usual causes:

  1. perspective correction warped/cropped the top of the image
  2. OCR produced no text element for the option labels (nothing to anchor to)
  3. the left-of-label sampling window never reaches the box
  4. boxes were found but the final "checked-only" filter dropped them
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from util.perspective_utils import correct_perspective
from util.checkbox_utils import detect_checkboxes_visual, _classify_window
from service.detection_service import detect_checkboxes_ocr_anchored
from util.checkbox_group_utils import detect_checkbox_groups
from service.ocr_service import get_ocr_data


def rule(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"could not read image: {path}")

    rule("1. INPUT")
    h, w = img.shape[:2]
    print(f"path        : {path}")
    print(f"size        : {w} x {h} ({'screenshot-like' if h > 1.4 * w else 'page-like'})")

    rule("2. PERSPECTIVE CORRECTION (screenshots often trigger this wrongly)")
    corrected, pc = correct_perspective(img)
    p = pc["perspective_correction"]
    print(f"applied     : {p['applied']}   reason: {p.get('reason')}")
    print(f"confidence  : {p.get('confidence')}")
    if p["applied"]:
        print(f"corners     : {p.get('corners')}")
        print(f"target size : {p.get('target_size')}  (was {w} x {h})")
        scale_x = p["target_size"][0] / w
        print(f"scale       : x{scale_x:.3f}  <- below ~0.7 shrinks small UI boxes a lot")
        if p.get("corners"):
            top_cut = min(c[1] for c in p["corners"])
            print(f"top crop    : {top_cut:.0f}px cut off the top of the form")
    image = corrected

    rule("3. OCR (checkboxes are anchored to OCR text)")
    elements = get_ocr_data(image)
    texts = [e for e in elements if e.get("type") == "text"]
    print(f"text elements: {len(texts)}")
    if not texts:
        print("  -> no OCR text at all, nothing can be anchored. Check the image itself.")
        return

    heights = [e.get("height") or (e["bbox"][3] - e["bbox"][1]) for e in texts]
    med = float(np.median(heights))
    print(f"median text h: {med:.1f}px")
    print(f"sample window reach left of a label: "
          f"{min(max(med, 12), 42) * 1.16:.0f}px  <- a box further left than this is never sampled")

    top = sorted(texts, key=lambda e: e["bbox"][1])[:8]
    print("\nfirst 8 text elements (top of form):")
    for e in top:
        print(f"  {str(e.get('text'))[:44]:<46} bbox={[int(v) for v in e['bbox']]}")

    rule("4. PIXEL SAMPLER (before the final filter)")
    raw = detect_checkboxes_visual(image, ocr_elements=elements)
    print(f"raw candidates: {len(raw)}")
    for c in raw[:10]:
        print(f"  state={c['state']:<9} mark={c['mark_type']:<7} conf={c['confidence']:<5} "
              f"bbox={[int(v) for v in c['bbox']]}  <- {str(c.get('associated_text'))[:30]}")

    if not raw:
        print("\n  -> the sampler found nothing. Either the boxes sit further left than")
        print("     the reach above, or they are too thin/light to survive")
        print("     adaptiveThreshold(blockSize=31, C=5).")
        print("     Probing the window just left of each of the top 8 labels:")
        for e in top:
            x1, y1, x2, y2 = [float(v) for v in e["bbox"]]
            th = max(y2 - y1, 10)
            bw = min(max(int(th * 1.0), 12), 42)
            pad = max(2, int(th * 0.16))
            win = (int(x1 - bw - pad), int(y1 - pad), int(x1 - pad), int(y2 + pad))
            r = _classify_window(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                 if len(image.shape) == 3 else image, *win)
            label = r and f"edge_dark={r[0]:.2f} inner_dark={r[1]:.2f}"
            print(f"    reach={bw + pad:>3}px  window={win}  {label or 'window too small'}  "
                  f"<- {str(e.get('text'))[:28]}")

    rule("5. FINAL FILTER (what doc_rep['checkboxes'] actually gets)")
    final = detect_checkboxes_ocr_anchored(image, elements)
    print(f"kept         : {len(final)}   <- THIS is the 'Checkboxes: N' log line")
    if raw and not final:
        print("  -> every candidate was dropped here. The filter keeps only")
        print("     state == 'checked' or source == 'ocr_mark', so unchecked /")
        print("     uncertain rings from a BLANK form are all discarded.")

    rule("6. CHECKBOX GROUPS (separate detector)")
    groups = detect_checkbox_groups(image, elements=elements)
    print(f"groups       : {len(groups)}   <- this is the 'Checkbox groups: N' log line")
    for g in groups[:6]:
        print(f"  bbox={[int(v) for v in g['bbox']]}  options={[o.get('text') for o in g.get('options', [])][:5]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
