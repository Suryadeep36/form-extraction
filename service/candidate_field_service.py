from util.candidate_fields_utils import (
    _is_label_like,
    _relationship_score
)

def generate_field_candidates(doc_rep, max_candidates=5, min_score=1.2):
    elements = doc_rep["elements"]
    regions = doc_rep["regions"]
    image_width = doc_rep["image_width"]
    image_height = doc_rep["image_height"]

    candidates = []

    for label in elements:

        if not _is_label_like(label):
            continue

        scored = []

        for value in elements:

            if value["id"] == label["id"]:
                continue

            if _is_label_like(value) and (
                abs(value["center"][1] - label["center"][1]) <= 8
            ):
                # Two label-like elements on the same row: unlikely a pair.
                continue

            score = _relationship_score(
                label, value, regions, image_width, image_height
            )

            if score >= min_score:
                scored.append(
                    {
                        "value_id": value["id"],
                        "score": round(score, 3),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)

        if scored:
            candidates.append(
                {
                    "label_id": label["id"],
                    "candidates": scored[:max_candidates],
                }
            )

    return candidates