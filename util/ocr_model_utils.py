from paddleocr import PaddleOCR
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_ocr_engine():
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang="en",
    )


@lru_cache(maxsize=1)
def _get_orientation_model():
    from paddlex import create_model

    return create_model("PP-LCNet_x1_0_doc_ori")