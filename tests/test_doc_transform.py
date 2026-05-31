from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.doc_transform import extract_rewritten_document


class ExtractRewrittenDocumentTest(unittest.TestCase):
    def test_extracts_inner_answer_tag(self) -> None:
        self.assertEqual(
            extract_rewritten_document("ignored prefix <answer>rewritten doc</answer> ignored suffix"),
            "rewritten doc",
        )

    def test_falls_back_to_raw_text_when_answer_tag_is_missing(self) -> None:
        self.assertEqual(extract_rewritten_document("rewritten doc"), "rewritten doc")


if __name__ == "__main__":
    unittest.main()
