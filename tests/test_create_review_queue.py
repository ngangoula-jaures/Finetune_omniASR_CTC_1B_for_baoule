import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.create_review_queue import build_manifests, suggestion_for  # noqa: E402


def row(source="waxal", split="train", reasons="", status="accept"):
    return {
        "source": source,
        "split": split,
        "review_reasons": reasons,
        "decode_ok": "True",
        "status": status,
    }


class ReviewSuggestionTest(unittest.TestCase):
    def test_cross_split_train_duplicate_is_excluded(self):
        priority, decision, _ = suggestion_for(
            row(reasons="text_duplicate_cross_split"), True
        )
        self.assertEqual((priority, decision), (0, "exclude"))

    def test_long_waxal_is_segmented(self):
        priority, decision, _ = suggestion_for(
            row(reasons="too_long_requires_alignment"), False
        )
        self.assertEqual((priority, decision), (3, "segment"))

    def test_clean_klayt_still_requires_listening(self):
        priority, decision, _ = suggestion_for(
            row(source="klayt"), False
        )
        self.assertEqual((priority, decision), (2, "review"))

    def test_clean_waxal_is_kept(self):
        priority, decision, _ = suggestion_for(row(), False)
        self.assertEqual((priority, decision), (9, "keep"))

    def test_manifest_contains_manual_edit_fields(self):
        audit_row = {
            **row(),
            "row_index": "12",
            "example_id": "sample-12",
            "speaker_id": "waxal:JK",
            "duration_seconds": "7.5",
            "words_per_second": "1.2",
            "rms_dbfs": "-24.0",
            "leading_silence_seconds": "0.4",
            "trailing_silence_seconds": "0.6",
            "text_raw": "Texte brut.",
            "text_normalized": "texte brut",
            "preprocessing_actions": "resample_to_16khz",
        }

        manifest, queue = build_manifests([audit_row], set())

        self.assertEqual(queue, [])
        self.assertEqual(manifest[0]["text_raw"], "Texte brut.")
        self.assertIn("segment_boundaries_seconds", manifest[0])


if __name__ == "__main__":
    unittest.main()
