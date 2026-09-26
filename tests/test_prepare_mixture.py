import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_mixture import selection_for  # noqa: E402


def row(**overrides):
    result = {
        "source": "klayt",
        "duration_seconds": "8.0",
        "text_normalized": "texte valide",
        "audit_status": "accept",
        "suggested_decision": "review",
        "final_decision": "",
        "trim_start_seconds": "",
        "trim_end_seconds": "",
    }
    result.update(overrides)
    return result


class HackathonSelectionTest(unittest.TestCase):
    def test_manual_exclusion_always_wins(self):
        result = selection_for(
            row(source="waxal", final_decision="exclude"), 40.0
        )
        self.assertFalse(result.include)
        self.assertEqual(result.reason, "manual_exclude")

    def test_unreviewed_short_waxal_is_selected(self):
        result = selection_for(row(source="waxal", duration_seconds="39.9"), 40.0)
        self.assertTrue(result.include)
        self.assertEqual(result.reason, "waxal_short_hackathon_policy")

    def test_unreviewed_klayt_is_deferred(self):
        result = selection_for(row(source="klayt"), 40.0)
        self.assertFalse(result.include)
        self.assertEqual(result.reason, "manual_review_pending")

    def test_long_waxal_is_deferred_for_segmentation(self):
        result = selection_for(row(source="waxal", duration_seconds="40.1"), 40.0)
        self.assertFalse(result.include)
        self.assertEqual(result.reason, "waxal_requires_segmentation")

    def test_automatic_keep_is_selected(self):
        result = selection_for(
            row(final_decision="keep", suggested_decision="keep"), 40.0
        )
        self.assertTrue(result.include)
        self.assertEqual(result.reason, "validated_automatic")

    def test_valid_trim_is_selected(self):
        result = selection_for(
            row(
                duration_seconds="45",
                final_decision="trim",
                trim_start_seconds="4",
                trim_end_seconds="39",
            ),
            40.0,
        )
        self.assertTrue(result.include)
        self.assertEqual(result.trim_start_seconds, 4.0)
        self.assertEqual(result.trim_end_seconds, 39.0)


if __name__ == "__main__":
    unittest.main()
