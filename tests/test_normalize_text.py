import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize_text import character_inventory, normalize_baoule_text  # noqa: E402


class NormalizeBaouleTextTest(unittest.TestCase):
    def test_harmonizes_apostrophes(self):
        self.assertEqual(
            normalize_baoule_text("Nʼdrɛ n’dɛ n‘zué"),
            "n'drɛ n'dɛ n'zué",
        )

    def test_preserves_baoule_letters_and_tones(self):
        self.assertEqual(
            normalize_baoule_text("Ɛ́, ɔ̀, ŋ et ɲ !"),
            "ɛ́ ɔ̀ ŋ et ɲ",
        )

    def test_removes_unpronounced_annotation_and_punctuation(self):
        self.assertEqual(
            normalize_baoule_text("Bla [rire] : ɔ kɔ !"),
            "bla ɔ kɔ",
        )

    def test_removes_only_standalone_numbers(self):
        self.assertEqual(normalize_baoule_text("abc2 123 n2"), "abc2 n2")

    def test_normalizes_decomposed_unicode_to_nfc(self):
        self.assertEqual(normalize_baoule_text("E\u0301"), "é")

    def test_character_inventory_ignores_spaces(self):
        self.assertEqual(character_inventory(["a b", "ba"]), {"a": 2, "b": 2})


if __name__ == "__main__":
    unittest.main()

