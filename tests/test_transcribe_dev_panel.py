import unittest

from src.transcribe_dev_panel import edit_distance


class TranscribePanelTest(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("a b c".split(), "a x c".split()), 1)
        self.assertEqual(edit_distance([], ["a", "b"]), 2)
        self.assertEqual(edit_distance(["a", "b"], []), 2)


if __name__ == "__main__":
    unittest.main()
