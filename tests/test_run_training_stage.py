import unittest

from src.run_training_stage import parse_validation_wers


class TrainingStageTest(unittest.TestCase):
    def test_wrapped_validation_wer_is_parsed(self):
        log = """Validation Metrics (step 50) - CTC Loss: 42.0 | Word
                 Error Rate (WER): 38.25 | Data Time: <1s
                 Validation Metrics (step 100) - CTC Loss: 40.0 | Word Error Rate (WER): 35.5"""
        self.assertEqual(
            parse_validation_wers(log),
            [{"step": 50, "wer": 38.25}, {"step": 100, "wer": 35.5}],
        )


if __name__ == "__main__":
    unittest.main()
