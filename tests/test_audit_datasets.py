import io
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ModuleNotFoundError:  # permet aux tests textuels de tourner sans extras audio
    np = None
    sf = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if np is not None and sf is not None:
    from src.audit_datasets import (  # noqa: E402
        AuditThresholds,
        DatasetSpec,
        _duplicate_groups,
        audit_example,
    )


@unittest.skipUnless(np is not None and sf is not None, "extras audio non installés")
class AuditDatasetsTest(unittest.TestCase):
    def test_decodes_audio_bytes_and_normalizes_text(self):
        sample_rate = 16_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = 0.1 * np.sin(2 * np.pi * 220 * time)
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")

        spec = DatasetSpec(
            source="synthetic",
            dataset_id="local/test",
            config=None,
            revision="test",
            text_column="sentence",
            speaker_column="client_id",
            id_column="path",
        )
        item = {
            "audio": {"bytes": buffer.getvalue(), "path": None},
            "sentence": "Nʼdrɛ ɔ kɔ.",
            "client_id": "speaker-1",
            "path": "sample.wav",
        }
        row = audit_example(
            item, 0, "train", spec, AuditThresholds(min_duration_seconds=0.5)
        )

        self.assertTrue(row["decode_ok"])
        self.assertEqual(row["sample_rate"], sample_rate)
        self.assertEqual(row["channels"], 1)
        self.assertEqual(row["text_normalized"], "n'drɛ ɔ kɔ")
        self.assertEqual(row["status"], "accept")
        self.assertEqual(row["preprocessing_actions"], "")
        self.assertGreater(float(row["words_per_second"]), 0.0)

    def test_duplicate_is_cross_split_only_when_partition_changes(self):
        base = {
            "audio_sha256": "abc",
            "text_sha256": "def",
            "text_normalized": "texte",
            "row_index": 0,
            "example_id": "id",
            "speaker_id": "speaker",
        }
        same_split = [
            {**base, "source": "waxal", "split": "train"},
            {**base, "source": "klayt", "split": "train", "row_index": 1},
        ]
        cross_split = [
            same_split[0],
            {**base, "source": "klayt", "split": "test", "row_index": 1},
        ]

        self.assertFalse(_duplicate_groups(same_split, "audio_sha256")[0]["cross_split"])
        self.assertTrue(_duplicate_groups(cross_split, "audio_sha256")[0]["cross_split"])


if __name__ == "__main__":
    unittest.main()
