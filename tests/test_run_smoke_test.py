import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_smoke_test import (  # noqa: E402
    inspect_checkpoints,
    package_version,
    summarize_gpu_csv,
)


class SmokeMonitorTest(unittest.TestCase):
    def test_missing_package_has_no_version(self):
        self.assertIsNone(package_version("tree-ai-lab-package-that-does-not-exist"))

    def test_gpu_maxima_are_computed_per_device(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "timestamp",
                        "gpu_index",
                        "memory_used_mib",
                        "memory_total_mib",
                        "utilization_percent",
                        "temperature_celsius",
                        "power_watts",
                    ]
                )
                writer.writerow(["t1", "0", "100", "15000", "20", "50", "30"])
                writer.writerow(["t2", "0", "900", "15000", "80", "60", "60"])
                writer.writerow(["t2", "1", "800", "15000", "70", "60", "60"])

            summary = summarize_gpu_csv(path)

        self.assertEqual(summary["0"]["max_memory_used_mib"], 900.0)
        self.assertEqual(summary["0"]["max_utilization_percent"], 80.0)
        self.assertEqual(summary["1"]["max_memory_used_mib"], 800.0)

    def test_full_distributed_checkpoint_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step = root / "checkpoints" / "step_20"
            for rank in range(2):
                files = [
                    step / "trainer" / f"rank_{rank:02d}.pt",
                    step / "model" / "pp_00" / "tp_00" / f"sdp_{rank:02d}.pt",
                    step
                    / "optimizer"
                    / "pp_00"
                    / "tp_00"
                    / f"sdp_{rank:02d}.pt",
                    step / "data_reader" / f"dp_{rank:02d}.pt",
                ]
                for path in files:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

            checkpoints = inspect_checkpoints(root, num_gpus=2)

        self.assertEqual(checkpoints[0]["step"], 20)
        self.assertTrue(checkpoints[0]["complete"])


if __name__ == "__main__":
    unittest.main()
