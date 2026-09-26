import tempfile
import unittest
from pathlib import Path

from src.stage_utils import (
    detach_resume_link,
    find_checkpoint_dirs,
    is_complete_checkpoint,
    prepare_resume_view,
)


def create_checkpoint(root: Path, step: int, ranks: int = 2) -> Path:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "model.yaml").write_text("name: checkpoint\n", encoding="utf-8")
    step_dir = checkpoints / f"step_{step}"
    for rank in range(ranks):
        paths = [
            step_dir / "trainer" / f"rank_{rank:02d}.pt",
            step_dir / "model" / "pp_00" / "tp_00" / f"sdp_{rank:02d}.pt",
            step_dir / "optimizer" / "pp_00" / "tp_00" / f"sdp_{rank:02d}.pt",
            step_dir / "data_reader" / f"dp_{rank:02d}.pt",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return checkpoints


class StageUtilsTest(unittest.TestCase):
    def test_complete_checkpoint_can_be_discovered_and_linked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = create_checkpoint(root / "input", 250)
            candidates = find_checkpoint_dirs(root / "input", 250)
            self.assertEqual(candidates, [source])
            output = root / "working" / "checkpoints"
            link = prepare_resume_view(source, output, 250)
            self.assertTrue(link.is_symlink())
            self.assertTrue(is_complete_checkpoint(link))
            detach_resume_link(link)
            self.assertFalse(link.exists())

    def test_incomplete_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = create_checkpoint(root / "input", 250, ranks=1)
            self.assertFalse(is_complete_checkpoint(checkpoints / "step_250"))
            self.assertEqual(find_checkpoint_dirs(root / "input", 250), [])


if __name__ == "__main__":
    unittest.main()
