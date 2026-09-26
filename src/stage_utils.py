"""Utilitaires sûrs pour reprendre des blocs FSDP depuis une sortie Kaggle."""

from __future__ import annotations

import shutil
from pathlib import Path


def checkpoint_state_counts(step_dir: Path) -> dict[str, int]:
    return {
        "trainer": len(list((step_dir / "trainer").glob("rank_*.pt"))),
        "model": len(list((step_dir / "model").rglob("sdp_*.pt"))),
        "optimizer": len(list((step_dir / "optimizer").rglob("sdp_*.pt"))),
        "data_reader": len(list((step_dir / "data_reader").glob("dp_*.pt"))),
    }


def is_complete_checkpoint(step_dir: Path, num_gpus: int = 2) -> bool:
    counts = checkpoint_state_counts(step_dir)
    return all(count >= num_gpus for count in counts.values())


def find_checkpoint_dirs(
    input_root: Path, step: int, num_gpus: int = 2
) -> list[Path]:
    candidates: list[Path] = []
    for step_dir in input_root.rglob(f"step_{step}"):
        if (
            step_dir.is_dir()
            and (step_dir.parent / "model.yaml").is_file()
            and is_complete_checkpoint(step_dir, num_gpus)
        ):
            candidates.append(step_dir.parent)
    return sorted(set(candidates))


def prepare_resume_view(
    source_checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    step: int,
    num_gpus: int = 2,
) -> Path:
    source_checkpoint_dir = source_checkpoint_dir.resolve()
    source_step = source_checkpoint_dir / f"step_{step}"
    if not (source_checkpoint_dir / "model.yaml").is_file():
        raise FileNotFoundError(source_checkpoint_dir / "model.yaml")
    if not is_complete_checkpoint(source_step, num_gpus):
        raise ValueError(
            f"Checkpoint incomplet : {source_step} — "
            f"{checkpoint_state_counts(source_step)}"
        )

    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_step = output_checkpoint_dir / source_step.name
    if target_step.exists() or target_step.is_symlink():
        raise FileExistsError(target_step)
    target_step.symlink_to(source_step, target_is_directory=True)
    shutil.copy2(
        source_checkpoint_dir / "model.yaml", output_checkpoint_dir / "model.yaml"
    )

    source_score = source_checkpoint_dir / "scores" / f"step_{step}.txt"
    if source_score.is_file():
        score_dir = output_checkpoint_dir / "scores"
        score_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_score, score_dir / source_score.name)
    return target_step


def detach_resume_link(link: Path | None) -> None:
    if link is None:
        return
    if not link.is_symlink():
        raise ValueError(f"Le chemin de reprise n'est pas un lien symbolique : {link}")
    link.unlink()
