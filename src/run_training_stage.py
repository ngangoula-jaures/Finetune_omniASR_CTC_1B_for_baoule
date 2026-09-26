#!/usr/bin/env python3
"""Lance un bloc distribué CTC-1B et archive ses mesures d'exécution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .run_smoke_test import (
        directory_size,
        gpu_count,
        inspect_checkpoints,
        monitor_gpus,
        summarize_gpu_csv,
    )
except ImportError:
    from run_smoke_test import (  # type: ignore
        directory_size,
        gpu_count,
        inspect_checkpoints,
        monitor_gpus,
        summarize_gpu_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--expected-start-step", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-revision")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_validation_wers(log_text: str) -> list[dict[str, float | int]]:
    return [
        {"step": int(step), "wer": float(wer)}
        for step, wer in re.findall(
            r"Validation Metrics \(step (\d+)\).*?Word\s+Error\s+Rate \(WER\):\s*([0-9.]+)",
            log_text,
            flags=re.DOTALL,
        )
    ]


def main() -> int:
    args = parse_args()
    if args.target_step <= args.expected_start_step:
        raise ValueError("target-step doit être supérieur à expected-start-step.")
    meta_repo = args.meta_repo.resolve()
    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if not (meta_repo / "workflows" / "recipes" / "wav2vec2" / "asr").is_dir():
        raise FileNotFoundError(f"Dépôt OmniASR invalide : {meta_repo}")
    if not config.is_file():
        raise FileNotFoundError(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if gpu_count() != args.num_gpus:
        raise RuntimeError(f"Il faut exactement {args.num_gpus} GPU visibles.")

    before = inspect_checkpoints(output_dir, args.num_gpus)
    complete_before = sorted(
        int(item["step"]) for item in before if bool(item["complete"])
    )
    if args.expected_start_step == 0:
        if complete_before:
            raise RuntimeError(f"Sortie déjà entraînée : {complete_before}")
    elif args.expected_start_step not in complete_before:
        raise RuntimeError(
            f"Checkpoint complet du pas {args.expected_start_step} absent : {before}"
        )

    label = f"stage_{args.expected_start_step:05d}_{args.target_step:05d}"
    log_path = output_dir / f"{label}_console.log"
    monitor_path = output_dir / f"{label}_gpu_metrics.csv"
    summary_path = output_dir / f"{label}_summary.json"
    for path in (log_path, monitor_path, summary_path):
        if path.exists():
            raise FileExistsError(path)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.num_gpus}",
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir),
        "--config-file",
        str(config),
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(args.num_gpus))),
        "OMP_NUM_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    }
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpus,
        args=(monitor_path, stop, args.monitor_interval),
        daemon=True,
    )
    disk_before = shutil.disk_usage(output_dir.parent).free
    started_at = datetime.now(timezone.utc)
    start = time.monotonic()
    monitor.start()
    return_code = -1
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=meta_repo,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
    finally:
        stop.set()
        monitor.join(timeout=max(5.0, args.monitor_interval * 2))

    elapsed = time.monotonic() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    checkpoints = inspect_checkpoints(output_dir, args.num_gpus)
    target = [item for item in checkpoints if item["step"] == args.target_step]
    target_complete = bool(target and target[0]["complete"])
    effective_code = return_code if return_code != 0 or target_complete else 3
    validation_wers = parse_validation_wers(log_text)
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": return_code,
        "effective_return_code": effective_code,
        "command": command,
        "start_step": args.expected_start_step,
        "target_step": args.target_step,
        "block_steps": args.target_step - args.expected_start_step,
        "elapsed_seconds": elapsed,
        "seconds_per_block_step_including_fixed_costs": elapsed
        / (args.target_step - args.expected_start_step),
        "overflow_count": len(re.findall(r"Overflow detected at step", log_text)),
        "validation_wers": validation_wers,
        "gpu_maxima": summarize_gpu_csv(monitor_path),
        "checkpoints": checkpoints,
        "target_checkpoint_complete": target_complete,
        "output_size_bytes": directory_size(output_dir),
        "disk_free_before_bytes": disk_before,
        "disk_free_after_bytes": shutil.disk_usage(output_dir.parent).free,
        "python_version": sys.version,
        "package_versions": {
            "torch": package_version("torch"),
            "torchaudio": package_version("torchaudio"),
            "fairseq2": package_version("fairseq2"),
            "omnilingual-asr": package_version("omnilingual-asr"),
        },
        "meta_commit": git_commit(meta_repo),
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return effective_code


if __name__ == "__main__":
    raise SystemExit(main())
