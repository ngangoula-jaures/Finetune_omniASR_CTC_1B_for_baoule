#!/usr/bin/env python3
"""Lance le smoke test OmniASR distribué et mesure les ressources GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--expected-steps", type=int, default=20)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-revision")
    return parser.parse_args()


def gpu_count() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return len([line for line in output.splitlines() if line.strip()])


def monitor_gpus(path: Path, stop: threading.Event, interval: float) -> None:
    fields = [
        "timestamp",
        "index",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
    ]
    query = ",".join(fields)
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
        while not stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.STDOUT,
                )
                for line in output.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) == len(fields):
                        writer.writerow(values)
                handle.flush()
            except Exception as error:
                writer.writerow(
                    [datetime.now(timezone.utc).isoformat(), "monitor_error", str(error)]
                )
                handle.flush()
            stop.wait(interval)


def summarize_gpu_csv(path: Path) -> dict[str, dict[str, float]]:
    maxima: dict[str, dict[str, float]] = defaultdict(
        lambda: {"max_memory_used_mib": 0.0, "max_utilization_percent": 0.0}
    )
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = row.get("gpu_index", "")
            if not index.isdigit():
                continue
            try:
                maxima[index]["max_memory_used_mib"] = max(
                    maxima[index]["max_memory_used_mib"],
                    float(row["memory_used_mib"]),
                )
                maxima[index]["max_utilization_percent"] = max(
                    maxima[index]["max_utilization_percent"],
                    float(row["utilization_percent"]),
                )
            except (TypeError, ValueError):
                continue
    return dict(sorted(maxima.items()))


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def inspect_checkpoints(output_dir: Path, num_gpus: int) -> list[dict[str, object]]:
    checkpoints: list[dict[str, object]] = []
    for step_dir in sorted(output_dir.rglob("step_*")):
        if not step_dir.is_dir() or step_dir.name.endswith(".tmp"):
            continue
        suffix = step_dir.name.removeprefix("step_")
        if not suffix.isdigit():
            continue
        counts = {
            "trainer": len(list((step_dir / "trainer").glob("rank_*.pt"))),
            "model": len(list((step_dir / "model").rglob("sdp_*.pt"))),
            "optimizer": len(list((step_dir / "optimizer").rglob("sdp_*.pt"))),
            "data_reader": len(
                list((step_dir / "data_reader").glob("dp_*.pt"))
            ),
        }
        checkpoints.append(
            {
                "step": int(suffix),
                "path": str(step_dir.relative_to(output_dir)),
                "state_file_counts": counts,
                "complete": all(count >= num_gpus for count in counts.values()),
            }
        )
    return checkpoints


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    meta_repo = args.meta_repo.resolve()
    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if not (meta_repo / "workflows" / "recipes" / "wav2vec2" / "asr").is_dir():
        raise FileNotFoundError(f"Dépôt OmniASR invalide : {meta_repo}")
    if not config.is_file():
        raise FileNotFoundError(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Le dossier de sortie n'est pas vide : {output_dir}. "
            "Utilise un nouveau dossier pour ne pas mélanger deux essais."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    detected_gpus = gpu_count()
    if detected_gpus != args.num_gpus:
        raise RuntimeError(
            f"{args.num_gpus} GPU attendus, {detected_gpus} détectés."
        )

    log_path = output_dir / "smoke_console.log"
    monitor_path = output_dir / "gpu_metrics.csv"
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
    disk_after = shutil.disk_usage(output_dir.parent).free
    output_size = directory_size(output_dir)
    shard_paths = sorted(output_dir.rglob("sdp_*.pt"))
    checkpoints = inspect_checkpoints(output_dir, args.num_gpus)
    gpu_names = command_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python_version": sys.version,
        "package_versions": {
            "torch": package_version("torch"),
            "torchaudio": package_version("torchaudio"),
            "fairseq2": package_version("fairseq2"),
            "omnilingual-asr": package_version("omnilingual-asr"),
        },
        "meta_commit": command_output(["git", "rev-parse", "HEAD"], cwd=meta_repo),
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "gpu_names": gpu_names.splitlines() if gpu_names else [],
        "return_code": return_code,
        "expected_steps": args.expected_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_expected_step_including_validation": (
            elapsed / args.expected_steps if args.expected_steps else None
        ),
        "rough_estimated_minutes_500_steps": (
            elapsed / args.expected_steps * 500 / 60
            if args.expected_steps
            else None
        ),
        "rough_estimated_hours_5000_steps": (
            elapsed / args.expected_steps * 5000 / 3600
            if args.expected_steps
            else None
        ),
        "gpu_maxima": summarize_gpu_csv(monitor_path),
        "checkpoint_shards": [str(path.relative_to(output_dir)) for path in shard_paths],
        "checkpoints": checkpoints,
        "output_size_bytes": output_size,
        "disk_free_before_bytes": disk_before,
        "disk_free_after_bytes": disk_after,
    }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
