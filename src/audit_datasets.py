#!/usr/bin/env python3
"""Audit reproductible de Waxal bau_tts et Klayt Baoulé Common Voice.

Le script charge l'audio sans torchcodec, le décode avec soundfile et produit
des manifests légers. Il ne prépare pas encore les fichiers MixtureParquet :
les exemples signalés doivent d'abord être examinés.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    from .normalize_text import (
        character_inventory,
        normalize_baoule_text,
        unicode_description,
    )
except ImportError:
    from normalize_text import (  # type: ignore
        character_inventory,
        normalize_baoule_text,
        unicode_description,
    )


DEFAULT_WAXAL_REVISION = "5f4d8ca24f2b9d168b2ee545f1febaaff4b40580"
DEFAULT_KLAYT_REVISION = "96eb4413b7cc3190c39a35ee03eccb321c2b9929"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    dataset_id: str
    config: str | None
    revision: str
    text_column: str
    speaker_column: str
    id_column: str


@dataclass
class AuditThresholds:
    min_duration_seconds: float = 1.0
    target_max_duration_seconds: float = 35.0
    review_leading_silence_seconds: float = 1.5
    review_trailing_silence_seconds: float = 2.0
    review_rms_dbfs: float = -45.0
    review_clipping_fraction: float = 0.001
    review_min_words_per_second: float = 0.4
    review_max_words_per_second: float = 3.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dataset_audit"),
        help="Dossier des rapports générés.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--waxal-revision", default=DEFAULT_WAXAL_REVISION)
    parser.add_argument("--klayt-revision", default=DEFAULT_KLAYT_REVISION)
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="Limite réservée au smoke test ; omettre pour l'audit complet.",
    )
    parser.add_argument("--min-duration-seconds", type=float, default=1.0)
    parser.add_argument("--target-max-duration-seconds", type=float, default=35.0)
    return parser.parse_args()


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _audio_payload(audio: Any) -> tuple[np.ndarray, int, bytes | None]:
    """Décode un objet datasets.Audio avec ou sans décodage automatique."""

    if isinstance(audio, dict) and audio.get("array") is not None:
        waveform = np.asarray(audio["array"], dtype=np.float32)
        return waveform, int(audio["sampling_rate"]), None

    if not isinstance(audio, dict):
        raise TypeError(f"Objet audio inattendu : {type(audio)!r}")

    raw_bytes = audio.get("bytes")
    path = audio.get("path")
    if raw_bytes is not None:
        waveform, sample_rate = sf.read(
            io.BytesIO(raw_bytes), dtype="float32", always_2d=True
        )
        return waveform, int(sample_rate), raw_bytes
    if path:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return waveform, int(sample_rate), None
    raise ValueError("L'objet audio ne contient ni bytes, ni path, ni array.")


def _mono(waveform: np.ndarray) -> tuple[np.ndarray, int]:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        return np.ascontiguousarray(waveform), 1
    if waveform.ndim != 2:
        raise ValueError(f"Forme audio non supportée : {waveform.shape}")
    # soundfile retourne [frames, channels]. Le cas inverse reste géré pour les
    # objets déjà décodés par datasets.
    if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
        channels = waveform.shape[0]
        mono = waveform.mean(axis=0)
    else:
        channels = waveform.shape[1]
        mono = waveform.mean(axis=1)
    return np.ascontiguousarray(mono, dtype=np.float32), int(channels)


def _silence_edges(
    waveform: np.ndarray, sample_rate: int, peak: float
) -> tuple[float, float, float]:
    """Mesure heuristique ; ce calcul n'est pas un détecteur de parole."""

    if waveform.size == 0 or peak <= 0.0:
        duration = waveform.size / max(sample_rate, 1)
        return duration, duration, 0.0
    threshold = max(1e-4, peak * 0.01)  # environ 40 dB sous le pic
    active = np.flatnonzero(np.abs(waveform) >= threshold)
    if active.size == 0:
        duration = waveform.size / sample_rate
        return duration, duration, threshold
    leading = float(active[0] / sample_rate)
    trailing = float((waveform.size - 1 - active[-1]) / sample_rate)
    return leading, trailing, threshold


def _waveform_sha256(
    waveform: np.ndarray, sample_rate: int, raw_bytes: bytes | None
) -> str:
    digest = hashlib.sha256()
    if raw_bytes is not None:
        digest.update(raw_bytes)
    else:
        digest.update(str(sample_rate).encode("ascii"))
        digest.update(np.asarray(waveform, dtype="<f4").tobytes())
    return digest.hexdigest()


def _safe_string(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    value = str(value).strip()
    return value or fallback


def audit_example(
    item: dict[str, Any],
    row_index: int,
    split: str,
    spec: DatasetSpec,
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    raw_text = _safe_string(item.get(spec.text_column), "")
    normalized_text = normalize_baoule_text(raw_text)
    example_id = _safe_string(item.get(spec.id_column), f"row-{row_index}")
    speaker_id = _safe_string(item.get(spec.speaker_column), "unknown")
    reasons: list[str] = []
    preprocessing_actions: list[str] = []

    row: dict[str, Any] = {
        "source": spec.source,
        "dataset_id": spec.dataset_id,
        "dataset_revision": spec.revision,
        "split": split,
        "row_index": row_index,
        "example_id": example_id,
        "speaker_id": f"{spec.source}:{speaker_id}",
        "text_raw": raw_text,
        "text_normalized": normalized_text,
        "text_words": len(normalized_text.split()),
        "text_characters": len(normalized_text),
        "words_per_second": "",
        "text_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
        "decode_ok": False,
        "sample_rate": "",
        "channels": "",
        "num_samples": "",
        "duration_seconds": "",
        "peak": "",
        "peak_dbfs": "",
        "rms": "",
        "rms_dbfs": "",
        "clipping_fraction": "",
        "leading_silence_seconds": "",
        "trailing_silence_seconds": "",
        "silence_amplitude_threshold": "",
        "audio_sha256": "",
        "status": "",
        "preprocessing_actions": "",
        "review_reasons": "",
        "audio_duplicate_cross_split": False,
        "text_duplicate_cross_split": False,
    }

    if not normalized_text:
        reasons.append("empty_normalized_text")

    try:
        waveform, sample_rate, raw_bytes = _audio_payload(item["audio"])
        waveform, channels = _mono(waveform)
        if sample_rate <= 0 or waveform.size == 0:
            raise ValueError("Signal audio vide ou fréquence invalide.")
        if not np.isfinite(waveform).all():
            raise ValueError("Le signal contient NaN ou Inf.")

        peak = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        duration = float(waveform.size / sample_rate)
        clipping_fraction = float(np.mean(np.abs(waveform) >= 0.999))
        leading, trailing, silence_threshold = _silence_edges(
            waveform, sample_rate, peak
        )

        row.update(
            {
                "decode_ok": True,
                "sample_rate": sample_rate,
                "channels": channels,
                "num_samples": int(waveform.size),
                "duration_seconds": duration,
                "words_per_second": len(normalized_text.split()) / duration,
                "peak": peak,
                "peak_dbfs": _dbfs(peak),
                "rms": rms,
                "rms_dbfs": _dbfs(rms),
                "clipping_fraction": clipping_fraction,
                "leading_silence_seconds": leading,
                "trailing_silence_seconds": trailing,
                "silence_amplitude_threshold": silence_threshold,
                "audio_sha256": _waveform_sha256(
                    waveform, sample_rate, raw_bytes
                ),
            }
        )

        if sample_rate != 16_000:
            preprocessing_actions.append("resample_to_16khz")
        if channels != 1:
            preprocessing_actions.append("downmix_to_mono")
        if duration < thresholds.min_duration_seconds:
            reasons.append("too_short")
        if duration > thresholds.target_max_duration_seconds:
            reasons.append("too_long_requires_alignment")
        if peak <= 1e-6:
            reasons.append("silent_audio")
        if _dbfs(rms) < thresholds.review_rms_dbfs:
            reasons.append("very_low_rms")
        if clipping_fraction > thresholds.review_clipping_fraction:
            reasons.append("clipping")
        if leading > thresholds.review_leading_silence_seconds:
            reasons.append("long_leading_silence")
        if trailing > thresholds.review_trailing_silence_seconds:
            reasons.append("long_trailing_silence")
        words_per_second = len(normalized_text.split()) / duration
        if words_per_second < thresholds.review_min_words_per_second:
            reasons.append("low_text_audio_ratio")
        if words_per_second > thresholds.review_max_words_per_second:
            reasons.append("high_text_audio_ratio")
    except Exception as error:  # l'erreur complète reste visible dans le CSV
        reasons.append(f"decode_error:{type(error).__name__}:{error}")

    hard_reject = (
        not row["decode_ok"]
        or "empty_normalized_text" in reasons
        or "silent_audio" in reasons
        or "too_short" in reasons
    )
    row["status"] = "reject" if hard_reject else ("review" if reasons else "accept")
    row["preprocessing_actions"] = "|".join(preprocessing_actions)
    row["review_reasons"] = "|".join(reasons)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _duplicate_groups(
    rows: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, ""))
        if key == "text_sha256" and not str(row.get("text_normalized", "")):
            continue
        if value:
            groups[value].append(row)

    result: list[dict[str, Any]] = []
    for value, members in groups.items():
        split_keys = {m["split"] for m in members}
        if len(members) < 2:
            continue
        result.append(
            {
                "key": key,
                "sha256": value,
                "count": len(members),
                "cross_split": len(split_keys) > 1,
                "members": [
                    {
                        "source": m["source"],
                        "split": m["split"],
                        "row_index": m["row_index"],
                        "example_id": m["example_id"],
                        "speaker_id": m["speaker_id"],
                    }
                    for m in members
                ],
            }
        )
    return result


def _mark_cross_split_duplicates(
    rows: list[dict[str, Any]], groups: list[dict[str, Any]], flag: str
) -> None:
    indexes = {
        (m["source"], m["split"], m["row_index"])
        for group in groups
        if group["cross_split"]
        for m in group["members"]
    }
    for row in rows:
        key = (row["source"], row["split"], row["row_index"])
        row[flag] = key in indexes


def _summarize(
    rows: list[dict[str, Any]], fingerprints: dict[str, str], args: argparse.Namespace
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "arguments": {
            "waxal_revision": args.waxal_revision,
            "klayt_revision": args.klayt_revision,
            "max_examples_per_split": args.max_examples_per_split,
            "min_duration_seconds": args.min_duration_seconds,
            "target_max_duration_seconds": args.target_max_duration_seconds,
        },
        "fingerprints": fingerprints,
        "groups": {},
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["split"])].append(row)

    for (source, split), members in sorted(grouped.items()):
        durations = [
            float(row["duration_seconds"])
            for row in members
            if row["decode_ok"] and row["duration_seconds"] != ""
        ]
        speakers = {row["speaker_id"] for row in members}
        status_counts = Counter(row["status"] for row in members)
        reason_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        for row in members:
            reason_counts.update(filter(None, str(row["review_reasons"]).split("|")))
            action_counts.update(
                filter(None, str(row["preprocessing_actions"]).split("|"))
            )
        summary["groups"][f"{source}:{split}"] = {
            "examples": len(members),
            "decoded": sum(bool(row["decode_ok"]) for row in members),
            "speakers": len(speakers),
            "hours": sum(durations) / 3600.0,
            "duration_min": min(durations) if durations else None,
            "duration_max": max(durations) if durations else None,
            "status_counts": dict(sorted(status_counts.items())),
            "preprocessing_action_counts": dict(sorted(action_counts.items())),
            "review_reason_counts": dict(sorted(reason_counts.items())),
        }
    return summary


def main() -> int:
    args = parse_args()
    thresholds = AuditThresholds(
        min_duration_seconds=args.min_duration_seconds,
        target_max_duration_seconds=args.target_max_duration_seconds,
    )
    specs = (
        DatasetSpec(
            source="waxal",
            dataset_id="google/WaxalNLP",
            config="bau_tts",
            revision=args.waxal_revision,
            text_column="text",
            speaker_column="speaker_id",
            id_column="id",
        ),
        DatasetSpec(
            source="klayt",
            dataset_id="Klayt/baoule-common-voice",
            config=None,
            revision=args.klayt_revision,
            text_column="sentence",
            speaker_column="client_id",
            id_column="path",
        ),
    )

    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise SystemExit(
            "La dépendance 'datasets' manque. Exécute : "
            "python -m pip install -r requirements-audit.txt"
        ) from error

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    inventories_raw: dict[str, list[str]] = defaultdict(list)
    inventories_normalized: dict[str, list[str]] = defaultdict(list)

    for spec in specs:
        print(f"Chargement {spec.dataset_id} @ {spec.revision}", flush=True)
        dataset = load_dataset(
            spec.dataset_id,
            spec.config,
            revision=spec.revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        for split in SPLITS:
            if split not in dataset:
                raise RuntimeError(f"Split manquant : {spec.source}:{split}")
            split_dataset = dataset[split].cast_column("audio", Audio(decode=False))
            fingerprints[f"{spec.source}:{split}"] = split_dataset._fingerprint
            limit = len(split_dataset)
            if args.max_examples_per_split is not None:
                limit = min(limit, args.max_examples_per_split)
            print(f"Audit {spec.source}:{split} — {limit} exemples", flush=True)
            for row_index in range(limit):
                row = audit_example(
                    split_dataset[row_index], row_index, split, spec, thresholds
                )
                rows.append(row)
                inventories_raw[spec.source].append(row["text_raw"])
                inventories_normalized[spec.source].append(row["text_normalized"])
                if (row_index + 1) % 100 == 0:
                    print(f"  {row_index + 1}/{limit}", flush=True)

    audio_duplicates = _duplicate_groups(rows, "audio_sha256")
    text_duplicates = _duplicate_groups(rows, "text_sha256")
    _mark_cross_split_duplicates(
        rows, audio_duplicates, "audio_duplicate_cross_split"
    )
    _mark_cross_split_duplicates(rows, text_duplicates, "text_duplicate_cross_split")

    for row in rows:
        duplicate_reasons = []
        if row["audio_duplicate_cross_split"]:
            duplicate_reasons.append("audio_duplicate_cross_split")
        if row["text_duplicate_cross_split"]:
            duplicate_reasons.append("text_duplicate_cross_split")
        if duplicate_reasons:
            existing = str(row["review_reasons"])
            row["review_reasons"] = "|".join(filter(None, [existing, *duplicate_reasons]))
            if row["status"] == "accept":
                row["status"] = "review"

    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(output_dir / "dataset_audit.csv", rows, fieldnames)
    review_rows = [row for row in rows if row["status"] != "accept"]
    _write_csv(output_dir / "manual_review.csv", review_rows, fieldnames)

    duplicates = {
        "audio": audio_duplicates,
        "normalized_text": text_duplicates,
    }
    (output_dir / "duplicate_groups.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    inventory_report: dict[str, Any] = {}
    for source in sorted(inventories_raw):
        raw_counts = character_inventory(inventories_raw[source])
        normalized_counts = character_inventory(inventories_normalized[source])
        inventory_report[source] = {
            "raw": [
                {**unicode_description(char), "count": count}
                for char, count in raw_counts.items()
            ],
            "normalized": [
                {**unicode_description(char), "count": count}
                for char, count in normalized_counts.items()
            ],
        }
    (output_dir / "character_inventory.json").write_text(
        json.dumps(inventory_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = _summarize(rows, fingerprints, args)
    summary["thresholds"] = asdict(thresholds)
    summary["totals"] = {
        "examples": len(rows),
        "accept": sum(row["status"] == "accept" for row in rows),
        "review": sum(row["status"] == "review" for row in rows),
        "reject": sum(row["status"] == "reject" for row in rows),
        "manual_review_rows": len(review_rows),
        "audio_duplicate_groups": len(audio_duplicates),
        "text_duplicate_groups": len(text_duplicates),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print(f"Rapports écrits dans {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
