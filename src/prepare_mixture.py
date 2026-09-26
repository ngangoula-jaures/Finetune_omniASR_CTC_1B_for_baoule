#!/usr/bin/env python3
"""Prépare le MixtureParquet Baoulé destiné au fine-tuning OmniASR.

La sélection hackathon respecte les décisions manuelles, conserve les exemples
déjà validés et ajoute les exemples Waxal non rejetés de 40 secondes maximum.
Les audios longs restent différés jusqu'à leur segmentation alignée.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LANGUAGE = "bci_Latn"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_WAXAL_REVISION = "5f4d8ca24f2b9d168b2ee545f1febaaff4b40580"
DEFAULT_KLAYT_REVISION = "96eb4413b7cc3190c39a35ee03eccb321c2b9929"
OUTPUT_SPLIT = {"train": "train", "validation": "dev", "test": "test"}
REQUIRED_MANIFEST_COLUMNS = {
    "review_id",
    "source",
    "split",
    "row_index",
    "duration_seconds",
    "text_normalized",
    "audit_status",
    "suggested_decision",
    "final_decision",
    "trim_start_seconds",
    "trim_end_seconds",
}


@dataclass(frozen=True)
class Selection:
    include: bool
    status: str
    reason: str
    trim_start_seconds: float | None = None
    trim_end_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--waxal-max-seconds", type=float, default=40.0)
    parser.add_argument("--rows-per-file", type=int, default=100)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Produit le manifeste et le résumé sans télécharger les datasets.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colonnes absentes du manifeste : {sorted(missing)}")
        rows = list(reader)
    ids = [row["review_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Le manifeste contient des review_id dupliqués.")
    return rows


def selection_for(row: dict[str, str], waxal_max_seconds: float) -> Selection:
    """Applique la politique hackathon avec priorité aux décisions humaines."""

    decision = row["final_decision"].strip()
    duration = float(row["duration_seconds"])

    if decision == "exclude":
        return Selection(False, "excluded", "manual_exclude")
    if decision == "segment":
        return Selection(False, "deferred", "manual_segment_later")
    if decision == "trim":
        try:
            start = float(row["trim_start_seconds"])
            end = float(row["trim_end_seconds"])
        except ValueError:
            return Selection(False, "excluded", "invalid_trim_bounds")
        if not 0.0 <= start < end <= duration + 1e-3:
            return Selection(False, "excluded", "invalid_trim_bounds")
        if end - start > waxal_max_seconds:
            return Selection(False, "deferred", "trimmed_audio_still_too_long")
        return Selection(True, "selected", "validated_manual_trim", start, end)
    if decision == "keep":
        if duration > waxal_max_seconds:
            return Selection(False, "deferred", "validated_but_too_long")
        reason = (
            "validated_automatic"
            if row["suggested_decision"] == "keep"
            else "validated_manual"
        )
        return Selection(True, "selected", reason)
    if decision:
        return Selection(False, "excluded", f"unknown_decision:{decision}")

    if (
        row["source"] == "waxal"
        and duration <= waxal_max_seconds
        and row["audit_status"] != "reject"
        and row["text_normalized"].strip()
    ):
        return Selection(True, "selected", "waxal_short_hackathon_policy")
    if row["source"] == "waxal" and duration > waxal_max_seconds:
        return Selection(False, "deferred", "waxal_requires_segmentation")
    return Selection(False, "deferred", "manual_review_pending")


def build_selection_manifest(
    rows: list[dict[str, str]], waxal_max_seconds: float
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        selection = selection_for(row, waxal_max_seconds)
        result.append(
            {
                **row,
                "preparation_status": selection.status,
                "preparation_reason": selection.reason,
                "effective_trim_start_seconds": (
                    ""
                    if selection.trim_start_seconds is None
                    else f"{selection.trim_start_seconds:.3f}"
                ),
                "effective_trim_end_seconds": (
                    ""
                    if selection.trim_end_seconds is None
                    else f"{selection.trim_end_seconds:.3f}"
                ),
                "output_split": OUTPUT_SPLIT[row["split"]],
                "language": LANGUAGE,
                "corpus": row["source"],
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Aucune ligne à écrire.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["preparation_status"] == "selected"]
    groups: dict[str, dict[str, Any]] = {}
    for (source, split), members in _group_by(selected, "source", "output_split"):
        groups[f"{source}:{split}"] = {
            "examples": len(members),
            "hours_before_materialization": sum(
                float(row["duration_seconds"]) for row in members
            )
            / 3600.0,
            "selection_reasons": dict(
                sorted(Counter(row["preparation_reason"] for row in members).items())
            ),
        }
    return {
        "policy": {
            "manual_decisions_take_precedence": True,
            "unreviewed_klayt_is_deferred": True,
            "long_audio_segmentation_is_deferred": True,
        },
        "totals": {
            "manifest_examples": len(rows),
            "selected_examples": len(selected),
            "selected_hours_before_materialization": sum(
                float(row["duration_seconds"]) for row in selected
            )
            / 3600.0,
            "status_counts": dict(
                sorted(Counter(row["preparation_status"] for row in rows).items())
            ),
            "reason_counts": dict(
                sorted(Counter(row["preparation_reason"] for row in rows).items())
            ),
        },
        "selected_groups": groups,
    }


def _group_by(
    rows: Iterable[dict[str, Any]], *keys: str
) -> Iterable[tuple[tuple[str, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    return sorted(groups.items())


def _resample(waveform: Any, source_rate: int) -> Any:
    import numpy as np
    from scipy.signal import resample_poly

    if source_rate == TARGET_SAMPLE_RATE:
        return np.asarray(waveform, dtype=np.float32)
    divisor = math.gcd(source_rate, TARGET_SAMPLE_RATE)
    result = resample_poly(
        waveform,
        TARGET_SAMPLE_RATE // divisor,
        source_rate // divisor,
    )
    return np.asarray(result, dtype=np.float32)


def _encode_flac(waveform: Any) -> list[int]:
    import numpy as np
    import soundfile as sf

    waveform = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, TARGET_SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return np.frombuffer(buffer.getvalue(), dtype=np.int8).tolist()


def _materialize_example(
    row: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    try:
        from .audit_datasets import _audio_payload, _mono
    except ImportError:
        from audit_datasets import _audio_payload, _mono  # type: ignore

    waveform, sample_rate, _ = _audio_payload(item["audio"])
    waveform, _ = _mono(waveform)
    waveform = _resample(waveform, sample_rate)

    if row["effective_trim_start_seconds"]:
        start = round(float(row["effective_trim_start_seconds"]) * TARGET_SAMPLE_RATE)
        end = round(float(row["effective_trim_end_seconds"]) * TARGET_SAMPLE_RATE)
        waveform = waveform[start:end]
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.size < TARGET_SAMPLE_RATE:
        raise ValueError(f"Audio final inférieur à une seconde : {row['review_id']}")
    if waveform.size > round(40.0 * TARGET_SAMPLE_RATE) + 1:
        raise ValueError(f"Audio final supérieur à 40 secondes : {row['review_id']}")

    return {
        "text": row["text_normalized"],
        "audio_bytes": _encode_flac(waveform),
        "audio_size": int(waveform.size),
    }


def _parquet_schema() -> Any:
    import pyarrow as pa

    # Les colonnes corpus/split/language sont portées par les partitions Hive,
    # comme avec ray.data.write_parquet(partition_cols=...) dans Meta.
    return pa.schema(
        [
            pa.field("text", pa.string(), nullable=False),
            pa.field("audio_bytes", pa.list_(pa.int8()), nullable=False),
            pa.field("audio_size", pa.int64(), nullable=False),
        ]
    )


def _write_part(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_parquet_schema())
    pq.write_table(table, path, compression="zstd", row_group_size=100)


def materialize(
    selected_rows: list[dict[str, Any]],
    dataset_root: Path,
    cache_dir: Path | None,
    rows_per_file: int,
) -> dict[str, Any]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise SystemExit(
            "Dépendances manquantes. Installe requirements-data.txt."
        ) from error

    if dataset_root.exists():
        raise FileExistsError(
            f"La destination existe déjà : {dataset_root}. Choisis un dossier vide."
        )
    datasets_by_source = {
        "waxal": load_dataset(
            "google/WaxalNLP",
            "bau_tts",
            revision=DEFAULT_WAXAL_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
        ),
        "klayt": load_dataset(
            "Klayt/baoule-common-voice",
            revision=DEFAULT_KLAYT_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
        ),
    }
    for source, split_map in datasets_by_source.items():
        datasets_by_source[source] = {
            name: split.cast_column("audio", Audio(decode=False))
            for name, split in split_map.items()
        }

    materialized_counts: Counter[tuple[str, str]] = Counter()
    materialized_samples: Counter[tuple[str, str]] = Counter()
    for (source, output_split), members in _group_by(
        selected_rows, "source", "output_split"
    ):
        partition = (
            dataset_root
            / f"corpus={source}"
            / f"split={output_split}"
            / f"language={LANGUAGE}"
        )
        buffer: list[dict[str, Any]] = []
        part_number = 0
        for position, row in enumerate(members, start=1):
            item = datasets_by_source[source][row["split"]][int(row["row_index"])]
            example = _materialize_example(row, item)
            buffer.append(example)
            materialized_counts[(source, output_split)] += 1
            materialized_samples[(source, output_split)] += example["audio_size"]
            if len(buffer) >= rows_per_file:
                _write_part(partition / f"part-{part_number:05d}.parquet", buffer)
                buffer = []
                part_number += 1
            if position % 100 == 0:
                print(f"{source}:{output_split} — {position}/{len(members)}", flush=True)
        if buffer:
            _write_part(partition / f"part-{part_number:05d}.parquet", buffer)

    return {
        f"{source}:{split}": {
            "examples": materialized_counts[(source, split)],
            "audio_samples": materialized_samples[(source, split)],
            "hours": materialized_samples[(source, split)]
            / TARGET_SAMPLE_RATE
            / 3600.0,
        }
        for source, split in sorted(materialized_counts)
    }


def write_distribution(path: Path, groups: dict[str, Any]) -> None:
    corpus_hours: dict[str, float] = defaultdict(float)
    for key, values in groups.items():
        corpus, split = key.split(":", 1)
        if split == "train":
            corpus_hours[corpus] += float(values["hours"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["corpus", "language", "hours"], delimiter="\t"
        )
        writer.writeheader()
        for corpus, hours in sorted(corpus_hours.items()):
            writer.writerow({"corpus": corpus, "language": LANGUAGE, "hours": hours})


def write_asset_card(path: Path, dataset_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "name: baoule_mixed",
                "dataset_family: mixture_parquet_asr_dataset",
                "dataset_config:",
                f"  data: {dataset_root}",
                "tokenizer_ref: omniASR_tokenizer_v1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_dataset_card(path: Path, summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    groups = summary["materialized_groups"]
    table_rows = []
    for key, values in sorted(groups.items()):
        source, split = key.split(":", 1)
        table_rows.append(
            f"| {source} | {split} | {values['examples']} | {values['hours']:.3f} |"
        )
    path.write_text(
        "\n".join(
            [
                "---",
                "pretty_name: Baoulé ASR Hackathon Mixture",
                "language:",
                "- bci",
                "task_categories:",
                "- automatic-speech-recognition",
                "tags:",
                "- audio",
                "- baoule",
                "- omnilingual-asr",
                "- mixture-parquet",
                "---",
                "",
                "# Baoulé ASR Hackathon Mixture",
                "",
                "Dataset MixtureParquet préparé par Tree AI Lab pour le fine-tuning de ",
                "`facebook/omniASR-CTC-1B` en baoulé (`bci_Latn`).",
                "",
                "## Contenu",
                "",
                f"- Exemples sélectionnés : {totals['selected_examples']}",
                f"- Durée avant réencodage : {totals['selected_hours_before_materialization']:.3f} h",
                "- Audio : FLAC mono 16 kHz, 40 secondes maximum",
                "- Texte : transcription baoulé normalisée",
                "",
                "| Source | Split | Exemples | Heures |",
                "|---|---:|---:|---:|",
                *table_rows,
                "",
                "## Sources et versions",
                "",
                f"- `google/WaxalNLP`, configuration `bau_tts`, révision `{DEFAULT_WAXAL_REVISION}`",
                f"- `Klayt/baoule-common-voice`, révision `{DEFAULT_KLAYT_REVISION}`",
                "",
                "WaxalNLP affiche des licences CC BY 4.0 et CC BY-SA 4.0 selon ses données. ",
                "Klayt/baoule-common-voice est publié sous CC0 1.0. Les utilisateurs doivent ",
                "respecter les licences et obligations d'attribution des sources d'origine.",
                "",
                "## Sélection hackathon",
                "",
                "Le corpus contient les exemples validés manuellement, les validations automatiques ",
                "et les Waxal non rejetés de 40 secondes maximum. Les exclusions manuelles sont ",
                "respectées. Les Klayt non révisés et les Waxal longs restent différés.",
                "",
                "Le fichier `hackathon_selection_manifest.csv` conserve la décision et la raison ",
                "pour chaque exemple audité.",
                "",
                "## Format OmniASR",
                "",
                "```text",
                "baoule_mixed/version=0/corpus=<waxal|klayt>/split=<train|dev|test>/language=bci_Latn/part-*.parquet",
                "```",
                "",
                "Chaque ligne contient `text`, `audio_bytes` et `audio_size`. Les colonnes ",
                "`corpus`, `split` et `language` proviennent des partitions Hive.",
                "",
                "## Limites",
                "",
                "Cette version est un sous-ensemble destiné au hackathon. La revue manuelle et la ",
                "segmentation des enregistrements longs continueront après l'événement.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.waxal_max_seconds <= 0 or args.waxal_max_seconds > 40:
        raise ValueError("waxal-max-seconds doit être dans ]0, 40].")
    if args.rows_per_file <= 0:
        raise ValueError("rows-per-file doit être positif.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_rows = read_manifest(args.review_manifest)
    rows = build_selection_manifest(input_rows, args.waxal_max_seconds)
    write_csv(output_dir / "hackathon_selection_manifest.csv", rows)

    summary = selection_summary(rows)
    summary["inputs"] = {
        "review_manifest": str(args.review_manifest.resolve()),
        "review_manifest_sha256": hashlib.sha256(
            args.review_manifest.read_bytes()
        ).hexdigest(),
        "waxal_revision": DEFAULT_WAXAL_REVISION,
        "klayt_revision": DEFAULT_KLAYT_REVISION,
        "language": LANGUAGE,
        "waxal_max_seconds": args.waxal_max_seconds,
    }

    if not args.selection_only:
        selected = [row for row in rows if row["preparation_status"] == "selected"]
        dataset_root = output_dir / "baoule_mixed" / "version=0"
        groups = materialize(
            selected, dataset_root, args.cache_dir, args.rows_per_file
        )
        summary["materialized_groups"] = groups
        write_distribution(output_dir / "language_distribution_0.tsv", groups)
        write_asset_card(
            output_dir / "assets" / "baoule_mixed.yaml", dataset_root
        )
        write_dataset_card(output_dir / "README.md", summary)

    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print(f"Sorties : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
