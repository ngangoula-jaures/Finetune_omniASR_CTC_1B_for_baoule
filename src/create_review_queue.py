#!/usr/bin/env python3
"""Construit une file de revue et un manifeste de décisions depuis l'audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_FINAL_DECISIONS = {"", "keep", "trim", "segment", "exclude"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--duplicates-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def duplicate_members(path: Path) -> set[tuple[str, str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (member["source"], member["split"], int(member["row_index"]))
        for group in payload.get("normalized_text", [])
        for member in group["members"]
    }


def suggestion_for(
    row: dict[str, str], is_any_text_duplicate: bool
) -> tuple[int, str, str]:
    reasons = set(filter(None, row["review_reasons"].split("|")))
    source = row["source"]
    split = row["split"]

    if row["decode_ok"] != "True" or row["status"] == "reject":
        return 0, "exclude", "rejet technique de l'audit"
    if "text_duplicate_cross_split" in reasons and split == "train":
        return 0, "exclude", "fuite textuelle train-test"
    if {"high_text_audio_ratio", "low_text_audio_ratio"} & reasons:
        return 0, "review", "alignement texte-audio suspect"
    if source == "klayt" and reasons:
        return 1, "review", "Klayt signalé par l'audit"
    if source == "klayt":
        return 2, "review", "écoute Klayt pour voix parasite non transcrite"
    if "too_long_requires_alignment" in reasons:
        return 3, "segment", "audio long à aligner avant découpage"
    if is_any_text_duplicate:
        return 3, "review", "transcription répétée à comparer"
    if reasons <= {"long_leading_silence", "long_trailing_silence"} and reasons:
        return 4, "trim", "silence de bord à vérifier puis rogner"
    if not reasons:
        return 9, "keep", "aucun signalement automatique"
    return 4, "review", "signalement restant à vérifier"


def build_manifests(
    audit_rows: list[dict[str, str]], duplicate_keys: set[tuple[str, str, int]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []

    for row in audit_rows:
        key = (row["source"], row["split"], int(row["row_index"]))
        priority, suggestion, rationale = suggestion_for(
            row, key in duplicate_keys
        )
        item: dict[str, Any] = {
            "review_id": f"{row['source']}:{row['split']}:{row['row_index']}",
            "priority": priority,
            "source": row["source"],
            "split": row["split"],
            "row_index": int(row["row_index"]),
            "example_id": row["example_id"],
            "speaker_id": row["speaker_id"],
            "duration_seconds": row["duration_seconds"],
            "words_per_second": row["words_per_second"],
            "rms_dbfs": row.get("rms_dbfs", ""),
            "leading_silence_seconds": row.get("leading_silence_seconds", ""),
            "trailing_silence_seconds": row.get("trailing_silence_seconds", ""),
            "text_raw": row.get("text_raw", ""),
            "text_normalized": row["text_normalized"],
            "audit_status": row["status"],
            "preprocessing_actions": row["preprocessing_actions"],
            "review_reasons": row["review_reasons"],
            "suggested_decision": suggestion,
            "suggestion_rationale": rationale,
            "final_decision": "keep" if suggestion == "keep" else "",
            "trim_start_seconds": "",
            "trim_end_seconds": "",
            "segment_boundaries_seconds": "",
            "review_notes": "",
        }
        manifest.append(item)
        if suggestion != "keep":
            queue.append(dict(item))

    queue.sort(
        key=lambda row: (
            int(row["priority"]),
            row["source"],
            row["split"],
            int(row["row_index"]),
        )
    )
    return manifest, queue


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Aucune ligne à écrire dans {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = read_csv(args.audit_csv)
    duplicates = duplicate_members(args.duplicates_json)
    manifest, queue = build_manifests(rows, duplicates)

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "decision_manifest.csv", manifest)
    write_csv(output_dir / "review_queue.csv", queue)

    summary = {
        "manifest_rows": len(manifest),
        "review_queue_rows": len(queue),
        "suggested_decisions": dict(
            sorted(Counter(row["suggested_decision"] for row in manifest).items())
        ),
        "queue_priorities": dict(
            sorted(Counter(str(row["priority"]) for row in queue).items())
        ),
    }
    (output_dir / "review_queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
