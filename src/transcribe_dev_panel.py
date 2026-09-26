#!/usr/bin/env python3
"""Transcrit un panneau dev fixe avec un modèle de base ou un checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-card")
    group.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--examples-per-corpus", type=int, default=2)
    return parser.parse_args()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def encoded_bytes(value: Any) -> bytes:
    import numpy as np

    return np.asarray(value, dtype=np.int8).tobytes()


def load_dev_rows(dataset_root: Path) -> list[dict[str, Any]]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(dataset_root, format="parquet", partitioning="hive")
    table = dataset.to_table(
        columns=["text", "audio_bytes", "audio_size", "corpus", "split"],
        filter=(ds.field("split") == "dev") & (ds.field("audio_size") <= 320_000),
    )
    rows = table.to_pylist()
    for row in rows:
        raw = encoded_bytes(row["audio_bytes"])
        row["audio_sha256"] = hashlib.sha256(raw).hexdigest()
        row["audio_encoded"] = raw
    return rows


def select_or_load_panel(
    rows: list[dict[str, Any]], manifest_path: Path, examples_per_corpus: int
) -> list[dict[str, Any]]:
    by_hash = {row["audio_sha256"]: row for row in rows}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = [item["audio_sha256"] for item in manifest["examples"]]
        missing = [value for value in hashes if value not in by_hash]
        if missing:
            raise RuntimeError(f"Audios du panneau absents du dataset : {missing}")
        return [by_hash[value] for value in hashes]

    selected: list[dict[str, Any]] = []
    for corpus in sorted({str(row["corpus"]) for row in rows}):
        candidates = [
            row
            for row in rows
            if row["corpus"] == corpus
            and 4 * 16_000 <= int(row["audio_size"]) <= 14 * 16_000
        ]
        candidates.sort(key=lambda row: row["audio_sha256"])
        if len(candidates) < examples_per_corpus:
            raise RuntimeError(
                f"Pas assez d'exemples dev représentatifs pour {corpus}."
            )
        selected.extend(candidates[:examples_per_corpus])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "selection": "sha256 croissant, 4 à 14 secondes, split dev",
                "examples": [
                    {
                        "panel_id": f"dev_{index:02d}",
                        "corpus": row["corpus"],
                        "duration_seconds": int(row["audio_size"]) / 16_000,
                        "audio_sha256": row["audio_sha256"],
                        "reference": row["text"],
                    }
                    for index, row in enumerate(selected, start=1)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def load_pipeline(args: argparse.Namespace):
    import torch
    import omnilingual_asr  # noqa: F401 — enregistre l'extension fairseq2
    from fairseq2 import init_fairseq2
    from fairseq2.composition.assets import register_checkpoint_models

    def extras(container) -> None:
        if args.checkpoint_dir is not None:
            register_checkpoint_models(container, args.checkpoint_dir.resolve())

    init_fairseq2(extras=extras)

    from fairseq2.data.tokenizers import load_tokenizer
    from fairseq2.models.hub import load_model
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    if args.checkpoint_dir is not None:
        if args.checkpoint_step is None:
            raise ValueError("--checkpoint-step est requis avec --checkpoint-dir.")
        card = f"checkpoint_step_{args.checkpoint_step}"
        label = card
    else:
        card = args.model_card
        label = str(card)
    model = load_model(card, device=torch.device("cuda:0"), dtype=torch.float16)
    tokenizer = load_tokenizer("omniASR_tokenizer_v1")
    pipeline = ASRInferencePipeline(
        model_card=None,
        model=model,
        tokenizer=tokenizer,
        device="cuda:0",
        dtype=torch.float16,
    )
    return pipeline, label


def main() -> int:
    args = parse_args()
    if args.checkpoint_dir is None and args.checkpoint_step is not None:
        raise ValueError("--checkpoint-step ne s'utilise qu'avec --checkpoint-dir.")
    rows = load_dev_rows(args.dataset_root.resolve())
    panel = select_or_load_panel(
        rows, args.panel_manifest.resolve(), args.examples_per_corpus
    )
    pipeline, model_label = load_pipeline(args)
    predictions = pipeline.transcribe(
        [row["audio_encoded"] for row in panel], batch_size=1
    )

    try:
        from .normalize_text import normalize_baoule_text
    except ImportError:
        from normalize_text import normalize_baoule_text  # type: ignore

    results = []
    total_word_errors = 0
    total_words = 0
    total_char_errors = 0
    total_chars = 0
    for index, (row, prediction) in enumerate(zip(panel, predictions), start=1):
        reference = normalize_baoule_text(str(row["text"]))
        hypothesis = normalize_baoule_text(str(prediction))
        word_errors = edit_distance(reference.split(), hypothesis.split())
        char_errors = edit_distance(list(reference.replace(" ", "")), list(hypothesis.replace(" ", "")))
        words = len(reference.split())
        chars = len(reference.replace(" ", ""))
        total_word_errors += word_errors
        total_words += words
        total_char_errors += char_errors
        total_chars += chars
        results.append(
            {
                "panel_id": f"dev_{index:02d}",
                "corpus": row["corpus"],
                "duration_seconds": int(row["audio_size"]) / 16_000,
                "audio_sha256": row["audio_sha256"],
                "reference": reference,
                "prediction": hypothesis,
                "wer": word_errors / max(1, words),
                "cer": char_errors / max(1, chars),
            }
        )
    output = {
        "model": model_label,
        "num_examples": len(results),
        "micro_wer": total_word_errors / max(1, total_words),
        "micro_cer": total_char_errors / max(1, total_chars),
        "examples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
