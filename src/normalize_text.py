"""Normalisation textuelle conservatrice pour les transcriptions baoulé.

La normalisation destinée à l'entraînement conserve les lettres et les marques
diacritiques. Elle harmonise uniquement les variantes typographiques connues et
retire la ponctuation qui n'est pas prononcée par le modèle CTC.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable


APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "’": "'",  # RIGHT SINGLE QUOTATION MARK
        "‘": "'",  # LEFT SINGLE QUOTATION MARK
        "ʼ": "'",  # MODIFIER LETTER APOSTROPHE, présent dans Common Voice
        "＇": "'",  # FULLWIDTH APOSTROPHE
        "`": "'",
        "´": "'",
    }
)

BRACKETED_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
STANDALONE_NUMBER_RE = re.compile(r"(?<!\S)\d+(?!\S)")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_baoule_text(text: object) -> str:
    """Retourne une transcription canonique sans supprimer les tons.

    Les chiffres inclus dans un mot sont conservés. Les tokens constitués
    uniquement de chiffres sont retirés, conformément à la recette précédente.
    """

    normalized = unicodedata.normalize("NFC", str(text or ""))
    normalized = BRACKETED_ANNOTATION_RE.sub(" ", normalized)
    normalized = normalized.translate(APOSTROPHE_TRANSLATION).lower()

    kept: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if char == "'" or char.isspace() or category[0] in {"L", "M", "N"}:
            kept.append(char)
        else:
            kept.append(" ")

    normalized = WHITESPACE_RE.sub(" ", "".join(kept)).strip()
    normalized = STANDALONE_NUMBER_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def character_inventory(texts: Iterable[str]) -> dict[str, int]:
    """Compte les caractères non blancs d'une collection de textes."""

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(char for char in text if not char.isspace())
    return dict(sorted(counts.items(), key=lambda item: ord(item[0])))


def unicode_description(char: str) -> dict[str, str | int]:
    """Décrit un caractère pour rendre l'audit Unicode lisible."""

    return {
        "character": char,
        "codepoint": f"U+{ord(char):04X}",
        "name": unicodedata.name(char, "UNKNOWN"),
        "category": unicodedata.category(char),
    }

