"""Telugu word-to-phone frontend used by the custom inference wrapper."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import regex as unicode_regex


def tokenize_telugu(text: str) -> list[str]:
    text = " ".join(unicodedata.normalize("NFC", str(text)).split()).strip()
    raw = unicode_regex.findall(r"[\p{L}\p{M}\p{N}]+(?:[’'][\p{L}\p{M}\p{N}]+)*", text)
    return [w.replace("’", "'").lower() if w.isascii() else w for w in raw]


def load_lexicon(path: str | Path) -> dict[str, list[str]]:
    lexicon: dict[str, list[str]] = {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Malformed lexicon line {line_no}: expected WORD<TAB>PHONES")
            word, phones = parts
            tokens = phones.split()
            if not word or not tokens:
                raise ValueError(f"Empty word/phone sequence on lexicon line {line_no}")
            lexicon[word] = tokens
    return lexicon


def words_to_phones(text: str, lexicon: dict[str, list[str]]) -> list[str]:
    words = tokenize_telugu(text)
    if not words:
        raise ValueError("Input text contains no Telugu words/numeric tokens")
    missing = [w for w in words if w not in lexicon]
    if missing:
        raise KeyError(
            "Words missing from Telugu lexicon: " + ", ".join(sorted(set(missing))) +
            ". Extend the lexicon with the assignment test sentences, review them, "
            "then rerun patch_fastspeech2.py."
        )
    return [phone for word in words for phone in lexicon[word]]
