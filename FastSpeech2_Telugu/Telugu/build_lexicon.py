#!/usr/bin/env python3
"""Generate an eSpeak-NG Telugu IPA candidate dictionary and review artifacts.

The output is a candidate, not a linguistically validated pronunciation lexicon.
Review common, long, code-switched, numeric and suspicious pronunciations with a
Telugu reader before treating the dictionary as final.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import regex as unicode_regex
from phonemizer import phonemize
from phonemizer.separator import Separator


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).strip()


def tokenize_words(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = unicode_regex.findall(r"[\p{L}\p{M}\p{N}]+(?:[’'][\p{L}\p{M}\p{N}]+)*", text)
    return [token.replace("’", "'").lower() if token.isascii() else token for token in tokens]


def read_text_column(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    column = next((c for c in ("normalized_text", "text", "transcript") if c in frame.columns), None)
    if column is None:
        raise ValueError(f"No text column in {path}; columns={list(frame.columns)}")
    return [normalize_text(x) for x in frame[column].dropna().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/processed/telugu/manifests/telugu_train_manifest.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/processed/telugu/manifests/telugu_validation_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/telugu/lexicon"))
    parser.add_argument("--language", default="te")
    parser.add_argument("--include-test-sentences", type=Path, default=None, help="Optional JSON with test sentences; adds words to the dictionary only, not to training data.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_counts: Counter[str] = Counter()
    val_counts: Counter[str] = Counter()
    for text in read_text_column(args.train_manifest):
        train_counts.update(tokenize_words(text))
    for text in read_text_column(args.validation_manifest):
        val_counts.update(tokenize_words(text))

    extra_counts: Counter[str] = Counter()
    if args.include_test_sentences:
        test_data = json.loads(args.include_test_sentences.read_text(encoding="utf-8"))
        for item in test_data:
            extra_counts.update(tokenize_words(item["text"]))

    words = sorted(set(train_counts) | set(val_counts) | set(extra_counts))
    if not words:
        raise RuntimeError("No words were found in the manifests.")

    separator = "~"
    phoneme_strings = phonemize(
        words,
        language=args.language,
        backend="espeak",
        separator=Separator(phone=separator, word=" "),
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        language_switch="keep-flags",
        words_mismatch="warn",
        njobs=1,
    )
    if len(phoneme_strings) != len(words):
        raise RuntimeError(f"eSpeak returned {len(phoneme_strings)} pronunciations for {len(words)} words.")

    language_flag = re.compile(r"\([a-z]{2,3}(?:-[a-z0-9]+)?\)", re.IGNORECASE)
    rows, errors, phone_inventory = [], [], set()
    for word, raw in zip(words, phoneme_strings):
        raw = str(raw).strip()
        if not raw or language_flag.search(raw):
            errors.append({"word": word, "raw_espeak_output": raw, "reason": "empty output or language-switch flag"})
            continue
        phones = [p.strip() for p in raw.split(separator) if p.strip()]
        if not phones:
            errors.append({"word": word, "raw_espeak_output": raw, "reason": "empty phone sequence"})
            continue
        phone_inventory.update(phones)
        rows.append({
            "word": word,
            "phones": " ".join(phones),
            "phone_count": len(phones),
            "train_occurrences": int(train_counts[word]),
            "validation_occurrences": int(val_counts[word]),
            "test_occurrences": int(extra_counts[word]),
            "source": "eSpeak NG Telugu voice (te); candidate only",
            "review_status": "NOT_REVIEWED",
            "corrected_phones": "",
            "review_notes": "",
        })

    lexicon = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors, columns=["word", "raw_espeak_output", "reason"])
    candidate_path = args.output_dir / "telugu_espeak_ipa_candidate.dict"
    audit_path = args.output_dir / "telugu_espeak_ipa_lexicon_audit.csv"
    review_path = args.output_dir / "telugu_espeak_ipa_manual_review.csv"
    errors_path = args.output_dir / "telugu_espeak_ipa_lexicon_errors.csv"
    phone_path = args.output_dir / "telugu_espeak_ipa_phone_inventory.json"
    override_path = args.output_dir / "telugu_lexicon_overrides.csv"
    reviewed_path = args.output_dir / "telugu_mfa_dictionary_reviewed.dict"

    with candidate_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in lexicon.sort_values("word").to_dict("records"):
            f.write(f"{row['word']}\t{row['phones']}\n")
    lexicon.to_csv(audit_path, index=False, encoding="utf-8-sig")
    lexicon.to_csv(review_path, index=False, encoding="utf-8-sig")
    error_df.to_csv(errors_path, index=False, encoding="utf-8-sig")
    phone_path.write_text(json.dumps(sorted(phone_inventory), ensure_ascii=False, indent=2), encoding="utf-8")

    if not override_path.exists():
        pd.DataFrame(columns=["word", "corrected_phones", "notes"]).to_csv(override_path, index=False, encoding="utf-8-sig")

    overrides = pd.read_csv(override_path, keep_default_na=False)
    override_map: dict[str, str] = {}
    for item in overrides.to_dict("records"):
        word = unicodedata.normalize("NFC", str(item.get("word", "")).strip())
        phones = str(item.get("corrected_phones", "")).strip()
        if not word and not phones:
            continue
        if not word or not phones or any(ch.isspace() for ch in word):
            raise ValueError(f"Invalid manual override row: {item}")
        if word in override_map:
            raise ValueError(f"Duplicate manual override: {word}")
        override_map[word] = " ".join(phones.split())

    mapping = {row["word"]: row["phones"] for row in rows}
    mapping.update(override_map)
    unresolved = sorted(set(error_df["word"]) - set(override_map)) if not error_df.empty else []
    with reviewed_path.open("w", encoding="utf-8", newline="\n") as f:
        for word in sorted(mapping):
            f.write(f"{word}\t{mapping[word]}\n")

    final_inventory = sorted({p for pronunciation in mapping.values() for p in pronunciation.split()})
    phone_path.write_text(json.dumps(final_inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "unique_train_words": len(train_counts),
        "unique_validation_words": len(val_counts),
        "validation_only_words": len(set(val_counts) - set(train_counts)),
        "test_only_words": len(set(extra_counts) - set(train_counts) - set(val_counts)),
        "candidate_entries": len(rows),
        "g2p_errors": len(errors),
        "manual_overrides": len(override_map),
        "unresolved_g2p_errors": len(unresolved),
        "phone_inventory_size": len(final_inventory),
        "candidate_dictionary": str(candidate_path),
        "reviewed_dictionary": str(reviewed_path),
        "review_sheet": str(review_path),
        "override_template": str(override_path),
    }
    (args.output_dir / "telugu_lexicon_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if unresolved:
        print("WARNING: unresolved G2P-error words require manual pronunciations before MFA training.")
    print("IMPORTANT: zero G2P errors does not mean pronunciations are verified.")


if __name__ == "__main__":
    main()
