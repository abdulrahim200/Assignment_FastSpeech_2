#!/usr/bin/env python3
"""Patch a separate FastSpeech 2 checkout with the Telugu IPA phone symbols.

The script appends @<phone> tokens to text/symbols.py. Run it only in the
Telugu copy of the reference implementation, never in the English checkout.
It is idempotent and stores a backup before changing the file.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MARKER_BEGIN = "# BEGIN GENERATED TELUGU IPA SYMBOLS"
MARKER_END = "# END GENERATED TELUGU IPA SYMBOLS"


def read_phones(dictionary: Path) -> list[str]:
    phones = set()
    for line_no, line in enumerate(dictionary.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Malformed dictionary line {line_no}")
        phones.update(parts[1].split())
    # MFA silence symbols can appear in TextGrid phone tiers even when they are
    # not ordinary dictionary phones.
    phones.update({"sil", "sp", "spn"})
    return sorted(phones)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="FastSpeech 2 Telugu checkout root")
    parser.add_argument("--dictionary", type=Path, default=Path("data/processed/telugu/lexicon/telugu_mfa_dictionary_reviewed.dict"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    symbols_file = repo_root / "text/symbols.py"
    dictionary = args.dictionary if args.dictionary.is_absolute() else (repo_root / args.dictionary)
    if not symbols_file.is_file():
        raise FileNotFoundError(symbols_file)
    phones = read_phones(dictionary)
    source = symbols_file.read_text(encoding="utf-8")
    if MARKER_BEGIN in source and MARKER_END in source:
        before = source.split(MARKER_BEGIN)[0].rstrip()
    else:
        backup = symbols_file.with_suffix(".py.pre_telugu_backup")
        if not backup.exists():
            shutil.copy2(symbols_file, backup)
        before = source.rstrip()
    block = (
        f"\n\n{MARKER_BEGIN}\n"
        f"_telugu_ipa_symbols = {json.dumps(['@' + p for p in phones], ensure_ascii=False, indent=4)}\n"
        "symbols = symbols + [s for s in _telugu_ipa_symbols if s not in symbols]\n"
        f"{MARKER_END}\n"
    )
    symbols_file.write_text(before + block, encoding="utf-8")
    (repo_root / "text/telugu_phone_inventory.json").write_text(
        json.dumps(phones, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Patched:", symbols_file)
    print("IPA phones added:", len(phones))
    print("Backup:", symbols_file.with_suffix(".py.pre_telugu_backup"))
    print("IMPORTANT: Recreate the Telugu model from scratch after changing its symbol inventory; do not load English checkpoints.")


if __name__ == "__main__":
    main()
