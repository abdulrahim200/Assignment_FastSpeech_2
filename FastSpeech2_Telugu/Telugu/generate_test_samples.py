#!/usr/bin/env python3
"""Generate the assignment's five Telugu test audios after training.

This script delegates each sentence to inference.py so that each output uses
its own test ID and preserves the exact text stored in test_sentences.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-step", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sentences", type=Path, default=HERE / "test_sentences.json")
    parser.add_argument("--dictionary", type=Path, default=PROJECT_ROOT / "data/processed/telugu/lexicon/telugu_mfa_dictionary_reviewed.dict")
    parser.add_argument("--speaker", default="female")
    parser.add_argument("--pitch-control", type=float, default=1.0)
    parser.add_argument("--energy-control", type=float, default=1.0)
    parser.add_argument("--duration-control", type=float, default=1.0)
    args = parser.parse_args()

    items = json.loads(args.sentences.read_text(encoding="utf-8"))
    if len(items) != 5:
        raise ValueError(f"Expected exactly five Telugu test items, found {len(items)}")

    for item in items:
        command = [
            sys.executable, str(HERE / "inference.py"),
            "--repo-root", str(args.repo_root.resolve()),
            "--dictionary", str(args.dictionary.resolve()),
            "--restore-step", str(args.restore_step),
            "--utterance-id", str(item["id"]),
            "--speaker", args.speaker,
            "--text", str(item["text"]),
            "--pitch-control", str(args.pitch_control),
            "--energy-control", str(args.energy_control),
            "--duration-control", str(args.duration_control),
        ]
        print(f"\nGenerating {item['id']}...", flush=True)
        subprocess.run(command, cwd=args.repo_root.resolve(), check=True)

    print("\nAll five Telugu test-sentence synthesis commands completed.")


if __name__ == "__main__":
    main()
