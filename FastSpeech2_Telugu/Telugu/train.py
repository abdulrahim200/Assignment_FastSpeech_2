#!/usr/bin/env python3
"""Guarded launcher for the reference FastSpeech 2 training script.

This wrapper does not implement a second FastSpeech 2 trainer. It validates the
Telugu artifacts and invokes the selected reference implementation's train.py.
Training is intentionally blocked until full MFA alignments and FastSpeech 2
feature arrays exist.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_PREPROCESSED = PROJECT_ROOT / "data/processed/telugu/fastspeech2_preprocessed"


def validate_preprocessed(root: Path) -> tuple[int, int]:
    train_meta = root / "train.txt"
    val_meta = root / "val.txt"
    for path in (train_meta, val_meta, root / "speakers.json", root / "stats.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required Telugu FastSpeech 2 artifact missing: {path}")
    train_lines = [x for x in train_meta.read_text(encoding="utf-8").splitlines() if x.strip()]
    val_lines = [x for x in val_meta.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not train_lines or not val_lines:
        raise ValueError("train.txt and val.txt must both contain records")
    for split, lines in (("train", train_lines), ("validation", val_lines)):
        ids = [line.split("|", 1)[0] for line in lines]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate utterance IDs in {split}.txt")
        for subdir in ("mel", "pitch", "energy", "duration"):
            missing = [utt for utt in ids if not (root / subdir / f"{utt}.npy").is_file()]
            if missing:
                raise FileNotFoundError(f"{split}: {len(missing)} missing {subdir} arrays; examples={missing[:5]}")
    train_ids = {line.split("|", 1)[0] for line in train_lines}
    val_ids = {line.split("|", 1)[0] for line in val_lines}
    if train_ids & val_ids:
        raise ValueError("Train/validation metadata IDs overlap")
    return len(train_lines), len(val_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT, help="Root of the ming024/FastSpeech2 source checkout")
    parser.add_argument("--preprocess-config", type=Path, default=HERE / "configs/preprocess.yaml")
    parser.add_argument("--model-config", type=Path, default=HERE / "configs/model.yaml")
    parser.add_argument("--train-config", type=Path, default=HERE / "configs/train.yaml")
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument("--restore-step", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Actually launch the reference trainer; without this flag only run preflight")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    reference_train = repo_root / "train.py"
    if not reference_train.is_file() or reference_train.resolve() == Path(__file__).resolve():
        raise FileNotFoundError(
            f"Reference FastSpeech 2 train.py not found at {reference_train}. "
            "Point --repo-root at the upstream ming024/FastSpeech2 checkout."
        )
    train_count, val_count = validate_preprocessed(args.preprocessed_root)
    for config_path in (args.preprocess_config, args.model_config, args.train_config):
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
    preprocess = yaml.safe_load(args.preprocess_config.read_text(encoding="utf-8"))
    configured_preprocessed = Path(preprocess["path"]["preprocessed_path"])
    if not configured_preprocessed.is_absolute():
        configured_preprocessed = (repo_root / configured_preprocessed).resolve()
    if configured_preprocessed.resolve() != args.preprocessed_root.resolve():
        raise ValueError(
            f"preprocess.yaml points to {configured_preprocessed}, but --preprocessed-root is {args.preprocessed_root}. "
            "Make the paths agree before training."
        )
    print(f"Preflight passed: {train_count} training and {val_count} validation records.")
    print("Training device is selected by the reference implementation (CUDA when available).")
    if not args.execute:
        print("No training was started. Re-run with --execute when you are ready.")
        return

    # The reference Dataset resolves train.txt and val.txt relative to
    # preprocess_config.path.preprocessed_path, so keep the generated metadata
    # in that directory and do not create or overwrite files in the repo root.
    command = [
        sys.executable, str(reference_train),
        "-p", str(args.preprocess_config.resolve()),
        "-m", str(args.model_config.resolve()),
        "-t", str(args.train_config.resolve()),
        "--restore_step", str(args.restore_step),
    ]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
