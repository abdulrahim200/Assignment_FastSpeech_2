#!/usr/bin/env python3
"""Plot FastSpeech 2 training and validation losses from reference log.txt files.

The reference trainer writes one record every log_step and validation record
at val_step. Run after a Telugu training session has created Results/Telugu/logs.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TRAIN_RE = re.compile(
    r"Step\s+(?P<step>\d+)/\d+.*?Total Loss:\s*(?P<total>[-+0-9.eE]+).*?"
    r"Mel Loss:\s*(?P<mel>[-+0-9.eE]+).*?Mel PostNet Loss:\s*(?P<postnet>[-+0-9.eE]+).*?"
    r"Pitch Loss:\s*(?P<pitch>[-+0-9.eE]+).*?Energy Loss:\s*(?P<energy>[-+0-9.eE]+).*?"
    r"Duration Loss:\s*(?P<duration>[-+0-9.eE]+)"
)
VAL_RE = re.compile(
    r"Validation Step\s+(?P<step>\d+).*?Total Loss:\s*(?P<total>[-+0-9.eE]+).*?"
    r"Mel Loss:\s*(?P<mel>[-+0-9.eE]+).*?Mel PostNet Loss:\s*(?P<postnet>[-+0-9.eE]+).*?"
    r"Pitch Loss:\s*(?P<pitch>[-+0-9.eE]+).*?Energy Loss:\s*(?P<energy>[-+0-9.eE]+).*?"
    r"Duration Loss:\s*(?P<duration>[-+0-9.eE]+)"
)


def parse_log(path: Path, pattern: re.Pattern) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            row = match.groupdict()
            rows.append({k: float(v) if k != "step" else int(v) for k, v in row.items()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path("Results/Telugu/logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("Results/Telugu"))
    args = parser.parse_args()
    train = parse_log(args.log_root / "train/log.txt", TRAIN_RE)
    validation = parse_log(args.log_root / "val/log.txt", VAL_RE)
    if train.empty or validation.empty:
        raise RuntimeError(
            f"Could not find both train and validation loss records: train={len(train)}, validation={len(validation)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "training_losses.csv", index=False)
    validation.to_csv(args.output_dir / "validation_losses.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train.step, train.total, label="Training total loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Total loss")
    ax.set_title("Telugu FastSpeech 2 training loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(validation.step, validation.total, marker="o", label="Validation total loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Total loss")
    ax.set_title("Telugu FastSpeech 2 validation loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_loss.png", dpi=180)
    plt.close(fig)
    print(f"Saved plots and parsed CSVs under {args.output_dir}")


if __name__ == "__main__":
    main()
