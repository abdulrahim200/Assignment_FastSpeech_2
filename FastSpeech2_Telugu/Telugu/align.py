#!/usr/bin/env python3
"""MFA 3.x command runner for the Telugu IndicTTS corpus.

The current recovered checkpoint should be copied to Telugu/checkpoints before
running align commands. This script does not train automatically; choose a
subcommand explicitly.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_ROOT = PROJECT_ROOT / "data/processed/telugu"
CORPUS_ROOT = DATA_ROOT / "mfa_corpus"
LEXICON = DATA_ROOT / "lexicon/telugu_mfa_dictionary_reviewed.dict"
MODEL = HERE / "checkpoints/telugu_acoustic_model_sat2_recovered.zip"
MFA_DEFAULT = Path("mfa")


def get_mfa_executable() -> str:
    configured = os.environ.get("MFA_BIN")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which(str(MFA_DEFAULT))
    if found:
        return found
    raise FileNotFoundError(
        "MFA executable not found. Set MFA_BIN or install MFA in a Conda environment."
    )


def make_env(mfa_executable: str, root_dir: Path, temp_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(Path(mfa_executable).resolve().parent) + os.pathsep + env.get("PATH", "")
    env["MFA_ROOT_DIR"] = str(root_dir)
    env["TMPDIR"] = str(temp_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "train", "smoke-test", "align-train", "align-validation"])
    parser.add_argument("--train-corpus", type=Path, default=CORPUS_ROOT / "train")
    parser.add_argument("--validation-corpus", type=Path, default=CORPUS_ROOT / "validation")
    parser.add_argument("--dictionary", type=Path, default=LEXICON)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT / "mfa_alignments")
    parser.add_argument("--num-jobs", type=int, default=2)
    parser.add_argument("--audio", type=Path, default=CORPUS_ROOT / "train/female/indictts_00002.wav")
    parser.add_argument("--transcript", type=Path, default=None)
    args = parser.parse_args()

    mfa_exe = get_mfa_executable()
    # Keep MFA caches in the platform temp area rather than hard-coding Colab.
    temp_base = Path(tempfile.gettempdir())
    root_dir = Path(os.environ.get("MFA_ROOT_DIR", str(temp_base / "mfa_root_telugu_repo")))
    env = make_env(mfa_exe, root_dir, temp_base / "mfa_temp_telugu_repo")
    if not args.dictionary.is_file():
        raise FileNotFoundError(args.dictionary)

    if args.action == "validate":
        if not args.train_corpus.is_dir():
            raise FileNotFoundError(args.train_corpus)
        report_dir = args.output_root / "validation_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        run([
            mfa_exe, "validate", "--num_jobs", str(args.num_jobs),
            "--overwrite", "--temporary_directory", str(temp_base / "mfa_temp_telugu_validate"),
            "--output_directory", str(report_dir),
            str(args.train_corpus), str(args.dictionary),
        ], env)
        return

    if args.action == "train":
        if not args.train_corpus.is_dir():
            raise FileNotFoundError(args.train_corpus)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        if args.model.exists():
            raise FileExistsError(f"Refusing to overwrite existing model: {args.model}")
        train_alignment_dir = args.output_root / "train_during_model_training"
        train_alignment_dir.mkdir(parents=True, exist_ok=True)
        run([
            mfa_exe, "train", "--num_jobs", str(args.num_jobs),
            "--temporary_directory", str(temp_base / "mfa_temp_telugu_repo_train"),
            "--output_directory", str(train_alignment_dir),
            str(args.train_corpus), str(args.dictionary), str(args.model),
        ], env)
        return

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Acoustic model missing: {args.model}. Copy the recovered SAT2 archive into Telugu/checkpoints."
        )

    if args.action == "smoke-test":
        transcript = args.transcript or args.audio.with_suffix(".lab")
        if not args.audio.is_file() or not transcript.is_file():
            raise FileNotFoundError(f"WAV/LAB pair not found: {args.audio} / {transcript}")
        output_dir = args.output_root / "smoke_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.audio.stem}.TextGrid"
        run([
            mfa_exe, "align_one", "--num_jobs", "1",
            "--overwrite", "--temporary_directory", str(temp_base / "mfa_temp_telugu_align_one"),
            "--output_format", "long_textgrid",
            str(args.audio), str(transcript), str(args.dictionary), str(args.model), str(output_path),
        ], env)
        print("TextGrid:", output_path)
        return

    if args.action == "align-train":
        corpus = args.train_corpus
        output_dir = args.output_root / "train_sat2_v1"
    else:
        corpus = args.validation_corpus
        output_dir = args.output_root / "validation_sat2_v1"
    if not corpus.is_dir():
        raise FileNotFoundError(corpus)
    output_dir.mkdir(parents=True, exist_ok=True)
    run([
        mfa_exe, "align", "--num_jobs", str(args.num_jobs),
        "--overwrite", "--temporary_directory", str(temp_base / f"mfa_temp_telugu_{args.action}"),
        "--output_format", "long_textgrid",
        str(corpus), str(args.dictionary), str(args.model), str(output_dir),
    ], env)
    textgrids = list(output_dir.rglob("*.TextGrid"))
    expected = len(list(corpus.rglob("*.wav")))
    print(f"TextGrids: {len(textgrids)} / {expected}")


if __name__ == "__main__":
    main()
