#!/usr/bin/env python3
"""Prepare the local IndicTTS Telugu split for MFA.

This script deliberately reuses the existing manifests and raw WAVs. It does not
redownload the dataset and never modifies source files under data/raw/Telugu.
It writes mono, 22.05-kHz WAV copies and same-basename UTF-8 .lab transcripts
under data/processed/telugu/mfa_corpus/{train,validation}/<speaker>/.

Example (run from the repository root):
  python Telugu/preprocess.py \
    --train-manifest data/manifests/telugu_train.csv \
    --validation-manifest data/manifests/telugu_validation.csv \
    --raw-root data/raw/Telugu \
    --output-root data/processed/telugu
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 22050
AUDIO_SUBTYPE = "PCM_16"


def normalize_text(value: Any) -> str:
    """Apply only Unicode NFC normalization and whitespace normalization."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).strip()


def normalize_speaker(value: Any) -> str:
    speaker = "speaker_0" if value is None or pd.isna(value) else str(value).strip().lower()
    speaker = re.sub(r"[^a-z0-9_.-]+", "_", speaker).strip("._")
    return speaker or "speaker_0"


def make_utterance_id(value: Any, fallback_index: int) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return f"indictts_{fallback_index:05d}"
    raw = str(value).strip()
    stem = Path(raw).stem
    if stem.isdigit():
        stem = f"indictts_{int(stem):05d}"
    elif stem.lower().startswith("indictts_"):
        match = re.fullmatch(r"indictts_(\d+)", stem, flags=re.IGNORECASE)
        if match:
            stem = f"indictts_{int(match.group(1)):05d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    return frame


def resolve_audio(raw_root: Path, split: str, utterance_id: str, row: pd.Series) -> Path:
    """Resolve audio by an explicit manifest path or the verified IndicTTS layout."""
    for column in ("audio_path", "wav_path", "file_path", "filepath", "audio_file", "filename"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            candidate = Path(str(value).strip())
            if candidate.is_file():
                resolved = candidate.resolve()
                if raw_root.resolve() in resolved.parents:
                    return resolved
            if not candidate.is_absolute():
                for base in (raw_root, raw_root.parent.parent):
                    possible = (base / candidate).resolve()
                    if possible.is_file() and raw_root.resolve() in possible.parents:
                        return possible
    expected = raw_root / split / f"{utterance_id}.wav"
    if expected.is_file():
        return expected.resolve()
    matches = list((raw_root / split).glob(f"{utterance_id}.*"))
    matches = [p for p in matches if p.suffix.lower() in {".wav", ".flac"}]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(
        f"Cannot uniquely resolve {utterance_id!r} in {raw_root / split}; "
        "expected indictts_XXXXX.wav or a valid audio_path column."
    )


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Invalid/empty audio: {path} shape={audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains NaN/Inf: {path}")
    return audio, int(sample_rate)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(source_sr, target_sr)
    converted = resample_poly(audio, target_sr // gcd, source_sr // gcd)
    return np.asarray(converted, dtype=np.float32)


def export_split(
    manifest_path: Path,
    split: str,
    raw_root: Path,
    corpus_root: Path,
    target_sr: int,
) -> pd.DataFrame:
    frame = read_manifest(manifest_path)
    id_column = next((c for c in ("utterance_id", "dataset_index", "id", "utt_id") if c in frame.columns), None)
    text_column = next((c for c in ("text", "transcript", "sentence", "transcription") if c in frame.columns), None)
    speaker_column = next((c for c in ("speaker", "speaker_id", "gender") if c in frame.columns), None)
    if id_column is None or text_column is None:
        raise ValueError(
            f"{manifest_path} must contain an ID column (dataset_index/utterance_id) "
            f"and a transcript column (transcript/text). Found: {list(frame.columns)}"
        )

    exported: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, row in frame.iterrows():
        utterance_id = make_utterance_id(row.get(id_column), int(row_index))
        text = normalize_text(row.get(text_column))
        if not text:
            raise ValueError(f"Empty transcript for {utterance_id} in {manifest_path}")
        if utterance_id in seen_ids:
            raise ValueError(f"Duplicate utterance ID {utterance_id} in {manifest_path}")
        seen_ids.add(utterance_id)
        speaker = normalize_speaker(row.get(speaker_column) if speaker_column else None)
        source_path = resolve_audio(raw_root, split, utterance_id, row)
        audio, source_sr = load_audio(source_path)
        audio = resample_audio(audio, source_sr, target_sr)

        destination_dir = corpus_root / split / speaker
        destination_dir.mkdir(parents=True, exist_ok=True)
        wav_path = destination_dir / f"{utterance_id}.wav"
        lab_path = destination_dir / f"{utterance_id}.lab"
        sf.write(str(wav_path), audio, target_sr, subtype=AUDIO_SUBTYPE)
        lab_path.write_text(text, encoding="utf-8")

        exported.append({
            "utterance_id": utterance_id,
            "speaker": speaker,
            "text": text,
            "split": split,
            "source_audio_path": str(source_path),
            "audio_path": str(wav_path),
            "lab_path": str(lab_path),
            "source_sample_rate": source_sr,
            "sample_rate": target_sr,
            "frames": int(len(audio)),
            "duration_sec": float(len(audio) / target_sr),
        })

    result = pd.DataFrame(exported)
    if result.empty:
        raise RuntimeError(f"No recordings exported for {split}.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/telugu_train.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/manifests/telugu_validation.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/Telugu"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/telugu"))
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    args = parser.parse_args()

    corpus_root = args.output_root / "mfa_corpus"
    manifest_root = args.output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    train = export_split(args.train_manifest, "train", args.raw_root, corpus_root, args.sample_rate)
    validation = export_split(args.validation_manifest, "validation", args.raw_root, corpus_root, args.sample_rate)

    overlap = set(train.utterance_id) & set(validation.utterance_id)
    if overlap:
        raise ValueError(f"Train/validation identifier overlap ({len(overlap)}): {sorted(overlap)[:10]}")
    for name, split_df in (("train", train), ("validation", validation)):
        audio_paths = split_df.audio_path.map(Path)
        lab_paths = split_df.lab_path.map(Path)
        if not all(p.is_file() for p in audio_paths) or not all(p.is_file() for p in lab_paths):
            raise RuntimeError(f"Missing exported WAV/LAB files in {name} split.")
        if not split_df.sample_rate.eq(args.sample_rate).all():
            raise RuntimeError(f"Unexpected sample rate in {name} split.")
        split_df.to_csv(manifest_root / f"telugu_{name}_manifest.csv", index=False, encoding="utf-8-sig")

    combined = pd.concat([train, validation], ignore_index=True)
    combined.to_csv(manifest_root / "telugu_combined_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source": "local IndicTTS Telugu manifests and raw audio",
        "target_sample_rate": args.sample_rate,
        "train_recordings": len(train),
        "train_hours": round(float(train.duration_sec.sum() / 3600), 6),
        "validation_recordings": len(validation),
        "validation_hours": round(float(validation.duration_sec.sum() / 3600), 6),
        "train_validation_overlap": len(overlap),
        "speaker_counts_train": train.speaker.value_counts().to_dict(),
        "speaker_counts_validation": validation.speaker.value_counts().to_dict(),
        "source_sample_rates": combined.source_sample_rate.value_counts().to_dict(),
        "corpus_root": str(corpus_root),
    }
    (args.output_root / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
