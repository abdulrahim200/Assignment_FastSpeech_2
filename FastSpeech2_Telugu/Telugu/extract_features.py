#!/usr/bin/env python3
"""Create FastSpeech 2 arrays and metadata from normalized WAVs + MFA TextGrids.

This is the final Telugu preprocessing stage. It requires real phone-tier
TextGrids for every utterance; it will not fabricate durations. It uses the
reference repository's TacotronSTFT for mel/energy so that feature extraction
matches the FastSpeech 2 implementation.

Example (after full train and validation alignment):
  python Telugu/extract_features.py --repo-root .

Expected TextGrid layout:
  data/processed/telugu/mfa_alignments/train_sat2_v1/<speaker>/<utt>.TextGrid
  data/processed/telugu/mfa_alignments/validation_sat2_v1/<speaker>/<utt>.TextGrid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import pyworld
import tgt

SAMPLE_RATE = 22050
HOP_LENGTH = 256
FILTER_LENGTH = 1024
WIN_LENGTH = 1024
N_MEL_CHANNELS = 80
MEL_FMIN = 0
MEL_FMAX = 8000
F0_MIN = 50
F0_MAX = 600
SILENCE_LABELS = {"sil", "sp", "spn"}


def find_textgrid(root: Path, utterance_id: str) -> Path:
    matches = list(root.rglob(f"{utterance_id}.TextGrid")) + list(root.rglob(f"{utterance_id}.textgrid"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one TextGrid for {utterance_id}, found {len(matches)} under {root}")
    return matches[0]


def load_phone_segments(textgrid_path: Path) -> list[tuple[str, float, float]]:
    grid = tgt.io.read_textgrid(str(textgrid_path))
    try:
        tier = grid.get_tier_by_name("phones")
    except Exception as exc:
        raise ValueError(f"No 'phones' tier in {textgrid_path}; tiers={grid.get_tier_names()}") from exc
    intervals = [iv for iv in tier.intervals if str(iv.text).strip()]
    if not intervals:
        raise ValueError(f"No non-empty phone intervals in {textgrid_path}")

    segments: list[tuple[str, float, float]] = []
    for i, iv in enumerate(intervals):
        label = str(iv.text).strip()
        start = float(iv.start_time)
        end = float(iv.end_time)
        if end < start:
            raise ValueError(f"Negative interval: {label} {start}..{end} in {textgrid_path}")
        if segments and start < segments[-1][2] - 0.001:
            raise ValueError(f"Overlapping phone intervals near {label} in {textgrid_path}")
        if segments and start - segments[-1][2] > 0.001:
            segments.append(("sil", segments[-1][2], start))
        segments.append((label, start, end))
    return segments


def extract_duration_targets(segments: list[tuple[str, float, float]], mel_frames: int) -> tuple[list[str], np.ndarray, float, float]:
    start_time = segments[0][1]
    end_time = segments[-1][2]
    if end_time <= start_time:
        raise ValueError("Invalid alignment time range")

    phones: list[str] = []
    boundaries = [0]
    for label, start, end in segments:
        if not label:
            continue
        phones.append(label)
        frame = int(np.round((end - start_time) * SAMPLE_RATE / HOP_LENGTH))
        boundaries.append(max(boundaries[-1], frame))

    if not phones:
        raise ValueError("No phones remain after TextGrid parsing")

    durations = np.diff(np.asarray(boundaries, dtype=np.int64))
    if np.any(durations < 0):
        raise ValueError("Negative phone duration generated")
    difference = int(mel_frames - durations.sum())
    # Centered STFT and TextGrid boundary quantization can differ by a frame or two.
    # Correct only a small discrepancy; larger differences indicate bad alignment.
    if abs(difference) > 3:
        raise ValueError(
            f"MFA duration sum {durations.sum()} differs from mel frames {mel_frames} by {difference}. "
            "Inspect the TextGrid/audio boundaries before continuing."
        )
    durations[-1] += difference
    if durations[-1] < 0:
        raise ValueError("Frame reconciliation made the final phone duration negative")
    return phones, durations.astype(np.int64), start_time, end_time


def fit_length(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) > length:
        return values[:length]
    if len(values) < length:
        return np.pad(values, (0, length - len(values)), mode="constant")
    return values


def make_stft(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    try:
        from audio.stft import TacotronSTFT
    except Exception as exc:
        raise RuntimeError(
            f"Could not import audio.stft.TacotronSTFT from {repo_root}. "
            "Run this script from a FastSpeech 2 repository checkout."
        ) from exc
    return TacotronSTFT(
        filter_length=FILTER_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mel_channels=N_MEL_CHANNELS,
        sampling_rate=SAMPLE_RATE,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
    )


def extract_features(wav_path: Path, textgrid_path: Path, stft: Any) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate}: {wav_path}")
    audio = np.asarray(audio, dtype=np.float32)

    segments = load_phone_segments(textgrid_path)
    phones = [seg[0] for seg in segments]
    start_time, end_time = segments[0][1], segments[-1][2]
    start_sample = max(0, int(round(start_time * SAMPLE_RATE)))
    end_sample = min(len(audio), int(round(end_time * SAMPLE_RATE)))
    trimmed = audio[start_sample:end_sample]
    if trimmed.size == 0:
        raise ValueError(f"Empty audio after MFA-boundary trimming: {wav_path}")

    audio_tensor = torch.from_numpy(trimmed).float().unsqueeze(0)
    with torch.no_grad():
        mel_tensor, energy_tensor = stft.mel_spectrogram(audio_tensor)
    mel = mel_tensor.squeeze(0).cpu().numpy().astype(np.float32).T
    energy_frame = energy_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    mel_frames = int(mel.shape[0])
    energy_frame = fit_length(energy_frame, mel_frames)

    frame_period_ms = HOP_LENGTH / SAMPLE_RATE * 1000.0
    f0_raw, time_axis = pyworld.dio(
        trimmed.astype(np.float64), SAMPLE_RATE,
        f0_floor=F0_MIN, f0_ceil=F0_MAX,
        frame_period=frame_period_ms,
    )
    f0 = pyworld.stonemask(trimmed.astype(np.float64), f0_raw, time_axis, SAMPLE_RATE).astype(np.float32)
    f0_frame = fit_length(f0, mel_frames)

    # Shift TextGrid times into the trimmed-audio coordinate system.
    shifted = [(label, start - start_time, end - start_time) for label, start, end in segments]
    phones, durations, _, _ = extract_duration_targets(shifted, mel_frames)
    if len(phones) != len(durations) or int(durations.sum()) != mel_frames:
        raise ValueError("Phone/duration/mel-frame contract failed")

    pitch_phone = np.zeros(len(durations), dtype=np.float32)
    energy_phone = np.zeros(len(durations), dtype=np.float32)
    cursor = 0
    for i, duration in enumerate(durations):
        d = int(duration)
        p_segment = f0_frame[cursor:cursor + d]
        e_segment = energy_frame[cursor:cursor + d]
        voiced = p_segment[p_segment > 0]
        pitch_phone[i] = float(voiced.mean()) if len(voiced) else 0.0
        energy_phone[i] = float(e_segment.mean()) if len(e_segment) else 0.0
        cursor += d

    if not all(np.isfinite(x).all() for x in (mel, pitch_phone, energy_phone)):
        raise ValueError(f"NaN/Inf feature values in {wav_path}")
    return {
        "phones": phones,
        "durations": durations,
        "mel": mel,
        "pitch": pitch_phone,
        "energy": energy_phone,
        "voiced_phone_count": int(np.count_nonzero(pitch_phone)),
        "trimmed_duration_sec": len(trimmed) / SAMPLE_RATE,
    }


def robust_reference(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        kept = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        if kept.size:
            values = kept
    mean = float(values.mean())
    std = float(values.std())
    return mean, (std if std > 1e-8 else 1.0)


def load_export_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"utterance_id", "speaker", "text", "audio_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df


def process_split(split: str, frame: pd.DataFrame, alignment_root: Path, stft: Any, output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, meta = [], []
    for row in frame.to_dict("records"):
        utt = str(row["utterance_id"])
        wav_path = Path(row["audio_path"])
        tg_path = find_textgrid(alignment_root, utt)
        features = extract_features(wav_path, tg_path, stft)
        records.append({**features, "utterance_id": utt, "speaker": str(row["speaker"]), "text": str(row["text"]), "split": split})
    return records, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="FastSpeech 2 repository root")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/telugu"))
    parser.add_argument("--train-alignment", type=Path, default=Path("data/processed/telugu/mfa_alignments/train_sat2_v1"))
    parser.add_argument("--validation-alignment", type=Path, default=Path("data/processed/telugu/mfa_alignments/validation_sat2_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/telugu/fastspeech2_preprocessed"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    processed_root = args.processed_root
    train_manifest = load_export_manifest(processed_root / "manifests/telugu_train_manifest.csv")
    val_manifest = load_export_manifest(processed_root / "manifests/telugu_validation_manifest.csv")
    if set(train_manifest.utterance_id) & set(val_manifest.utterance_id):
        raise ValueError("Train/validation identifiers overlap")

    for name, df, tg_root in (("train", train_manifest, args.train_alignment), ("validation", val_manifest, args.validation_alignment)):
        missing = [u for u in df.utterance_id if not list(tg_root.rglob(f"{u}.TextGrid")) and not list(tg_root.rglob(f"{u}.textgrid"))]
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} missing TextGrids; first examples: {missing[:10]}")

    stft = make_stft(repo_root)
    train_records, _ = process_split("train", train_manifest, args.train_alignment, stft, args.output_root)
    val_records, _ = process_split("validation", val_manifest, args.validation_alignment, stft, args.output_root)

    # Fit normalization statistics on training records only.
    raw_train_pitch = np.concatenate([r["pitch"] for r in train_records])
    raw_train_energy = np.concatenate([r["energy"] for r in train_records])
    pitch_fit = raw_train_pitch[raw_train_pitch > 0]
    energy_fit = raw_train_energy
    pitch_mean, pitch_std = robust_reference(pitch_fit if pitch_fit.size else raw_train_pitch)
    energy_mean, energy_std = robust_reference(energy_fit)

    speakers = sorted({r["speaker"] for r in train_records + val_records})
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(speakers)}
    output_root = args.output_root
    for sub in ("mel", "pitch", "energy", "duration"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    def write_records(records: list[dict[str, Any]], split: str) -> list[str]:
        metadata_lines = []
        for r in records:
            utt = r["utterance_id"]
            mel = np.asarray(r["mel"], dtype=np.float32)
            durations = np.asarray(r["durations"], dtype=np.int64)
            pitch = (np.asarray(r["pitch"], dtype=np.float32) - pitch_mean) / pitch_std
            energy = (np.asarray(r["energy"], dtype=np.float32) - energy_mean) / energy_std
            if mel.ndim != 2 or mel.shape[1] != N_MEL_CHANNELS:
                raise ValueError(f"Invalid mel shape for {utt}: {mel.shape}")
            if int(durations.sum()) != len(mel) or len(durations) != len(r["phones"]):
                raise ValueError(f"Duration/phone/mel mismatch for {utt}")
            if len(pitch) != len(durations) or len(energy) != len(durations):
                raise ValueError(f"Pitch/energy length mismatch for {utt}")
            np.save(output_root / "mel" / f"{utt}.npy", mel)
            np.save(output_root / "duration" / f"{utt}.npy", durations)
            np.save(output_root / "pitch" / f"{utt}.npy", pitch.astype(np.float32))
            np.save(output_root / "energy" / f"{utt}.npy", energy.astype(np.float32))
            phone_sequence = "{" + " ".join(r["phones"]) + "}"
            speaker_label = r["speaker"]
            metadata_lines.append(f"{utt}|{speaker_label}|{phone_sequence}|{r['text']}")
        return metadata_lines

    train_lines = write_records(train_records, "train")
    val_lines = write_records(val_records, "validation")
    (output_root / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_root / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    (output_root / "speakers.json").write_text(json.dumps(speaker_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    # Min/max are computed after training-stat normalization; mean/std retain
    # the raw training statistics, matching the reference repo's stats.json convention.
    all_pitch_norm = np.concatenate([(r["pitch"] - pitch_mean) / pitch_std for r in train_records])
    all_energy_norm = np.concatenate([(r["energy"] - energy_mean) / energy_std for r in train_records])
    stats = {
        "pitch": [float(all_pitch_norm.min()), float(all_pitch_norm.max()), float(pitch_mean), float(pitch_std)],
        "energy": [float(all_energy_norm.min()), float(all_energy_norm.max()), float(energy_mean), float(energy_std)],
    }
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    summary = {
        "train_recordings": len(train_records),
        "validation_recordings": len(val_records),
        "train_phones": int(sum(len(r["phones"]) for r in train_records)),
        "validation_phones": int(sum(len(r["phones"]) for r in val_records)),
        "train_mel_frames": int(sum(len(r["mel"]) for r in train_records)),
        "validation_mel_frames": int(sum(len(r["mel"]) for r in val_records)),
        "speaker_to_id": speaker_to_id,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "output_root": str(output_root),
    }
    (output_root / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
