#!/usr/bin/env python3
"""Plot FastSpeech 2 predicted duration, pitch and energy for Telugu test text.

Requires a trained Telugu FastSpeech 2 checkpoint and the Telugu-specific
symbol inventory. This script is prepared for the post-training evaluation
stage; it has not been executed because Telugu FastSpeech 2 training is pending.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Telugu.text_frontend import load_lexicon, words_to_phones  # noqa: E402


def load_model_and_assets(args):
    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    from text import text_to_sequence
    from utils.model import get_model
    from utils.tools import to_device

    preprocess = yaml.safe_load(args.preprocess_config.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    train_config = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))

    checkpoint = Path(train_config["path"]["ckpt_path"]) / f"{args.restore_step}.pth.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Telugu FastSpeech 2 checkpoint not found: {checkpoint}")

    preprocessed = Path(preprocess["path"]["preprocessed_path"])
    if not preprocessed.is_absolute():
        preprocessed = (repo_root / preprocessed).resolve()
    speaker_map_path = preprocessed / "speakers.json"
    stats_path = preprocessed / "stats.json"
    if not speaker_map_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("speakers.json and stats.json must exist in the Telugu preprocessed directory")

    speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
    if args.speaker not in speaker_map:
        raise KeyError(f"Speaker {args.speaker!r} not found; available: {speaker_map}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    lexicon = load_lexicon(args.dictionary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(restore_step=args.restore_step)
    model = get_model(model_args, (preprocess, model_config, train_config), device, train=False).to(device)
    model.eval()

    return model, device, preprocess, model_config, train_config, speaker_map, stats, lexicon, text_to_sequence, to_device


def _valid_1d(tensor: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    values = tensor.detach().float().squeeze(0)
    valid = (~mask.detach().bool()).squeeze(0)
    while valid.ndim > 1:
        valid = valid.squeeze(0)
    if values.ndim > 1:
        values = values.squeeze(-1)
    return values[valid].cpu().numpy()


def plot_one(item: dict, args, assets) -> None:
    (model, device, preprocess, model_config, train_config, speaker_map,
     stats, lexicon, text_to_sequence, to_device) = assets

    phones = words_to_phones(item["text"], lexicon)
    # The reference frontend maps tokens inside braces to @phone symbols.
    phone_text = "{" + " ".join(phones) + "}"
    cleaners = preprocess["preprocessing"]["text"].get("text_cleaners", [])
    sequence = np.asarray(text_to_sequence(phone_text, cleaners), dtype=np.int64)
    if sequence.size != len(phones):
        raise ValueError(
            f"Frontend produced {sequence.size} IDs for {len(phones)} phones in {item['id']}. "
            "Check the Telugu symbols patch and phone inventory."
        )

    speakers = np.asarray([int(speaker_map[args.speaker])], dtype=np.int64)
    texts = np.asarray([sequence], dtype=np.int64)
    text_lens = np.asarray([len(sequence)], dtype=np.int64)
    batch = ([item["id"]], [item["text"]], speakers, texts, text_lens, int(len(sequence)))
    batch = to_device(batch, device)

    with torch.no_grad():
        output = model(
            *(batch[2:]),
            p_control=args.pitch_control,
            e_control=args.energy_control,
            d_control=args.duration_control,
        )

    # Reference FastSpeech2 output order:
    # 2=pitch predictions, 3=energy predictions, 4=log-duration predictions,
    # 5=rounded predicted durations, 6=source padding mask.
    src_mask = output[6]
    durations = _valid_1d(output[5], src_mask)
    pitch_norm = _valid_1d(output[2], src_mask)
    energy_norm = _valid_1d(output[3], src_mask)

    pitch_stats = stats["pitch"]
    energy_stats = stats["energy"]
    pitch_hz = pitch_norm * float(pitch_stats[3]) + float(pitch_stats[2])
    energy_raw = energy_norm * float(energy_stats[3]) + float(energy_stats[2])

    if not (len(phones) == len(durations) == len(pitch_hz) == len(energy_raw)):
        raise ValueError(f"Predicted variance arrays do not match phones for {item['id']}")

    out_root = args.output_dir
    dirs = {name: out_root / name for name in ("duration", "pitch", "energy")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(phones))
    phone_labels = phones
    fig, ax = plt.subplots(figsize=(max(10, len(phones) * 0.23), 4.2))
    ax.bar(x, durations)
    ax.set_title(f"{item['id']} - predicted duration (mel frames)")
    ax.set_xlabel("Phone index")
    ax.set_ylabel("Predicted duration (frames)")
    ax.set_xticks(x)
    ax.set_xticklabels(phone_labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(dirs["duration"] / f"{item['id']}_duration.png", dpi=160)
    plt.close(fig)

    for name, values, ylabel, title in (
        ("pitch", pitch_hz, "Predicted F0 (Hz)", "predicted pitch"),
        ("energy", energy_raw, "Predicted energy (reference feature scale)", "predicted energy"),
    ):
        fig, ax = plt.subplots(figsize=(max(10, len(phones) * 0.23), 4.2))
        ax.plot(x, values, marker="o", linewidth=1.2, markersize=3)
        ax.set_title(f"{item['id']} - {title}")
        ax.set_xlabel("Phone index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(phone_labels, rotation=90, fontsize=7)
        fig.tight_layout()
        fig.savefig(dirs[name] / f"{item['id']}_{name}.png", dpi=160)
        plt.close(fig)

    csv_path = out_root / f"{item['id']}_variance_predictions.csv"
    import pandas as pd
    pd.DataFrame({
        "phone_index": x,
        "phone": phones,
        "predicted_duration_frames": durations.astype(int),
        "predicted_pitch_hz": pitch_hz,
        "predicted_energy_raw_scale": energy_raw,
    }).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved duration/pitch/energy plots for {item['id']} to {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-step", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dictionary", type=Path, default=PROJECT_ROOT / "data/processed/telugu/lexicon/telugu_mfa_dictionary_reviewed.dict")
    parser.add_argument("--sentences", type=Path, default=HERE / "test_sentences.json")
    parser.add_argument("--speaker", default="female")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Results/Telugu/variance_adaptor_plots")
    parser.add_argument("--preprocess-config", type=Path, default=HERE / "configs/preprocess.yaml")
    parser.add_argument("--model-config", type=Path, default=HERE / "configs/model.yaml")
    parser.add_argument("--train-config", type=Path, default=HERE / "configs/train.yaml")
    parser.add_argument("--pitch-control", type=float, default=1.0)
    parser.add_argument("--energy-control", type=float, default=1.0)
    parser.add_argument("--duration-control", type=float, default=1.0)
    args = parser.parse_args()

    data = json.loads(args.sentences.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Test-sentence JSON must be a non-empty list of {id, text} objects")

    assets = load_model_and_assets(args)
    for item in data:
        if not item.get("id") or not item.get("text"):
            raise ValueError(f"Invalid test-sentence item: {item}")
        plot_one(item, args, assets)


if __name__ == "__main__":
    main()
