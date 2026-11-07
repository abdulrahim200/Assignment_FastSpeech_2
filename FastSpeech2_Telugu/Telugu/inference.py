#!/usr/bin/env python3
"""Single-sentence Telugu inference wrapper for the reference FastSpeech 2 code.

Requires: a completed Telugu FastSpeech 2 checkpoint, a patched Telugu IPA
symbol vocabulary, a reviewed dictionary, and the reference repo's HiFi-GAN
files. This script has not been run yet because a Telugu FastSpeech 2 model has
not been trained at the current project checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Telugu.text_frontend import load_lexicon, words_to_phones  # noqa: E402


def synthesize_one(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    import os
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    from text import text_to_sequence
    from utils.model import get_model, get_vocoder
    from utils.tools import to_device, synth_samples

    preprocess_config = yaml.safe_load(args.preprocess_config.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    train_config = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    lexicon = load_lexicon(args.dictionary)
    phones = words_to_phones(args.text, lexicon)
    phone_text = "{" + " ".join(phones) + "}"
    cleaner_names = preprocess_config["preprocessing"]["text"].get("text_cleaners", [])
    sequence = np.asarray(text_to_sequence(phone_text, cleaner_names), dtype=np.int64)
    if sequence.size == 0:
        raise ValueError("Telugu text frontend returned an empty sequence")

    speaker_map_path = Path(preprocess_config["path"]["preprocessed_path"]) / "speakers.json"
    speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
    if args.speaker not in speaker_map:
        raise KeyError(f"Speaker {args.speaker!r} not in {speaker_map_path}; available={speaker_map}")
    speaker_id = int(speaker_map[args.speaker])

    checkpoint_path = Path(train_config["path"]["ckpt_path"]) / f"{args.restore_step}.pth.tar"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Telugu FastSpeech 2 checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(restore_step=args.restore_step)
    configs = (preprocess_config, model_config, train_config)
    model = get_model(model_args, configs, device, train=False).to(device)
    vocoder = get_vocoder(model_config, device)

    ids = [args.utterance_id]
    raw_texts = [args.text]
    speakers = np.asarray([speaker_id], dtype=np.int64)
    texts = np.asarray([sequence], dtype=np.int64)
    text_lens = np.asarray([len(sequence)], dtype=np.int64)
    batch = (ids, raw_texts, speakers, texts, text_lens, int(len(sequence)))
    batch = to_device(batch, device)
    model.eval()
    with torch.no_grad():
        output = model(
            *(batch[2:]),
            p_control=args.pitch_control,
            e_control=args.energy_control,
            d_control=args.duration_control,
        )
        result_dir = Path(train_config["path"]["result_path"])
        result_dir.mkdir(parents=True, exist_ok=True)
        synth_samples(batch, output, vocoder, model_config, preprocess_config, str(result_dir))
    print("Raw Telugu text:", args.text)
    print("Phone sequence:", phone_text)
    print("Speaker ID:", speaker_id)
    print("Results directory:", result_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--utterance-id", default="telugu_inference")
    parser.add_argument("--speaker", default="female", help="Speaker key from speakers.json (female or male)")
    parser.add_argument("--restore-step", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dictionary", type=Path, default=PROJECT_ROOT / "data/processed/telugu/lexicon/telugu_mfa_dictionary_reviewed.dict")
    parser.add_argument("--preprocess-config", type=Path, default=HERE / "configs/preprocess.yaml")
    parser.add_argument("--model-config", type=Path, default=HERE / "configs/model.yaml")
    parser.add_argument("--train-config", type=Path, default=HERE / "configs/train.yaml")
    parser.add_argument("--pitch-control", type=float, default=1.0)
    parser.add_argument("--energy-control", type=float, default=1.0)
    parser.add_argument("--duration-control", type=float, default=1.0)
    args = parser.parse_args()
    synthesize_one(args)


if __name__ == "__main__":
    main()
