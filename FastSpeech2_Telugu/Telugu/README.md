# Telugu FastSpeech 2 pipeline (IndicTTS)

This folder contains the Telugu-specific scripts extracted and organized from the Colab preparation notebook. It is designed to live inside the project root alongside the reference FastSpeech 2 code. It intentionally keeps Telugu paths, lexicon, model, speaker map and outputs separate from English.

## Current implementation status (7 November 2026)

Completed and verified:

- Reused local `telugu_train.csv` and `telugu_validation.csv` manifests and raw IndicTTS Telugu audio; no second Hugging Face download was made.
- Preserved the existing split: 1,544 train utterances / 5.003101 h; 313 validation utterances / 1.004286 h; no identifier overlap.
- Verified all 1,857 audio paths and WAV headers; source audio is mono, 1,146 files at 48 kHz and 711 at 16 kHz.
- Exported working WAV/LAB copies at 22,050 Hz. Training speaker counts: female 596, male 948; validation: female 115, male 198.
- Generated an eSpeak NG Telugu candidate lexicon with 13,461 words and 47 candidate phones. There were zero G2P generation errors, but the pronunciations were not manually verified. A standalone `ʲ` token occurs in 234 entries; do not assume those entries are correct.
- Installed MFA 3.4.3 in an isolated `/content/mfa-env` and installed/verified OpenFST (`fstcompile`). Structural validation passed: 1,544 WAVs, 1,544 LAB files, two speakers, no audio read errors, no missing transcriptions, and no OOV words.
- MFA training reached SAT2 and produced a recoverable `acoustic_model.zip`. The final pronunciation-probability calculation failed with `ZeroDivisionError`; the SAT2 archive was preserved and inspected.
- One `align_one` test for `female/indictts_00002.wav` succeeded and generated a 27,305-byte TextGrid.

Not completed yet:

- Full training-corpus TextGrid export using the recovered checkpoint.
- Full validation-corpus alignment.
- Bulk phone-duration, mel, pitch and energy feature export.
- Telugu FastSpeech 2 training, loss curves, neural checkpoint, inference audio, and variance-adaptor plots.

Do not describe the Telugu FastSpeech 2 model as trained until those steps finish.

## Expected project layout

```text
project-root/
├── Telugu/
│   ├── preprocess.py
│   ├── build_lexicon.py
│   ├── align.py
│   ├── extract_features.py
│   ├── patch_fastspeech2_symbols.py
│   ├── text_frontend.py
│   ├── train.py
│   ├── inference.py
│   ├── generate_test_samples.py
│   ├── plot_variance.py
│   ├── plot_losses.py
│   ├── config.yaml
│   ├── configs/
│   │   ├── preprocess.yaml
│   │   ├── model.yaml
│   │   └── train.yaml
│   ├── test_sentences.json
│   └── checkpoints/
├── data/                         # private/local; do not commit raw speech
│   ├── raw/Telugu/{train,validation}/
│   ├── manifests/
│   └── processed/telugu/
└── Results/Telugu/
```

## 1. Install Python dependencies

Use a Python environment compatible with the selected FastSpeech 2 checkout and PyTorch build. Install a CUDA-enabled PyTorch build separately if using Colab L4.

```bash
pip install -r Telugu/requirements.txt
sudo apt-get update && sudo apt-get install -y espeak-ng
```

MFA is isolated from the FastSpeech 2 Python environment. A Conda environment specification is provided at `Telugu/environment_mfa.yml`. The tested Colab environment used `/content/mfa-env/bin/mfa` (MFA 3.4.3) and `/content/mfa-env/bin/fstcompile`.

## 2. Reuse the existing local data and export MFA WAV/LAB files

The expected source files are:

- `data/manifests/telugu_train.csv`
- `data/manifests/telugu_validation.csv`
- `data/raw/Telugu/train/indictts_XXXXX.wav`
- `data/raw/Telugu/validation/indictts_XXXXX.wav`

Run from the project root:

```bash
python Telugu/preprocess.py \
  --train-manifest data/manifests/telugu_train.csv \
  --validation-manifest data/manifests/telugu_validation.csv \
  --raw-root data/raw/Telugu \
  --output-root data/processed/telugu
```

The source audio is read-only. The script creates 22.05-kHz mono working copies and `.lab` files under `data/processed/telugu/mfa_corpus/` and writes new manifest copies under `data/processed/telugu/manifests/`.

## 3. Build/review the Telugu pronunciation dictionary

```bash
python Telugu/build_lexicon.py
```

Outputs are written to `data/processed/telugu/lexicon/`:

- `telugu_espeak_ipa_candidate.dict`
- `telugu_espeak_ipa_lexicon_audit.csv`
- `telugu_espeak_ipa_manual_review.csv`
- `telugu_espeak_ipa_lexicon_errors.csv`
- `telugu_espeak_ipa_phone_inventory.json`
- `telugu_lexicon_overrides.csv`
- `telugu_mfa_dictionary_reviewed.dict`

The override CSV is for reviewed/manual pronunciations. Zero G2P errors means only that the conversion ran without parser-reported failures. It does not mean that a fluent Telugu reader has verified the output. Review frequent words, long words, numerals, names, code-switching, and standalone modifier tokens (notably `ʲ`).

To include the assignment test text in the lexicon vocabulary without adding it to the training corpus, run:

```bash
python Telugu/build_lexicon.py --include-test-sentences Telugu/test_sentences.json
```

Review the additional entries as well.

## 4. MFA validation/alignment

Copy the recovered MFA model from Drive to:

`Telugu/checkpoints/telugu_acoustic_model_sat2_recovered.zip`

Then set `MFA_BIN` if MFA is not on PATH. Examples:

```bash
export MFA_BIN=/content/mfa-env/bin/mfa
python Telugu/align.py validate
python Telugu/align.py smoke-test
python Telugu/align.py align-train --num-jobs 2
python Telugu/align.py align-validation --num-jobs 2
```

`validate` runs full validation (it does not use `--ignore_acoustics`). `smoke-test` aligns `indictts_00002.wav` by default. `align-train` and `align-validation` write separate TextGrid folders. The validation corpus is never passed to `mfa train`.

The optional `python Telugu/align.py train` command starts a new MFA acoustic-model training run; it is not needed to use the recovered SAT2 package and will refuse to overwrite an existing model path.

## 5. Extract FastSpeech 2 features after full TextGrid alignment

Run only after train and validation TextGrids are present:

```bash
python Telugu/extract_features.py --repo-root .
```

This script uses the reference repository's `audio.stft.TacotronSTFT`, WORLD F0 extraction, phone intervals from the MFA `phones` tier, and a strict `sum(durations) == mel_frames` check. Pitch/energy normalization statistics are fit from training records only and then applied to validation.

It writes `mel/`, `pitch/`, `energy/`, `duration/`, `train.txt`, `val.txt`, `speakers.json`, `stats.json`, and `feature_summary.json` under `data/processed/telugu/fastspeech2_preprocessed/`. It refuses to run if any utterance is missing its TextGrid.

## 6. Patch the separate Telugu symbol vocabulary

The reference implementation's `text/symbols.py` must include the MFA IPA phone tokens. Never patch the English checkout after it has been trained. Use the Telugu checkout/copy only:

```bash
python Telugu/patch_fastspeech2_symbols.py --repo-root .
```

The patch is idempotent and creates a backup of `text/symbols.py` before editing it. Recreate the Telugu model from scratch after changing its input vocabulary; do not load English checkpoints.

## 7. FastSpeech 2 training

`Telugu/train.py` is a guarded launcher around the reference implementation's root-level `train.py`. It validates all feature arrays and metadata first.

Preflight only:

```bash
python Telugu/train.py --repo-root .
```

Launch training only after preflight passes:

```bash
python Telugu/train.py --repo-root . --execute
```

The current `configs/train.yaml` is a 5,000-step pilot configuration (batch size 16, warm-up 1,000). Adjust it based on available compute and the assignment's training requirements.

## 8. Telugu inference

Inference requires a completed Telugu FastSpeech 2 checkpoint, the patched phone vocabulary, the reviewed dictionary and the universal HiFi-GAN checkpoint under the reference repo's `hifigan/` directory.

```bash
python Telugu/inference.py \
  --repo-root . \
  --text "న్యూరోఫార్మాకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు." \
  --speaker female \
  --restore-step 5000
```

The script reports OOV words instead of silently dropping them. Add reviewed pronunciations for the assignment test sentences before inference. To generate all five assignment samples after training, use:

```bash
python Telugu/generate_test_samples.py --repo-root . --restore-step 5000 --speaker female
```

## 9. Loss and variance-adaptor plots

After training has written `Results/Telugu/logs/train/log.txt` and `Results/Telugu/logs/val/log.txt`, create loss curves:

```bash
python Telugu/plot_losses.py --log-root Results/Telugu/logs --output-dir Results/Telugu
```

After a Telugu checkpoint is available, plot the predicted duration, pitch and energy for the five sentences:

```bash
python Telugu/plot_variance.py --repo-root . --restore-step 5000 --speaker female
```

The variance plotting script reads the reference FastSpeech 2 output tuple, converts the model's log-duration predictions to rounded frame durations, and inverse-normalizes pitch/energy using the training-only statistics in `stats.json`. These scripts are included for the post-training evaluation stage and have not been executed because Telugu FastSpeech 2 training has not yet happened.

## Data and checkpoint publishing policy

Do not add `data/raw/Telugu`, processed WAVs, TextGrids, large `.npy` feature directories, or dataset manifests containing source data to a public repository unless the dataset license and assignment rules explicitly allow redistribution. Add only source code, compact config files, small reports/plots, and links/checksums for large checkpoints. Use Git LFS or an accessible external download link for model archives.
