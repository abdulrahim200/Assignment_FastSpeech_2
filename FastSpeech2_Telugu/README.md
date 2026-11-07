# FastSpeech 2 — English and Telugu assignment artifacts

This repository section contains the Telugu-specific implementation and an interim technical report. Keep the existing `English/` implementation separate; do not mix language vocabularies, configs, checkpoints, or result directories.

## Telugu status (7 November 2026)

Verified: IndicTTS split reuse (1,544 train / 313 validation; 5.0031 h / 1.0043 h; zero overlap), 1,857/1,857 audio path and WAV-header checks, mono 22,050-Hz WAV/LAB corpus export, candidate eSpeak NG lexicon generation (13,461 entries; 47 phones; manual pronunciation review still required), MFA 3.4.3 structural validation, recovered SAT2 acoustic archive inspection, and one successful `align_one` TextGrid smoke test.

Not yet verified: full train/validation TextGrid alignment, phone-level FastSpeech 2 feature serialization, Telugu FastSpeech 2 training/checkpoint, loss curves, five assignment test audios, and per-sentence duration/pitch/energy variance-adaptor plots. Do not represent these as completed.

## Layout

- `Telugu/` — preprocessing, lexicon, MFA alignment, feature extraction, Telugu frontend, training/inference launchers, plotting scripts, configs and notebook-cell provenance.
- `Results/Telugu/` — expected result folders with placeholders only until actual outputs are generated.
- `Report/Technical_Report.pdf` — interim report grounded in recorded outputs. DOCX is included as an editable copy.

## Start here

Read `Telugu/README.md` for dependencies, input paths, commands and stage prerequisites. The scripts expect the upstream `ming024/FastSpeech2` repository at the project root. The original IndicTTS audio, generated WAV/LAB corpus, TextGrids, feature arrays and MFA checkpoint are intentionally not bundled.

## Data and checkpoint safety

Do not commit source audio, processed speech, TextGrids, large feature arrays, or dataset-derived manifests to a public repository unless the dataset license explicitly permits redistribution. Store large model files using Git LFS or a permitted external host. See `Telugu/checkpoints/README.md`.
