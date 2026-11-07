# Extracted from Telugu_FastSpeech2_Notebook.ipynb cells 43–44.
# The original notebook uses RUN_MFA_* flags set False by default.

# 9E.1 MFA validation, acoustic-model training, and held-out validation alignment
from pathlib import Path

MFA_NJOBS = 2
RUN_MFA_VALIDATE = False
RUN_MFA_TRAIN = False
RUN_MFA_ALIGN_VALIDATION = False

MFA_MODEL_DIR = TELUGU_PROCESSED_DIR / "mfa_models"
MFA_ALIGNMENT_DIR = TELUGU_PROCESSED_DIR / "mfa_alignments"
MFA_TRAIN_ALIGNMENT_DIR = MFA_ALIGNMENT_DIR / "train"
MFA_VALIDATION_ALIGNMENT_DIR = MFA_ALIGNMENT_DIR / "validation"
MFA_MODEL_PATH = MFA_MODEL_DIR / "telugu_indictts_espeak_ipa.zip"

MFA_TRAIN_CORPUS = TELUGU_MFA_DIR / "train"
MFA_VALIDATION_CORPUS = TELUGU_MFA_DIR / "validation"

for folder in [MFA_MODEL_DIR, MFA_ALIGNMENT_DIR, MFA_TRAIN_ALIGNMENT_DIR, MFA_VALIDATION_ALIGNMENT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# The original 9C placeholder set this to None. Repoint it to the new candidate
# reviewed dictionary produced in section 8D.3; the human-review gate remains mandatory.
MFA_TELUGU_DICTIONARY = Path(MFA_TELUGU_DICTIONARY_REVIEWED)
MFA_TELUGU_ACOUSTIC_MODEL = MFA_MODEL_PATH if MFA_MODEL_PATH.exists() else None


def run_mfa_cli(arguments, *, check=True):
    """Run MFA inside the isolated micromamba environment."""
    result = mfa_environment_command(["mfa", *map(str, arguments)], check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"MFA command failed with exit code {result.returncode}")
    return result


actions_requested = RUN_MFA_VALIDATE or RUN_MFA_TRAIN or RUN_MFA_ALIGN_VALIDATION
if actions_requested:
    if not MFA_TELUGU_LEXICON_REVIEWED:
        raise RuntimeError(
            "MFA is gated. Review the candidate pronunciations and then set "
            "MFA_TELUGU_LEXICON_REVIEWED=True."
        )
    if not globals().get("LEXICON_ERRORS_RESOLVED", False):
        raise RuntimeError("Some G2P words are still missing. Add manual pronunciations in telugu_lexicon_overrides.csv, rerun 8D.3, and check again.")
    if not MFA_TELUGU_DICTIONARY.is_file():
        raise FileNotFoundError(f"Telugu candidate dictionary not found: {MFA_TELUGU_DICTIONARY}")
    if not MFA_ENV_PREFIX.exists():
        raise RuntimeError("Isolated MFA environment not found. Run cell 9D.1 with INSTALL_MFA_ENV_NOW=True.")

if RUN_MFA_VALIDATE:
    print("Validating the TRAIN corpus and candidate dictionary. Inspect the report before training.")
    run_mfa_cli([
        "validate",
        "--num_jobs", str(MFA_NJOBS),
        "--output_directory", str(MFA_MODEL_DIR / "validation_report"),
        str(MFA_TRAIN_CORPUS),
        str(MFA_TELUGU_DICTIONARY),
    ])

if RUN_MFA_TRAIN:
    if not RUN_MFA_VALIDATE:
        print("Reminder: run and inspect MFA validation before enabling training.")
    if MFA_MODEL_PATH.exists():
        raise FileExistsError(
            f"Model already exists: {MFA_MODEL_PATH}. Rename it deliberately before retraining."
        )
    train_command = [
        "train",
        "--num_jobs", str(MFA_NJOBS),
        "--output_directory", str(MFA_TRAIN_ALIGNMENT_DIR),
        str(MFA_TRAIN_CORPUS),
        str(MFA_TELUGU_DICTIONARY),
        str(MFA_MODEL_PATH),
    ]
    print("Training the acoustic model with TRAIN audio only:")
    run_mfa_cli(train_command)
    MFA_TELUGU_ACOUSTIC_MODEL = MFA_MODEL_PATH
    print("Acoustic model created:", MFA_MODEL_PATH)

if RUN_MFA_ALIGN_VALIDATION:
    if not MFA_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the Telugu MFA acoustic model first: {MFA_MODEL_PATH}")
    validation_command = [
        "align",
        "--output_format", "long_textgrid",
        "--num_jobs", str(MFA_NJOBS),
        str(MFA_VALIDATION_CORPUS),
        str(MFA_TELUGU_DICTIONARY),
        str(MFA_MODEL_PATH),
        str(MFA_VALIDATION_ALIGNMENT_DIR),
    ]
    print("Aligning the held-out VALIDATION corpus:")
    run_mfa_cli(validation_command)
    print("Validation TextGrids written to:", MFA_VALIDATION_ALIGNMENT_DIR)

if not actions_requested:
    print("MFA actions are disabled by default.")
    print("Review lexicon -> install MFA -> validate train corpus -> train acoustic model -> align validation.")

# ---- CELL 44 ----
# 9E.2 Inspect TextGrid coverage and a sample alignment after MFA has run
from pathlib import Path


def find_textgrids(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".textgrid"
    )

train_textgrids = find_textgrids(MFA_TRAIN_ALIGNMENT_DIR)
validation_textgrids = find_textgrids(MFA_VALIDATION_ALIGNMENT_DIR)

print("Training TextGrids:", len(train_textgrids), "/", len(train_export_df))
print("Validation TextGrids:", len(validation_textgrids), "/", len(val_export_df))

if train_textgrids or validation_textgrids:
    from praatio import textgrid as praatio_textgrid
    sample_grid = (train_textgrids or validation_textgrids)[0]
    tg = praatio_textgrid.openTextgrid(str(sample_grid), includeEmptyIntervals=True)
    print("Sample TextGrid:", sample_grid)
    print("Tier names:", tg.tierNames)
    for tier_name in tg.tierNames:
        tier = tg.getTier(tier_name)
        entries = getattr(tier, "entries", [])
        print(f"Tier {tier_name!r}: {len(entries)} intervals")
        for interval in entries[:10]:
            print("  ", interval)
else:
    print("No TextGrids yet. This is expected until MFA acoustic-model training/alignment is complete.")
