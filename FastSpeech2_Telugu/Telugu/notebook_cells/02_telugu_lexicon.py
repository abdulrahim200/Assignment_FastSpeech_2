# Extracted from Telugu_FastSpeech2_Notebook.ipynb cells 33–35.
# These cells depend on train_export_df, val_export_df, TELUGU_LEXICON_DIR,
# normalize_telugu_text, and the paths initialized earlier in the notebook.

# 8D.1 Generate a candidate lexicon from the existing Telugu train + validation text
import json
import re as std_re
import unicodedata
from collections import Counter
import regex as unicode_regex
import pandas as pd
from phonemizer import phonemize
from phonemizer.separator import Separator

assert EXPORT_AUDIO_NOW, "Run the existing audio export cell first."
assert len(train_export_df) == 1544, f"Expected 1544 train rows, got {len(train_export_df)}"
assert len(val_export_df) == 313, f"Expected 313 validation rows, got {len(val_export_df)}"

PHONE_SEPARATOR = "~"
LEXICON_LANGUAGE = "te"


def tokenize_telugu_alignment_text(text):
    """Tokenize orthographic words; retain Telugu letters, marks, digits and internal apostrophes."""
    text = unicodedata.normalize("NFC", normalize_telugu_text(text))
    tokens = unicode_regex.findall(
        r"[\p{L}\p{M}\p{N}]+(?:[’'][\p{L}\p{M}\p{N}]+)*",
        text,
    )
    normalized = []
    for token in tokens:
        token = token.replace("’", "'")
        if token.isascii():
            token = token.lower()
        normalized.append(token)
    return normalized


train_word_counts = Counter()
val_word_counts = Counter()
for text in train_export_df["normalized_text"].astype(str):
    train_word_counts.update(tokenize_telugu_alignment_text(text))
for text in val_export_df["normalized_text"].astype(str):
    val_word_counts.update(tokenize_telugu_alignment_text(text))

all_lexicon_words = sorted(set(train_word_counts) | set(val_word_counts))
validation_only_words = sorted(set(val_word_counts) - set(train_word_counts))

if not all_lexicon_words:
    raise RuntimeError("No orthographic words were extracted from the Telugu manifests.")

print("Unique training words:", len(train_word_counts))
print("Unique validation words:", len(val_word_counts))
print("Validation-only words:", len(validation_only_words))
print("Total unique dictionary words:", len(all_lexicon_words))

# eSpeak's phone separator is used to preserve the backend's phoneme boundaries.
# `tie` is intentionally left at its default False because tie and phone separator
# are incompatible in phonemizer.
phoneme_strings = phonemize(
    all_lexicon_words,
    language=LEXICON_LANGUAGE,
    backend="espeak",
    separator=Separator(phone=PHONE_SEPARATOR, word=" "),
    strip=True,
    preserve_punctuation=False,
    with_stress=False,
    language_switch="keep-flags",
    words_mismatch="warn",
    njobs=1,
)

if len(phoneme_strings) != len(all_lexicon_words):
    raise RuntimeError(
        f"eSpeak returned {len(phoneme_strings)} pronunciations for "
        f"{len(all_lexicon_words)} words. Stop rather than misalign dictionary rows."
    )

lexicon_rows = []
lexicon_errors = []
phone_inventory = set()
language_flag_pattern = std_re.compile(r"\([a-z]{2,3}(?:-[a-z0-9]+)?\)", std_re.IGNORECASE)

for word, raw_output in zip(all_lexicon_words, phoneme_strings):
    raw_output = str(raw_output).strip()

    # Do not silently keep words that eSpeak marked as coming from another language.
    if language_flag_pattern.search(raw_output):
        lexicon_errors.append({
            "word": word,
            "raw_espeak_output": raw_output,
            "reason": "language-switch flag; review and add a pronunciation manually",
        })
        continue

    # Each input is one orthographic word. A remaining word separator means that
    # eSpeak emitted an unexpected multiword result; send it for review.
    if " " in raw_output or "\t" in raw_output or "\n" in raw_output:
        lexicon_errors.append({
            "word": word,
            "raw_espeak_output": raw_output,
            "reason": "unexpected multiword / whitespace output",
        })
        continue

    phones = [p for p in raw_output.split(PHONE_SEPARATOR) if p]
    if not phones:
        lexicon_errors.append({
            "word": word,
            "raw_espeak_output": raw_output,
            "reason": "empty phone sequence",
        })
        continue

    if any(any(ch.isspace() for ch in phone) for phone in phones):
        lexicon_errors.append({
            "word": word,
            "raw_espeak_output": raw_output,
            "reason": "phone token contains whitespace",
        })
        continue

    phone_string = " ".join(phones)
    phone_inventory.update(phones)
    lexicon_rows.append({
        "word": word,
        "phones": phone_string,
        "phone_count": len(phones),
        "train_occurrences": int(train_word_counts.get(word, 0)),
        "validation_occurrences": int(val_word_counts.get(word, 0)),
        "seen_in_train": word in train_word_counts,
        "seen_in_validation": word in val_word_counts,
        "source": "eSpeak NG Telugu voice (te); candidate only",
    })

lexicon_df = pd.DataFrame(lexicon_rows)
lexicon_error_df = pd.DataFrame(lexicon_errors)

TELUGU_LEXICON_DIR = TELUGU_PROCESSED_DIR / "lexicon"
TELUGU_LEXICON_DIR.mkdir(parents=True, exist_ok=True)

TELUGU_MFA_DICTIONARY_CANDIDATE = TELUGU_LEXICON_DIR / "telugu_espeak_ipa_candidate.dict"
LEXICON_AUDIT_CSV = TELUGU_LEXICON_DIR / "telugu_espeak_ipa_lexicon_audit.csv"
LEXICON_REVIEW_CSV = TELUGU_LEXICON_DIR / "telugu_espeak_ipa_manual_review.csv"
LEXICON_ERRORS_CSV = TELUGU_LEXICON_DIR / "telugu_espeak_ipa_lexicon_errors.csv"
PHONE_INVENTORY_JSON = TELUGU_LEXICON_DIR / "telugu_espeak_ipa_phone_inventory.json"

# MFA dictionary format is: orthographic_word<TAB>phone1 phone2 phone3 ...
with TELUGU_MFA_DICTIONARY_CANDIDATE.open("w", encoding="utf-8", newline="\n") as f:
    for row in lexicon_rows:
        f.write(f"{row['word']}\t{row['phones']}\n")

lexicon_df.to_csv(LEXICON_AUDIT_CSV, index=False, encoding="utf-8-sig")
lexicon_error_df.to_csv(LEXICON_ERRORS_CSV, index=False, encoding="utf-8-sig")

# Create an editable review sheet covering every candidate word. Start with
# blank approval fields; no pronunciation is marked approved automatically.
review_df = lexicon_df.copy()
review_df["review_status"] = ""
review_df["corrected_phones"] = ""
review_df["review_notes"] = ""
review_df = review_df.sort_values(
    ["train_occurrences", "validation_occurrences", "word"],
    ascending=[False, False, True],
)
review_df.to_csv(LEXICON_REVIEW_CSV, index=False, encoding="utf-8-sig")

PHONE_INVENTORY_JSON.write_text(
    json.dumps(sorted(phone_inventory), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nCandidate dictionary entries:", len(lexicon_df))
print("Words needing review / G2P errors:", len(lexicon_error_df))
print("Candidate phone inventory size:", len(phone_inventory))
print("Candidate dictionary:", TELUGU_MFA_DICTIONARY_CANDIDATE)
print("Full lexicon audit:", LEXICON_AUDIT_CSV)
print("Editable review sheet:", LEXICON_REVIEW_CSV)
print("Error report:", LEXICON_ERRORS_CSV)
print("Phone inventory:", PHONE_INVENTORY_JSON)

if len(lexicon_df) != len(all_lexicon_words):
    print("WARNING: Some words were deliberately omitted from the candidate dictionary. Review the errors CSV and add pronunciations before MFA.")
else:
    print("Every unique transcript word received a candidate entry; this does NOT mean the pronunciations are verified.")

# Keep this False until a fluent Telugu reader has inspected the review sheet,
# the error report, code-switched words, numerals, proper nouns and loan words.
MFA_TELUGU_LEXICON_REVIEWED = False

# ---- CELL 34 ----
# 8D.2 Inspect the candidate phone inventory and highest-frequency pronunciations

if len(lexicon_df):
    display(lexicon_df.sort_values("train_occurrences", ascending=False).head(40))

    # Longest pronunciations often expose unexpected tokenization and loan words.
    long_examples = lexicon_df.sort_values(
        ["phone_count", "word"], ascending=[False, True]
    ).head(25)
    display(long_examples)

if len(lexicon_error_df):
    print("Words withheld from the candidate dictionary; inspect these first:")
    display(lexicon_error_df.head(100))
else:
    print("No automatic G2P errors were detected. Manual pronunciation review is still required.")

print("Candidate phone inventory:")
print(sorted(phone_inventory))
print("\nReview file (one row per candidate word):", LEXICON_REVIEW_CSV)
print("Review all common words and flagged items before setting MFA_TELUGU_LEXICON_REVIEWED=True.")

# ---- CELL 35 ----
# 8D.3 Apply manual overrides and prepare the dictionary that MFA will actually use

from pathlib import Path
import pandas as pd
import unicodedata
import json

MFA_TELUGU_DICTIONARY_REVIEWED = TELUGU_LEXICON_DIR / "telugu_mfa_dictionary_reviewed.dict"
LEXICON_OVERRIDES_CSV = TELUGU_LEXICON_DIR / "telugu_lexicon_overrides.csv"

# This template is optional: keep it empty if the candidate pronunciations need
# no corrections. Add a row for every word to fix or every G2P error to supply.
if not LEXICON_OVERRIDES_CSV.exists():
    pd.DataFrame(columns=["word", "corrected_phones", "notes"]).to_csv(
        LEXICON_OVERRIDES_CSV, index=False, encoding="utf-8-sig"
    )

candidate_map = {row["word"]: row["phones"] for row in lexicon_rows}
override_df = pd.read_csv(LEXICON_OVERRIDES_CSV, keep_default_na=False)
required_columns = {"word", "corrected_phones"}
if not required_columns.issubset(override_df.columns):
    raise ValueError(f"Override CSV must contain columns: {sorted(required_columns)}")

seen_override_words = set()
for row in override_df.to_dict("records"):
    word = unicodedata.normalize("NFC", str(row.get("word", "")).strip())
    corrected = str(row.get("corrected_phones", "")).strip()

    if not word and not corrected:
        continue
    if not word or not corrected:
        raise ValueError(f"Incomplete override row: {row}")
    if any(ch.isspace() for ch in word):
        raise ValueError(f"Expected one orthographic word, not a phrase: {word!r}")
    if word in seen_override_words:
        raise ValueError(f"Duplicate manual override for {word!r}")

    phone_tokens = corrected.split()
    if not phone_tokens:
        raise ValueError(f"Empty phone sequence for {word!r}")
    if any(any(ch.isspace() for ch in phone) for phone in phone_tokens):
        raise ValueError(f"Invalid phone token in override for {word!r}")

    seen_override_words.add(word)
    candidate_map[word] = " ".join(phone_tokens)

# Words that the G2P parser withheld must be covered by explicit overrides.
missing_error_overrides = (
    sorted(set(lexicon_error_df["word"].tolist()) - seen_override_words)
    if len(lexicon_error_df) else []
)
LEXICON_ERRORS_RESOLVED = len(missing_error_overrides) == 0

with MFA_TELUGU_DICTIONARY_REVIEWED.open("w", encoding="utf-8", newline="\n") as f:
    for word in sorted(candidate_map):
        f.write(f"{word}\t{candidate_map[word]}\n")

final_phone_inventory = sorted({
    phone for pronunciation in candidate_map.values()
    for phone in pronunciation.split()
})
PHONE_INVENTORY_JSON.write_text(
    json.dumps(final_phone_inventory, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Final MFA dictionary entries:", len(candidate_map))
print("Manual override rows applied:", len(seen_override_words))
print("G2P words still missing a manual pronunciation:", len(missing_error_overrides))
if missing_error_overrides:
    print("First missing words:", missing_error_overrides[:50])
print("Edit this file to correct pronunciations:", LEXICON_OVERRIDES_CSV)
print("Dictionary that MFA will use:", MFA_TELUGU_DICTIONARY_REVIEWED)

# Leave False until you have checked common and long words plus all flagged items.
# After review, change this line to True and rerun this cell.
MFA_TELUGU_LEXICON_REVIEWED = False
