# Extracted from Telugu_FastSpeech2_Notebook.ipynb cells 50–54.
# Cell 51 is a synthetic shape-contract demonstration only, NOT training data.

def aggregate_frames_to_phone_level(frame_values, durations, *, voiced_only=False):
    frame_values = np.asarray(frame_values, dtype=np.float32)
    durations = np.asarray(durations, dtype=np.int64)
    if (durations < 0).any():
        raise ValueError("Durations cannot be negative.")
    if int(durations.sum()) != len(frame_values):
        raise ValueError(f"Duration sum {int(durations.sum())} != frame count {len(frame_values)}")

    output = []
    cursor = 0
    for duration in durations:
        segment = frame_values[cursor:cursor + int(duration)]
        cursor += int(duration)
        if len(segment) == 0:
            output.append(0.0)
        elif voiced_only:
            voiced = segment[segment > 0]
            output.append(float(voiced.mean()) if len(voiced) else 0.0)
        else:
            output.append(float(segment.mean()))
    return np.asarray(output, dtype=np.float32)


def validate_fastspeech2_record(phones, durations, mel, pitch, energy):
    phones = np.asarray(phones)
    durations = np.asarray(durations, dtype=np.int64)
    mel = np.asarray(mel)
    pitch = np.asarray(pitch)
    energy = np.asarray(energy)
    errors = []
    if mel.ndim != 2 or mel.shape[1] != N_MEL_CHANNELS:
        errors.append(f"mel must be (frames, {N_MEL_CHANNELS}); got {mel.shape}")
    if len(phones) != len(durations):
        errors.append("phones and durations lengths differ")
    if int(durations.sum()) != len(mel):
        errors.append(f"sum(durations)={int(durations.sum())} but mel frames={len(mel)}")
    if len(pitch) != len(phones):
        errors.append("phone-level pitch length differs from phones")
    if len(energy) != len(phones):
        errors.append("phone-level energy length differs from phones")
    for name, arr in [("mel", mel), ("pitch", pitch), ("energy", energy)]:
        if not np.isfinite(arr).all():
            errors.append(f"{name} contains NaN/Inf")
    if errors:
        raise ValueError("Invalid FastSpeech 2 record:\n- " + "\n- ".join(errors))
    return True


print("Aggregation and validation helpers are ready.")

# ---- CELL 51 (synthetic demo only) ----
# 11B. Example using a placeholder alignment result (demonstrates the length contract only)
# This synthetic demonstration is NOT training data and must never be saved as a real example.
example_durations = np.array([3, 5, 4], dtype=np.int64)
example_frame_pitch = np.array([0, 120, 125, 130, 0, 140, 142, 0, 0, 0, 155, 158], dtype=np.float32)
example_frame_energy = np.linspace(0.1, 0.9, int(example_durations.sum()), dtype=np.float32)
print("Duration sum:", int(example_durations.sum()))
print("Phone-level pitch:", aggregate_frames_to_phone_level(example_frame_pitch, example_durations, voiced_only=True))
print("Phone-level energy:", aggregate_frames_to_phone_level(example_frame_energy, example_durations))

# ---- CELL 54 ----
# 12B. FastSpeech 2 record writer — call only after you have real aligned phone IDs/durations.
def save_fastspeech2_record(utterance_id, phone_ids, durations, mel_frames_time_major,
                            frame_pitch, frame_energy, output_root=TELUGU_FEATURE_DIR):
    """Save one aligned record. phone_ids/durations must come from the audited Telugu frontend+aligner."""
    mel = np.asarray(mel_frames_time_major, dtype=np.float32)
    durations = np.asarray(durations, dtype=np.int64)
    pitch_phone = aggregate_frames_to_phone_level(frame_pitch, durations, voiced_only=True)
    energy_phone = aggregate_frames_to_phone_level(frame_energy, durations, voiced_only=False)
    validate_fastspeech2_record(phone_ids, durations, mel, pitch_phone, energy_phone)

    output_root = Path(output_root)
    for subdir in ["mel", "pitch", "energy", "duration"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    np.save(output_root / "mel" / f"{utterance_id}.npy", mel)
    np.save(output_root / "pitch" / f"{utterance_id}.npy", pitch_phone)
    np.save(output_root / "energy" / f"{utterance_id}.npy", energy_phone)
    np.save(output_root / "duration" / f"{utterance_id}.npy", durations)
    return {
        "utterance_id": utterance_id,
        "phones": len(phone_ids),
        "mel_frames": len(mel),
        "duration_sum": int(durations.sum()),
    }

print("Record writer defined. It intentionally requires externally verified alignment outputs.")
