# Extracted from Telugu_FastSpeech2_Notebook.ipynb cells 46–48.
# Requires notebook imports/variables: np, sf, plt, SAMPLE_RATE, EXPORT_AUDIO_NOW.

# 10A. Shared acoustic settings
SAMPLE_RATE = 22050
HOP_LENGTH = 256
FILTER_LENGTH = 1024
WIN_LENGTH = 1024
N_MEL_CHANNELS = 80
MEL_FMIN = 0
MEL_FMAX = 8000
F0_MIN = 50
F0_MAX = 600

FRAME_PERIOD_MS = HOP_LENGTH / SAMPLE_RATE * 1000
print("Frame period (ms):", FRAME_PERIOD_MS)
print("Mel channels:", N_MEL_CHANNELS, "FFT:", FILTER_LENGTH, "hop:", HOP_LENGTH)

# 10B. Frame-level mel / pitch / energy extraction for one normalized WAV
import librosa


def fit_length(values, target_len):
    values = np.asarray(values)
    if len(values) > target_len:
        return values[:target_len]
    if len(values) < target_len:
        return np.pad(values, (0, target_len-len(values)), mode="constant")
    return values


def extract_frame_level_features(wav_path):
    waveform, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz WAV, found {sr}: {wav_path}")
    waveform = np.asarray(waveform, dtype=np.float32)

    # Magnitude mel spectrogram, then log compression. Output orientation is (frames, mel_channels).
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=FILTER_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
        power=1.0,
        n_mels=N_MEL_CHANNELS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
    )
    mel = np.log(np.maximum(mel_power, 1e-5)).T.astype(np.float32)

    # WORLD F0. Unvoiced frames are represented by 0 Hz.
    try:
        import pyworld
        f0_raw, time_axis = pyworld.dio(
            waveform.astype(np.float64),
            SAMPLE_RATE,
            f0_floor=F0_MIN,
            f0_ceil=F0_MAX,
            frame_period=FRAME_PERIOD_MS,
        )
        f0 = pyworld.stonemask(waveform.astype(np.float64), f0_raw, time_axis, SAMPLE_RATE).astype(np.float32)
    except Exception as exc:
        raise RuntimeError("WORLD pitch extraction failed. Check pyworld installation and waveform.") from exc

    energy = librosa.feature.rms(
        y=waveform,
        frame_length=FILTER_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )[0].astype(np.float32)

    n_frames = mel.shape[0]
    f0 = fit_length(f0, n_frames).astype(np.float32)
    energy = fit_length(energy, n_frames).astype(np.float32)

    assert mel.shape[1] == N_MEL_CHANNELS
    assert len(f0) == len(energy) == n_frames
    assert np.isfinite(mel).all() and np.isfinite(f0).all() and np.isfinite(energy).all()
    return {"mel": mel, "pitch_frame": f0, "energy_frame": energy}


if EXPORT_AUDIO_NOW:
    feature_demo_path = train_export_df.iloc[0].audio_path
    demo_features = extract_frame_level_features(feature_demo_path)
    print("Demo WAV:", feature_demo_path)
    print("Mel:", demo_features["mel"].shape)
    print("Pitch frames:", demo_features["pitch_frame"].shape)
    print("Energy frames:", demo_features["energy_frame"].shape)
else:
    print("Run audio export before testing frame-level feature extraction.")

# 10C. Quick visual inspection of one clip's mel, F0, and energy
if EXPORT_AUDIO_NOW:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].imshow(demo_features["mel"].T, origin="lower", aspect="auto")
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title("Log-mel spectrogram")
    axes[1].plot(demo_features["pitch_frame"])
    axes[1].set_ylabel("F0 (Hz)")
    axes[1].set_title("Frame-level pitch (0 = unvoiced)")
    axes[2].plot(demo_features["energy_frame"])
    axes[2].set_ylabel("RMS")
    axes[2].set_xlabel("Frame index")
    axes[2].set_title("Frame-level energy")
    plt.tight_layout()
    plt.show()
