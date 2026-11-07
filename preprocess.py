"""Convert notebook-produced Svarah NPZ features to FastSpeech2's file layout."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def remove_outlier(x):
    x = np.asarray(x)
    if not len(x): return x
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    return x[(x > q25 - 1.5 * iqr) & (x < q75 + 1.5 * iqr)]

def records(manifest, folder):
    df = pd.read_csv(manifest)
    found = []
    for _, row in df.iterrows():
        p = Path(folder) / f"svarah_{int(row['dataset_index']):05d}.npz"
        if p.exists(): found.append((row, p))
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d = cfg["data"]; out = Path(d["preprocessed_dir"])
    train = records(d["train_manifest"], d["train_npz_dir"])
    valid = records(d["validation_manifest"], d["validation_npz_dir"])
    print("Matched train:", len(train), "| validation:", len(valid))
    if not train or not valid: raise RuntimeError("No NPZ files found; check config paths.")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"{out} is not empty; back it up or use --force.")
    for folder in ("mel", "pitch", "energy", "duration"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    ps, es = StandardScaler(), StandardScaler()
    for _, p in tqdm(train, desc="Fit training-only normalization"):
        with np.load(p, allow_pickle=False) as z:
            a, b = remove_outlier(z["pitch"]), remove_outlier(z["energy"])
        if len(a): ps.partial_fit(a.reshape(-1, 1))
        if len(b): es.partial_fit(b.reshape(-1, 1))
    pm, pst = float(ps.mean_[0]), float(ps.scale_[0])
    em, est = float(es.mean_[0]), float(es.scale_[0])
    speaker = cfg.get("speaker", "speaker_01")

    def convert(rows, split):
        meta = []
        for row, p in tqdm(rows, desc=f"Convert {split}"):
            with np.load(p, allow_pickle=False) as z:
                phones, dur = z["phones"], z["durations"].astype(np.int64)
                mel = z["mel"].astype(np.float32).T
                pitch, energy = z["pitch"].astype(np.float64), z["energy"].astype(np.float64)
            tokens = [x.decode() if isinstance(x, bytes) else str(x) for x in phones]
            if any(not x.startswith("@") for x in tokens):
                raise ValueError(f"Phone without @ prefix in {p.name}")
            tokens = [x[1:] for x in tokens]
            if not (len(tokens) == len(dur) == len(pitch) == len(energy)):
                raise ValueError(f"Phone/variance length mismatch: {p.name}")
            if mel.ndim != 2 or mel.shape[1] != 80 or int(dur.sum()) != mel.shape[0]:
                raise ValueError(f"Mel/duration mismatch: {p.name}")
            if np.any(dur < 0) or not all(np.isfinite(x).all() for x in (mel,pitch,energy)):
                raise ValueError(f"Invalid feature values: {p.name}")
            pitch = ((pitch-pm)/pst).astype(np.float32)
            energy = ((energy-em)/est).astype(np.float32)
            name = p.stem
            np.save(out/"mel"/f"{speaker}-mel-{name}.npy", mel)
            np.save(out/"pitch"/f"{speaker}-pitch-{name}.npy", pitch)
            np.save(out/"energy"/f"{speaker}-energy-{name}.npy", energy)
            np.save(out/"duration"/f"{speaker}-duration-{name}.npy", dur)
            raw = " ".join(str(row["final_transcript"]).split()).replace("|", " ")
            meta.append(f"{name}|{speaker}|{{{' '.join(tokens)}}}|{raw}")
        return meta

    train_meta, val_meta = convert(train, "train"), convert(valid, "validation")
    (out/"train.txt").write_text("\\n".join(train_meta)+"\\n", encoding="utf-8")
    (out/"val.txt").write_text("\\n".join(val_meta)+"\\n", encoding="utf-8")
    (out/"speakers.json").write_text(json.dumps({speaker:0}, indent=4))
    def mm(folder):
        vals = [np.load(p) for p in (out/folder).glob("*.npy")]
        return min(float(x.min()) for x in vals), max(float(x.max()) for x in vals)
    pmin,pmax=mm("pitch"); emin,emax=mm("energy")
    stats={"pitch":[pmin,pmax,pm,pst],"energy":[emin,emax,em,est]}
    (out/"stats.json").write_text(json.dumps(stats,indent=4))
    print(f"Done: {len(train_meta)} train, {len(val_meta)} validation samples at {out}")

if __name__ == "__main__": main()
