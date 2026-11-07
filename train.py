"""Launcher for the reference FastSpeech2 training script."""
import argparse, shutil, subprocess
from pathlib import Path
import yaml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--restore-step",type=int,default=None)
    a=ap.parse_args()
    here=Path(__file__).resolve().parent
    cfg=yaml.safe_load(Path(a.config).read_text())
    repo=(here/cfg["upstream_repo"]).resolve()
    data=Path(cfg["data"]["preprocessed_dir"]).resolve()
    if not (repo/"train.py").is_file(): raise FileNotFoundError(f"Clone reference FastSpeech2 into {repo}")
    for name in ("train.txt","val.txt","speakers.json","stats.json"):
        src=data/name
        if not src.exists(): raise FileNotFoundError(f"Missing {src}; run preprocess.py first")
        shutil.copy2(src,repo/name)
    for name in ("preprocess.yaml","model.yaml","train.yaml"):
        if not (here/"configs"/name).exists(): raise FileNotFoundError(here/"configs"/name)
    step=a.restore_step if a.restore_step is not None else cfg["training"]["restore_step"]
    cmd=["python","train.py","--restore_step",str(step),
         "--preprocess_config",str(here/"configs/preprocess.yaml"),
         "--model_config",str(here/"configs/model.yaml"),
         "--train_config",str(here/"configs/train.yaml")]
    subprocess.run(cmd,cwd=repo,check=True)
if __name__=="__main__": main()
