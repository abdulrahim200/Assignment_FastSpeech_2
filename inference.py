"""English synthesis launcher with notebook's dictionary/checkpoint fixes."""
import argparse, re, shutil, subprocess
from pathlib import Path
import yaml

def patch_upstream(repo):
    model=repo/"utils/model.py"
    src=model.read_text()
    old="ckpt = torch.load(ckpt_path)"
    new="ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)"
    if old in src:
        backup=model.with_name("model.py.english-backup")
        if not backup.exists(): shutil.copy2(model,backup)
        model.write_text(src.replace(old,new,1))

    synth=repo/"synthesize.py"
    src=synth.read_text()
    start=src.find("def read_lexicon(lex_path):")
    end=src.find("\\ndef ",start+1)
    if start<0 or end<0: raise RuntimeError("Cannot locate read_lexicon() in synthesize.py")
    backup=synth.with_name("synthesize.py.english-backup")
    if not backup.exists(): shutil.copy2(synth,backup)
    replacement = """def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = line.strip().split()
            if len(temp) < 2:
                continue
            word = temp[0]
            has_mfa_metadata = (
                len(temp) >= 6
                and all(re.fullmatch(r"-?\\\\d+(?:\\\\.\\\\d+)?", token) for token in temp[1:5])
            )
            phones = temp[5:] if has_mfa_metadata else temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon
"""
    synth.write_text(src[:start]+replacement+src[end:])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--restore-step",type=int,required=True)
    ap.add_argument("--text",required=True)
    a=ap.parse_args()
    here=Path(__file__).resolve().parent
    cfg=yaml.safe_load(Path(a.config).read_text())
    repo=(here/cfg["upstream_repo"]).resolve()
    if not (repo/"synthesize.py").exists(): raise FileNotFoundError(f"Reference repo not found: {repo}")
    patch_upstream(repo)
    cmd=["python","synthesize.py","--restore_step",str(a.restore_step),
         "--mode","single","--text",a.text,
         "-p",str(here/"configs/preprocess.yaml"),
         "-m",str(here/"configs/model.yaml"),
         "-t",str(here/"configs/train.yaml")]
    subprocess.run(cmd,cwd=repo,check=True)
if __name__=="__main__": main()
