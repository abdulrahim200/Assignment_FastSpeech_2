# English FastSpeech 2 (Svarah)

These files are extracted/adapted from the uploaded assignment notebook. This is a language-specific wrapper around the reference FastSpeech2 implementation, not a standalone reimplementation.

## Included
- `preprocess.py`: converts the notebook's extracted NPZ features into the reference repo's `mel/`, `pitch/`, `energy/`, `duration/` arrays and `train.txt`/`val.txt`.
- `train.py`: launches the reference implementation's `train.py`.
- `inference.py`: launches `synthesize.py` and applies the notebook's two fixes:
  - PyTorch 2.6+ trusted-checkpoint loading (`weights_only=False`).
  - Hybrid MFA/CMU dictionary parsing (skip four numeric metadata columns where present).
- `config.yaml`: English/Svarah paths and pilot settings.
- `configs/preprocess.yaml`: preprocessing configuration.

## Important missing inputs
The notebook's complete preparation process also includes dataset selection, transcript normalization, MFA dictionary preparation, MFA alignment, and mel/pitch/energy/duration extraction. This compact `preprocess.py` starts **after** those stages, using their `.npz` outputs. Those inputs, the original FastSpeech2 source repository, HiFi-GAN weights, and trained checkpoints are not embedded here.

## Expected project arrangement
```text
project/
├── English/                 # this folder
├── FastSpeech2/              # clone the same reference implementation used in the notebook
├── data/
│   ├── manifests/
│   ├── mfa/
│   └── processed/english/
└── Results/English/
```
Update `upstream_repo` and data paths in `config.yaml` if your folders differ.

## Finish the reference configs
Copy these files from the reference repository:
- `FastSpeech2/config/LJSpeech/model.yaml` → `English/configs/model.yaml`
- `FastSpeech2/config/LJSpeech/train.yaml` → `English/configs/train.yaml`

In `configs/model.yaml`, keep the LJSpeech architecture and set:
```yaml
multi_speaker: false
max_seq_len: 1000
```

In `configs/train.yaml`, keep the reference settings except:
```yaml
optimizer:
  warm_up_step: 1000
  anneal_steps: []
step:
  total_step: 5000
  log_step: 100
  synth_step: 2000
  val_step: 1000
  save_step: 1000
```
Also set `path.ckpt_path`, `path.log_path`, and `path.result_path` to the English output folders. The 5,000-step values are the notebook's pilot settings; they do not establish convergence.

## Run
Install dependencies:
```bash
pip install -r requirements.txt
```
Convert the NPZ features:
```bash
python preprocess.py --config config.yaml
```
Train:
```bash
python train.py --config config.yaml
```
Resume:
```bash
python train.py --config config.yaml --restore-step 5000
```
Synthesize:
```bash
python inference.py --config config.yaml --restore-step 5000 --text "Hello, welcome to the world of artificial intelligence."
```

## GitHub
Keep the empty `checkpoints/` directory with `.gitkeep`. Model checkpoints are large; use Git LFS or an external download link as allowed by the assignment. Do not commit raw datasets or private assets.
