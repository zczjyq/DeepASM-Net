# Wavefield Dehazing (Framework A)

Physics-inspired, propagation-structured dehazing with a stackable transformer-like residual block (Framework A / ABlock).

## Setup

Use the provided Python executable:

```
c:/Users/wechyu/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r requirements.txt
```

## Dataset

Folder layout (relative to repo root):

- ./images/images_add_fog/   (hazy images)
- ./images/images_clear/     (clean images)

Matching rule:
- clear_1.jpg  <-> haze_clear_1_1.jpg ... haze_clear_1_8.jpg

Train/val split is done by clean-id so all haze variants of the same clean image stay in the same split.

## Training

```
c:/Users/wechyu/AppData/Local/Programs/Python/Python310/python.exe train.py --config configs/default.yaml
```

Each run creates a timestamped folder in ./runs/<timestamp>/ containing:
- config snapshot
- logs (loss/metrics)
- checkpoints (best.pth, last.pth)
- sample grids per epoch
- code snapshot (exact code used)

## Evaluation

```
c:/Users/wechyu/AppData/Local/Programs/Python/Python310/python.exe eval.py --ckpt <path_to_ckpt> --config configs/default.yaml
```

## Inference

```
c:/Users/wechyu/AppData/Local/Programs/Python/Python310/python.exe infer.py --ckpt <path_to_best.pth> --input ./images/images_add_fog --output_root ./outputs
```

Inference creates ./outputs/<timestamp>/ with dehazed images and an HTML summary.

## Notes

- PhaseModule and ZModule are modular. Edit their depth/width in `configs/default.yaml`.
- The Angular Spectrum Method (ASM) here is a structural prior, not a physically exact scattering simulation.
- If you enable perceptual loss, torchvision VGG weights must be available locally.
