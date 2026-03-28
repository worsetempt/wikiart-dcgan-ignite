# WikiArt DCGAN (PyTorch)

This project trains a DCGAN to generate 64×64 artwork images from the WikiArt dataset.

---

## Overview

* DCGAN (Generator + Discriminator)
* Mixed precision training (AMP)
* Checkpointing, logging, and sample generation
* FID and Inception Score evaluation
* Dataset caching for faster training

---

## Dataset

Dataset: https://www.kaggle.com/datasets/steubk/wikiart

### Setup

1. Download the dataset
2. Place it at:

```
data/wikiart/
```

The `data/` directory is not tracked by Git.

---

## Preprocess (recommended)

Cache and resize images to 64×64:

```
python scripts/cache_wikiart_64.py \
    --data-root data/wikiart \
    --out-root data/cache/wikiart_64 \
    --image-size 64
```

This creates a processed dataset and index file for faster loading 

---

## Training

```
python scripts/train_dcgan.py \
    --data-root data/cache/wikiart_64 \
    --run-name experiment_1 \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.00015
```

Training:

* uses BCE loss with label smoothing 
* logs metrics to CSV
* saves checkpoints and sample images

---

## Outputs

```
outputs/<run-name>/
```

Structure:

```
checkpoints/
samples/
metrics/
hparams.json
```

Hyperparameters are saved automatically 

---

## Sampling

Generate a grid:

```
python scripts/sample_4x4.py
```

Or export images:

```
python scripts/sample_export.py \
    --ckpt outputs/<run-name>/checkpoints/generator_last.pt \
    --out-dir outputs/<run-name>/generated \
    --n 64 \
    --save-grid \
    --save-individual
```

---

## Evaluation

Compute FID and Inception Score:

```
python scripts/eval_gan_metrics.py \
    --real-root data/cache/wikiart_64 \
    --ckpt outputs/<run-name>/checkpoints/generator_last.pt
```

---

## Model

DCGAN architecture:

* Generator: transposed convolutions + batch norm + ReLU
* Discriminator: convolutions + batch norm + LeakyReLU 

---

## Notes

* Do not commit `data/`, `outputs/`, or checkpoints
* Use GPU for training
* Dataset caching significantly improves performance

---
