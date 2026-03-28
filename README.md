# WikiArt DCGAN (PyTorch) + FID/IS (Ignite)

## 0) Dataset
Place extracted dataset under:
- data/raw/wikiart

Images can be in nested folders; code scans recursively for jpg/jpeg/png/webp.

## 1) Environment
Activate your conda env and install deps:
- pip install -U pytorch-ignite matplotlib numpy pillow tqdm tensorboard scipy
Install torch/torchvision separately if needed.

## 2) Make a small sample list (fast test)
Creates data/sample/index_2000.txt with 2000 image paths (relative to data/raw/wikiart):
- python scripts/make_subset_index.py --data-root data/raw/wikiart --out data/sample/index_2000.txt --n 2000 --seed 0

## 3) Train on the small sample (sanity check)
- python -m src.train.train_dcgan --data-root data/raw/wikiart --subset-file data/sample/index_2000.txt --run-name sample_2000 --epochs 3

Outputs:
- outputs/samples/<run-name>/epoch_*.png  (generated grids)
- outputs/checkpoints/<run-name>/*.pt      (weights)
- outputs/metrics/<run-name>/losses.csv    (loss curves)
- runs/<run-name>/                         (tensorboard)

TensorBoard:
- tensorboard --logdir runs

## 4) Train on full dataset
- python -m src.train.train_dcgan --data-root data/raw/wikiart --run-name full --epochs 25

## 5) Evaluate FID + Inception Score (Ignite)
This evaluates generated samples vs real images from the dataset.

Example (generates 10k fake images internally in batches):
- python -m src.eval.eval_gan_metrics --data-root data/raw/wikiart --ckpt outputs/checkpoints/full/generator_last.pt --run-name full_eval --num-fakes 10000 --batch-size 64

Results saved:
- outputs/metrics/<run-name>/fid_is.json