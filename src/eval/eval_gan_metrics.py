import argparse
import json
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`TorchScript` support for functional optimizers*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root: Path):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return sorted(paths)


class ImageFolderFlat(Dataset):
    def __init__(self, root: str, max_images: int | None = None):
        self.paths = list_images(Path(root))
        if max_images is not None:
            self.paths = self.paths[: int(max_images)]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            return self.to_tensor(im)  # [0,1]


def try_import_generator():
    try:
        from src.models.dcgan import Generator  # type: ignore
        return Generator
    except Exception:
        pass

    try:
        from dcgan import Generator  # type: ignore
        return Generator
    except Exception as e:
        raise ImportError(
            "Could not import Generator. Expected either `src.models.dcgan.Generator` "
            "or local `dcgan.py` with `Generator`."
        ) from e


@torch.no_grad()
def to_01(x: torch.Tensor) -> torch.Tensor:
    # Real images are [0,1]. Generator is tanh => [-1,1].
    if x.min() < -0.01 or x.max() > 1.01:
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    else:
        x = x.clamp(0.0, 1.0)
    return x


def inception_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(
                (299, 299),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class InceptionPool3(nn.Module):
    """
    Returns:
      logits: (B,1000)
      pool3:  (B,2048) (avgpool output)
    """

    def __init__(self, device: torch.device):
        super().__init__()
        # torchvision version compatibility: pretrained weights can enforce aux_logits=True
        try:
            m = inception_v3(weights="IMAGENET1K_V1", aux_logits=True, transform_input=False)
        except Exception:
            m = inception_v3(pretrained=True, aux_logits=True, transform_input=False)

        m.eval().to(device)
        self.m = m

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        m = self.m

        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)

        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)

        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)

        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)

        x = m.avgpool(x)              # (B,2048,1,1)
        pool3 = torch.flatten(x, 1)   # (B,2048)

        logits = m.fc(m.dropout(pool3))  # (B,1000)
        return logits, pool3


def cov_stats(x: np.ndarray):
    # x: (N,D)
    mu = np.mean(x, axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    return mu, cov


def symmetrize(mat: np.ndarray):
    return 0.5 * (mat + mat.T)


def sqrtm_psd(mat: np.ndarray, eps: float = 1e-6):
    # Symmetrize then PSD sqrt via eigen
    mat = symmetrize(mat)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def fid_from_stats(mu1, sigma1, mu2, sigma2, eps: float = 1e-6):
    # Make sure covariances are symmetric PSD-ish
    sigma1 = symmetrize(sigma1)
    sigma2 = symmetrize(sigma2)

    # Add small jitter for stability
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2

    # Stable sqrt of product via sandwich
    s1_sqrt = sqrtm_psd(sigma1, eps=eps)
    mid = s1_sqrt @ sigma2 @ s1_sqrt
    covmean = sqrtm_psd(mid, eps=eps)

    fid = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))

    # Clamp only if tiny negative due to floating error
    if fid < 0:
        fid = float(max(fid, 0.0))

    return fid


def inception_score_from_probs(probs: np.ndarray, splits: int = 10, eps: float = 1e-16):
    N = probs.shape[0]
    splits = max(1, int(splits))
    split_size = N // splits
    if split_size == 0:
        splits = 1
        split_size = N

    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        kl = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl)))

    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def collect_real(extractor, preprocess, real_root, n, batch_size, num_workers, device, debug=False):
    ds = ImageFolderFlat(real_root, max_images=n)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    need = (n // batch_size) * batch_size
    feats, probs = [], []

    if debug:
        print("First real image paths:")
        for i in range(min(5, len(ds.paths))):
            print("  ", ds.paths[i])

    pbar = tqdm(total=need, desc="Real", unit="img", dynamic_ncols=True)
    got = 0
    for batch in dl:
        if got >= need:
            break
        x = batch.to(device, non_blocking=True)
        x = preprocess(to_01(x))
        logits, pool3 = extractor(x)
        p = F.softmax(logits, dim=1)

        feats.append(pool3.cpu().numpy())
        probs.append(p.cpu().numpy())

        got += x.size(0)
        pbar.update(x.size(0))
    pbar.close()

    feats = np.concatenate(feats, axis=0)[:need]
    probs = np.concatenate(probs, axis=0)[:need]
    return feats, probs, ds


@torch.no_grad()
def collect_fake(extractor, preprocess, G, z_dim, n, batch_size, device, fake_mode="gan", debug=False):
    need = (n // batch_size) * batch_size
    feats, probs = [], []

    pbar = tqdm(total=need, desc="Fake", unit="img", dynamic_ncols=True)
    got = 0
    first_dbg = False

    while got < need:
        cur = min(batch_size, need - got)

        if fake_mode == "noise":
            fake = torch.rand(cur, 3, 64, 64, device=device) * 2.0 - 1.0  # [-1,1]
        else:
            z = torch.randn(cur, z_dim, 1, 1, device=device)
            fake = G(z)

        fake01 = to_01(fake)
        x = preprocess(fake01)
        logits, pool3 = extractor(x)
        p = F.softmax(logits, dim=1)

        if debug and not first_dbg:
            first_dbg = True
            print("DEBUG fake:")
            print("  fake01 min/max:", float(fake01.min()), float(fake01.max()))
            print("  x mean/std:", float(x.mean()), float(x.std()))

        feats.append(pool3.cpu().numpy())
        probs.append(p.cpu().numpy())

        got += cur
        pbar.update(cur)

    pbar.close()

    feats = np.concatenate(feats, axis=0)[:need]
    probs = np.concatenate(probs, axis=0)[:need]
    return feats, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--num-reals", type=int, default=10000)
    ap.add_argument("--num-fakes", type=int, default=10000)

    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--g-ch", type=int, default=64)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--is-splits", type=int, default=10)
    ap.add_argument("--fake-mode", choices=("gan", "noise"), default="gan")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    seed_all(int(args.seed))
    device = torch.device(args.device)

    preprocess = inception_preprocess()
    extractor = InceptionPool3(device=device)

    Generator = try_import_generator()
    G = Generator(z_dim=int(args.z_dim), base_ch=int(args.g_ch)).to(device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    G.load_state_dict(state, strict=True)
    G.eval()

    real_feats, real_probs, _ = collect_real(
        extractor, preprocess, args.real_root,
        int(args.num_reals), int(args.batch_size), int(args.num_workers),
        device, debug=bool(args.debug)
    )
    fake_feats, fake_probs = collect_fake(
        extractor, preprocess, G, int(args.z_dim),
        int(args.num_fakes), int(args.batch_size),
        device, fake_mode=args.fake_mode, debug=bool(args.debug)
    )

    is_mean, is_std = inception_score_from_probs(fake_probs, splits=int(args.is_splits))
    mu_r, sig_r = cov_stats(real_feats)
    mu_f, sig_f = cov_stats(fake_feats)
    fid = fid_from_stats(mu_r, sig_r, mu_f, sig_f)

    used_reals = int(real_feats.shape[0])
    used_fakes = int(fake_feats.shape[0])

    print(f"\nUsed: {used_reals} real, {used_fakes} fake")
    print(f"FID: {fid:.6f}")
    print(f"IS:  {is_mean:.6f} (std {is_std:.6f}, splits={int(args.is_splits)})")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fid": float(fid),
            "inception_score_mean": float(is_mean),
            "inception_score_std": float(is_std),
            "used_reals": used_reals,
            "used_fakes": used_fakes,
            "batch_size": int(args.batch_size),
            "real_root": str(Path(args.real_root).expanduser().resolve()),
            "ckpt": str(Path(args.ckpt).expanduser().resolve()),
            "device": str(device),
            "seed": int(args.seed),
            "z_dim": int(args.z_dim),
            "g_ch": int(args.g_ch),
            "fake_mode": args.fake_mode,
            "is_splits": int(args.is_splits),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Saved:", out_path)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()