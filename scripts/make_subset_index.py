import argparse
import os
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = [p for p in iter_images(root)]
    if not paths:
        raise RuntimeError(f"No images found under {root}")

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    n = min(args.n, len(paths))
    chosen = paths[:n]

    # Write relative paths so it works on other machines
    rels = [str(p.relative_to(root)).replace("\\", "/") for p in chosen]
    out_path.write_text("\n".join(rels), encoding="utf-8")

    print(f"Wrote {len(rels)} paths to: {out_path}")
    print(f"Dataset root: {root}")
    print("Example:", rels[0])


if __name__ == "__main__":
    main()