import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def read_subset(root: Path, subset_file: Path):
    rels = [ln.strip() for ln in subset_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    paths = [(root / r) for r in rels]
    paths = [p for p in paths if p.exists()]
    rel_out = [str(Path(r).with_suffix(".jpg")) for r in rels if (root / r).exists()]
    return paths, rel_out


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)
    return img


def process_one(src_path: Path, dst_path: Path, size: int, quality: int):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        im = center_crop_resize(im, size)
        im.save(dst_path, format="JPEG", quality=quality, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--subset-file", type=str, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 8)))
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    src_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.subset_file:
        subset_path = Path(args.subset_file).resolve()
        src_paths, rel_out = read_subset(src_root, subset_path)
    else:
        src_paths = list_images(src_root)
        rel_out = [str(p.relative_to(src_root).with_suffix(".jpg")) for p in src_paths]

    if args.max_images is not None:
        src_paths = src_paths[: args.max_images]
        rel_out = rel_out[: args.max_images]

    total = len(src_paths)
    if total == 0:
        raise RuntimeError("No images found to cache.")

    index_path = out_root / f"index_{total}.txt"
    index_path.write_text("\n".join(rel_out) + "\n", encoding="utf-8")

    print("Source root:", src_root)
    print("Cache root :", out_root)
    print("Images     :", total)
    print("Workers    :", args.workers)
    print("Size       :", args.image_size)
    print("Index file :", index_path)

    ok = 0
    bad = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for src, rel in zip(src_paths, rel_out):
            dst = out_root / rel
            futures.append(ex.submit(process_one, src, dst, args.image_size, args.quality))

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
                ok += 1
            except Exception as e:
                bad += 1
                if bad <= 20:
                    print("FAIL:", repr(e))

            if i % 500 == 0 or i == total:
                print(f"Progress: {i}/{total} (ok={ok}, bad={bad})")

    print("Done caching.")
    print("OK :", ok)
    print("BAD:", bad)
    print("Cache root:", out_root)
    print("Index file:", index_path)


if __name__ == "__main__":
    main()