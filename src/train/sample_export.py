import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid
from PIL import Image


def load_generator(ckpt_path: str, z_dim: int, g_ch: int, device: torch.device):
    # Import here so this script can live anywhere in the repo
    from src.models.dcgan import Generator

    G = Generator(z_dim=z_dim, base_ch=g_ch).to(device)
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state)
    G.eval()
    return G


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    img_tensor: (3,H,W) in [-1,1]
    Returns PIL Image in RGB uint8.
    """
    x = img_tensor.detach().cpu()
    x = (x.clamp(-1, 1) + 1) * 0.5  # [-1,1] -> [0,1]
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray(x, mode="RGB")


def upscale_pil(im: Image.Image, target: int, method: str) -> Image.Image:
    if target <= 0:
        return im
    resample = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }.get(method.lower(), Image.LANCZOS)
    return im.resize((target, target), resample=resample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--g-ch", type=int, default=64)

    ap.add_argument("--n", type=int, default=64, help="number of images to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Individual PNGs
    ap.add_argument("--save-individual", action="store_true")
    ap.add_argument("--individual-upscale", type=int, default=0, help="e.g. 512 to upscale each image to 512x512")
    ap.add_argument("--individual-upscale-method", type=str, default="lanczos",
                    choices=["nearest", "bilinear", "bicubic", "lanczos"])

    # Grid PNG
    ap.add_argument("--save-grid", action="store_true")
    ap.add_argument("--grid-nrow", type=int, default=8)
    ap.add_argument("--grid-upscale", type=int, default=0,
                    help="target size for final grid image (square), e.g. 2048 to make 2048x2048")
    ap.add_argument("--grid-upscale-method", type=str, default="lanczos",
                    choices=["nearest", "bilinear", "bicubic", "lanczos"])
    ap.add_argument("--grid-pad", type=int, default=2)

    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    G = load_generator(args.ckpt, args.z_dim, args.g_ch, device)

    with torch.no_grad():
        z = torch.randn(args.n, args.z_dim, 1, 1, device=device)
        fake = G(z)  # (N,3,H,W) in [-1,1]

    # Save individual PNGs (optionally upscaled)
    if args.save_individual:
        indiv_dir = out_dir / "individual"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for i in range(fake.size(0)):
            im = tensor_to_pil(fake[i])
            im = upscale_pil(im, args.individual_upscale, args.individual_upscale_method)
            im.save(indiv_dir / f"sample_{i:04d}.png", optimize=True)

    # Save grid (optionally upscaled to exact target size)
    if args.save_grid:
        grid = make_grid(
            fake,
            nrow=args.grid_nrow,
            padding=args.grid_pad,
            normalize=True,
            value_range=(-1, 1),
        )
        # Save a base grid first
        grid_path = out_dir / "grid.png"
        save_image(grid, grid_path)

        # If requested, upscale the entire grid to a square (e.g. 2048x2048)
        if args.grid_upscale and args.grid_upscale > 0:
            im = Image.open(grid_path).convert("RGB")
            im = upscale_pil(im, args.grid_upscale, args.grid_upscale_method)
            up_path = out_dir / f"grid_{args.grid_upscale}.png"
            im.save(up_path, optimize=True)

    print("Done. Output:", str(out_dir))


if __name__ == "__main__":
    main()