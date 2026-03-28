import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.wikiart import WikiArtImages
from src.models.dcgan import Generator, Discriminator, weights_init
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.utils.image import save_sample_grid
from src.utils.plots import plot_losses_csv


def make_noise(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, 1, 1, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--subset-file", type=str, default=None)
    ap.add_argument("--run-name", type=str, default="run")

    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--g-ch", type=int, default=64)
    ap.add_argument("--d-ch", type=int, default=64)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=100)

    # stability knobs
    ap.add_argument("--real-label", type=float, default=0.9)  # one-sided label smoothing
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    project_root = Path(".").resolve()
    out_root = ensure_dir(project_root / "outputs")

    run_dir = ensure_dir(out_root / args.run_name)
    samples_dir = ensure_dir(run_dir / "samples")
    metrics_dir = ensure_dir(run_dir / "metrics")
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    hparams = vars(args).copy()
    hparams["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(run_dir / "hparams.json", hparams)

    ds = WikiArtImages(
        root=args.data_root,
        image_size=args.image_size,
        subset_file=args.subset_file,
    )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2  # good default on Windows

    dl = DataLoader(ds, **dl_kwargs)

    G = Generator(z_dim=args.z_dim, base_ch=args.g_ch).to(device)
    D = Discriminator(base_ch=args.d_ch).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    if args.compile and device.type == "cuda":
        try:
            G = torch.compile(G)
            D = torch.compile(D)
        except Exception:
            pass

    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler("cuda", enabled=use_amp)

    fixed_z = make_noise(64, args.z_dim, device)

    losses_csv = metrics_dir / "losses.csv"
    loss_f = losses_csv.open("w", newline="", encoding="utf-8")
    loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "epoch", "loss_d", "loss_g", "dx", "dgz1", "dgz2"])
    loss_writer.writeheader()

    step = 0
    start_time = time.time()

    print("Saving to:", run_dir)
    print("Dataset length:", len(ds))
    print("Batches per epoch:", len(dl))
    print("Device:", device)
    print("num_workers:", args.num_workers, "persistent_workers:", (args.num_workers > 0))
    print("AMP:", use_amp, "compile:", bool(args.compile and device.type == "cuda"))
    print("real_label:", args.real_label)

    try:
        for epoch in range(1, args.epochs + 1):
            G.train()
            D.train()

            running_d = 0.0
            running_g = 0.0
            running_dx = 0.0
            running_dgz2 = 0.0
            n_batches = 0

            pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

            for real in pbar:
                real = real.to(device, non_blocking=True)
                bsz = real.size(0)

                # one-sided label smoothing for real
                ones = torch.full((bsz,), float(args.real_label), device=device)
                zeros = torch.zeros(bsz, device=device)

                # ---- Train D ----
                optD.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=use_amp):
                    out_real = D(real)
                    loss_real = criterion(out_real, ones)

                    z = make_noise(bsz, args.z_dim, device)
                    fake = G(z).detach()
                    out_fake = D(fake)
                    loss_fake = criterion(out_fake, zeros)

                    loss_d = loss_real + loss_fake

                scaler.scale(loss_d).backward()
                scaler.step(optD)

                # ---- Train G ----
                optG.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=use_amp):
                    z = make_noise(bsz, args.z_dim, device)
                    fake2 = G(z)
                    out_fake2 = D(fake2)
                    loss_g = criterion(out_fake2, ones)

                scaler.scale(loss_g).backward()
                scaler.step(optG)
                scaler.update()

                with torch.no_grad():
                    dx = torch.sigmoid(out_real).mean().item()
                    dgz1 = torch.sigmoid(out_fake).mean().item()
                    dgz2 = torch.sigmoid(out_fake2).mean().item()

                running_d += float(loss_d.item())
                running_g += float(loss_g.item())
                running_dx += float(dx)
                running_dgz2 += float(dgz2)
                n_batches += 1

                if step % args.log_every == 0:
                    loss_writer.writerow(
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss_d": float(loss_d.item()),
                            "loss_g": float(loss_g.item()),
                            "dx": float(dx),
                            "dgz1": float(dgz1),
                            "dgz2": float(dgz2),
                        }
                    )
                    loss_f.flush()

                pbar.set_postfix(
                    loss_d=f"{loss_d.item():.3f}",
                    loss_g=f"{loss_g.item():.3f}",
                    D_x=f"{dx:.3f}",
                    D_Gz=f"{dgz2:.3f}",
                )
                step += 1

            avg_d = running_d / max(n_batches, 1)
            avg_g = running_g / max(n_batches, 1)
            avg_dx = running_dx / max(n_batches, 1)
            avg_dgz2 = running_dgz2 / max(n_batches, 1)
            elapsed = time.time() - start_time

            print(
                f"[Epoch {epoch}/{args.epochs}] "
                f"avgD={avg_d:.4f} avgG={avg_g:.4f} D(x)={avg_dx:.3f} D(G(z))={avg_dgz2:.3f} "
                f"elapsed={elapsed/60:.1f}min"
            )

            if epoch % args.sample_every == 0:
                out_img = samples_dir / f"epoch_{epoch:03d}.png"
                save_sample_grid(G, fixed_z, out_img)

            if epoch % args.save_every == 0:
                torch.save(G.state_dict(), ckpt_dir / f"generator_epoch_{epoch:03d}.pt")
                torch.save(D.state_dict(), ckpt_dir / f"discriminator_epoch_{epoch:03d}.pt")

        torch.save(G.state_dict(), ckpt_dir / "generator_last.pt")
        torch.save(D.state_dict(), ckpt_dir / "discriminator_last.pt")

    finally:
        loss_f.close()

    plot_losses_csv(losses_csv, metrics_dir / "loss_curve.png")

    print("Done.")
    print("Run dir:", run_dir)
    print("Latest generator:", ckpt_dir / "generator_last.pt")
    print("Samples dir:", samples_dir)


if __name__ == "__main__":
    main()