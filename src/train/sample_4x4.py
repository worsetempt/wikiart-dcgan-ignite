import torch
from torchvision.utils import make_grid, save_image
from pathlib import Path

from src.models.dcgan import Generator

# -------- CONFIG --------
ckpt_path = "outputs/full_cached_b256_tuned_v2/checkpoints/generator_last.pt"
out_path = "outputs/full_cached_b256_tuned_v2/sample_4x4.png"

z_dim = 128
g_ch = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

device = torch.device(device)

# Load generator
G = Generator(z_dim=z_dim, base_ch=g_ch).to(device)
G.load_state_dict(torch.load(ckpt_path, map_location=device))
G.eval()

# Generate 16 samples (4x4 grid)
with torch.no_grad():
    z = torch.randn(16, z_dim, 1, 1, device=device)
    fake = G(z)

grid = make_grid(fake, nrow=4, normalize=True, value_range=(-1, 1))
save_image(grid, out_path)

print("Saved:", out_path)