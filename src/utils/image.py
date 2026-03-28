import torch
from torchvision.utils import make_grid, save_image


@torch.no_grad()
def save_sample_grid(generator, z, out_path, normalize=True):
    generator.eval()
    fake = generator(z).detach().cpu()
    grid = make_grid(fake, nrow=int((fake.size(0)) ** 0.5), normalize=normalize, value_range=(-1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))