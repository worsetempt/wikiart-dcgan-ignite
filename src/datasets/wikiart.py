from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


class WikiArtImages(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 64,
        subset_file: str | None = None,
        max_images: int | None = None,
    ):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.image_size = int(image_size)

        if subset_file:
            sub = Path(subset_file).resolve()
            rels = [ln.strip() for ln in sub.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.paths = [self.root / r for r in rels]
        else:
            self.paths = _list_images(self.root)

        self.paths = [p for p in self.paths if p.exists()]
        if not self.paths:
            raise RuntimeError(f"No images found. root={self.root}, subset_file={subset_file}")

        if max_images is not None:
            self.paths = self.paths[: int(max_images)]

        self.tf = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            return self.tf(img)