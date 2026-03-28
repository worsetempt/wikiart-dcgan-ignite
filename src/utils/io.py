import json
from pathlib import Path


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj: dict):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_text_lines(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]