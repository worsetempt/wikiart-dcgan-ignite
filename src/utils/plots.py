from pathlib import Path
import csv
import matplotlib.pyplot as plt


def plot_losses_csv(csv_path: Path, out_png: Path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    x = [int(r["step"]) for r in rows]
    d = [float(r["loss_d"]) for r in rows]
    g = [float(r["loss_g"]) for r in rows]

    plt.figure()
    plt.plot(x, d, label="D loss")
    plt.plot(x, g, label="G loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()