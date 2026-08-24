from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Crea train/valid/test para una clase 'ave'")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="data/codigua-yolo")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--valid", type=float, default=0.10)
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out = Path(args.out)

    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))

    if len(pairs) < 3:
        raise SystemExit("Se requieren al menos 3 imágenes anotadas para crear train/valid/test")

    if args.train <= 0 or args.valid <= 0 or args.train + args.valid >= 1:
        raise SystemExit("Splits inválidos: train y valid deben ser >0 y dejar espacio para test")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = max(1, round(n * args.train))
    n_valid = max(1, round(n * args.valid))
    if n_train + n_valid >= n:
        n_train = max(1, n - 2)
        n_valid = 1

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test": pairs[n_train + n_valid:],
    }

    for split, items in splits.items():
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, label in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(label, lbl_out / label.name)

    yaml = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "names:\n"
        "  0: ave\n"
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    print("Dataset creado:")
    for split, items in splits.items():
        print(f"  {split}: {len(items)}")
    print("  config:", out / "data.yaml")


if __name__ == "__main__":
    main()
