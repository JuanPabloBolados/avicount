from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_pio_root(root: Path) -> Path:
    candidates = []
    for p in root.rglob('images'):
        if (p / 'train').is_dir() and (p.parent / 'labels' / 'train').is_dir():
            candidates.append(p.parent)
    if not candidates:
        raise SystemExit(f'No encontré estructura PIO images/train + labels/train dentro de {root}')
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def paired_images(img_dir: Path, label_dir: Path) -> list[Path]:
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and (label_dir / f'{p.stem}.txt').exists():
            out.append(p)
    return out


def copy_split(src_root: Path, split_src: str, dst_root: Path, split_dst: str, limit: int | None, seed: int) -> int:
    img_src = src_root / 'images' / split_src
    lbl_src = src_root / 'labels' / split_src
    images = paired_images(img_src, lbl_src)
    rng = random.Random(seed)
    rng.shuffle(images)
    if limit is not None:
        images = images[:limit]

    img_dst = dst_root / split_dst / 'images'
    lbl_dst = dst_root / split_dst / 'labels'
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img in images:
        shutil.copy2(img, img_dst / img.name)
        shutil.copy2(lbl_src / f'{img.stem}.txt', lbl_dst / f'{img.stem}.txt')
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser(description='Normaliza PIO a layout YOLO esperado por RF-DETR')
    ap.add_argument('source', help='Directorio extraído por download_pio.py')
    ap.add_argument('--out', default='data/pio-rfdetr')
    ap.add_argument('--train-limit', type=int, default=None)
    ap.add_argument('--valid-limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=24082026)
    args = ap.parse_args()

    src = find_pio_root(Path(args.source))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n_train = copy_split(src, 'train', out, 'train', args.train_limit, args.seed)
    val_name = 'val' if (src / 'images' / 'val').is_dir() else 'valid'
    n_valid = copy_split(src, val_name, out, 'valid', args.valid_limit, args.seed + 1)

    data = {
        'names': ['ave'],
        'nc': 1,
        'train': 'train/images',
        'val': 'valid/images',
    }
    (out / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(f'PIO RF-DETR listo: train={n_train}, valid={n_valid}, source={src}')


if __name__ == '__main__':
    main()
