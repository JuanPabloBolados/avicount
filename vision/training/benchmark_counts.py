from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rfdetr import RFDETRBase


def count_labels(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark de conteo por imagen para aves-v1")
    ap.add_argument("checkpoint")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out", default="benchmark-aves-v1.json")
    args = ap.parse_args()

    model = RFDETRBase(pretrain_weights=args.checkpoint)
    images = Path(args.images)
    labels = Path(args.labels)
    rows = []

    for img in sorted(images.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = labels / f"{img.stem}.txt"
        if not label.exists():
            continue
        gt = count_labels(label)
        pred = len(model.predict(str(img), threshold=args.threshold))
        err = pred - gt
        rows.append(
            {
                "image": img.name,
                "ground_truth": gt,
                "predicted": pred,
                "error": err,
                "abs_error": abs(err),
                "ape_pct": (abs(err) / gt * 100.0) if gt > 0 else None,
            }
        )

    if not rows:
        raise SystemExit("No hay pares imagen/label para evaluar")

    mae = float(np.mean([r["abs_error"] for r in rows]))
    ape = [r["ape_pct"] for r in rows if r["ape_pct"] is not None]
    mape = float(np.mean(ape)) if ape else None
    bias = float(np.mean([r["error"] for r in rows]))

    report = {
        "images": len(rows),
        "threshold": args.threshold,
        "mae": round(mae, 4),
        "mape_pct": round(mape, 4) if mape is not None else None,
        "bias": round(bias, 4),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
