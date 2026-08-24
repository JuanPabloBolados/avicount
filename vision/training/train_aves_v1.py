from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr import RFDETRBase


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tuning RF-DETR Base para AviCount aves-v1")
    ap.add_argument("dataset", help="Directorio del dataset COCO o YOLO")
    ap.add_argument("--out", default="runs/aves-v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=672, help="Debe ser divisible por 56")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset no encontrado: {dataset}")
    if args.resolution % 56 != 0:
        raise SystemExit("RF-DETR requiere resolution divisible por 56")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = RFDETRBase()
    model.train(
        dataset_dir=str(dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        resolution=args.resolution,
        device=args.device,
        output_dir=str(out),
        early_stopping=True,
        early_stopping_patience=12,
        use_ema=True,
        gradient_checkpointing=True,
        tensorboard=True,
        resume=args.resume,
    )

    best = out / "checkpoint_best_total.pth"
    print("Entrenamiento terminado.")
    print("Checkpoint esperado:", best)
    print("Promoción AviCount: copiar/renombrar como vision/models/aves-v1.pth SOLO después del benchmark.")


if __name__ == "__main__":
    main()
