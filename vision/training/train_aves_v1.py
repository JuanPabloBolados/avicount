from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall

MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
}
DEFAULT_RESOLUTION = {
    "nano": 384,
    "small": 512,
    "base": 560,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tuning RF-DETR para AviCount aves-v1")
    ap.add_argument("dataset", help="Directorio del dataset COCO o YOLO")
    ap.add_argument("--model", choices=MODELS, default="base")
    ap.add_argument("--out", default="runs/aves-v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-interval", type=int, default=1)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset no encontrado: {dataset}")

    resolution = args.resolution or DEFAULT_RESOLUTION[args.model]
    divisor = 32 if args.model in {"nano", "small"} else 56
    if resolution % divisor != 0:
        raise SystemExit(f"{args.model} requiere resolution divisible por {divisor}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = MODELS[args.model]()
    model.train(
        dataset_dir=str(dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        resolution=resolution,
        device=args.device,
        output_dir=str(out),
        early_stopping=args.epochs > 3,
        early_stopping_patience=min(12, max(2, args.epochs // 4)),
        use_ema=True,
        gradient_checkpointing=args.device != "cpu",
        tensorboard=True,
        resume=args.resume,
        num_workers=args.workers,
        eval_interval=args.eval_interval,
        notes={
            "project": "AviCount",
            "model": "aves-v1",
            "variant": args.model,
            "purpose": "bootstrap poultry detector; not production until benchmark passes",
        },
    )

    best = out / "checkpoint_best_total.pth"
    print("Entrenamiento terminado.")
    print("Checkpoint esperado:", best)
    print("Promoción AviCount: copiar/renombrar como vision/models/aves-v1.pth SOLO después del benchmark.")


if __name__ == "__main__":
    main()
