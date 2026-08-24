from __future__ import annotations

import argparse
from pathlib import Path

import cv2


class Annotator:
    def __init__(self, image_path: Path, label_path: Path):
        self.image_path = image_path
        self.label_path = label_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise RuntimeError(f"No se pudo abrir {image_path}")
        self.base = self.image.copy()
        self.boxes: list[tuple[int, int, int, int]] = []
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.label_path.exists():
            return
        h, w = self.image.shape[:2]
        for line in self.label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            self.boxes.append((x1, y1, x2, y2))

    def mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            x1, y1 = self.start
            x2, y2 = x, y
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                self.boxes.append((x1, y1, x2, y2))
            self.start = None
            self.current = None

    def render(self):
        img = self.base.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.start and self.current:
            cv2.rectangle(img, self.start, self.current, (255, 255, 0), 1)
        cv2.putText(
            img,
            f"boxes: {len(self.boxes)} | arrastra=crear  u=deshacer  s=guardar  n=guardar+siguiente  q=salir",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"boxes: {len(self.boxes)} | arrastra=crear  u=deshacer  s=guardar  n=guardar+siguiente  q=salir",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return img

    def save(self) -> None:
        h, w = self.image.shape[:2]
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for x1, y1, x2, y2 in self.boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        self.label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Anotador local YOLO de una sola clase: ave")
    ap.add_argument("images", help="Carpeta con imágenes")
    ap.add_argument("--labels", default="labels", help="Carpeta de salida para .txt YOLO")
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        raise SystemExit("No hay imágenes para anotar")

    window = "AviCount annotate_boxes"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for image_path in files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        ann = Annotator(image_path, label_path)
        cv2.setMouseCallback(window, ann.mouse)
        while True:
            cv2.imshow(window, ann.render())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u"):
                if ann.boxes:
                    ann.boxes.pop()
            elif key == ord("s"):
                ann.save()
            elif key == ord("n"):
                ann.save()
                break
            elif key == ord("q"):
                ann.save()
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
