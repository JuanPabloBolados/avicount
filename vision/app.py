from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel
from trackers import BoTSORTTracker, ByteTrackTracker
from ultralytics import YOLO

APP_VERSION = "1.5.0-dev"
MODEL_VERSION = os.getenv("AVICOUNT_MODEL_VERSION", "sin-version")
MODEL_PATH = os.getenv("AVICOUNT_MODEL_PATH", "models/best.pt")
API_TOKEN = os.getenv("AVICOUNT_API_TOKEN", "").strip()
CONFIDENCE = float(os.getenv("AVICOUNT_CONFIDENCE", "0.25"))
IOU = float(os.getenv("AVICOUNT_IOU", "0.50"))
SLICE_SIZE = int(os.getenv("AVICOUNT_SLICE_SIZE", "640"))
SLICE_OVERLAP = int(os.getenv("AVICOUNT_SLICE_OVERLAP", "128"))
USE_SLICER = os.getenv("AVICOUNT_USE_SLICER", "1") not in {"0", "false", "False"}
CLASS_IDS = {
    int(v.strip())
    for v in os.getenv("AVICOUNT_CLASS_IDS", "").split(",")
    if v.strip()
}

app = FastAPI(title="AviCount Vision", version=APP_VERSION)
_model: YOLO | None = None


class CountRequest(BaseModel):
    image_b64: str


def _authorize(authorization: str | None) -> None:
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="token inválido")


def _get_model() -> YOLO:
    global _model
    if _model is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"modelo no disponible: {path}",
            )
        _model = YOLO(str(path))
    return _model


def _filter_detections(detections: sv.Detections) -> sv.Detections:
    if len(detections) == 0:
        return detections
    mask = np.ones(len(detections), dtype=bool)
    if detections.confidence is not None:
        mask &= detections.confidence >= CONFIDENCE
    if CLASS_IDS and detections.class_id is not None:
        mask &= np.isin(detections.class_id, list(CLASS_IDS))
    return detections[mask]


def _detect_tile(image: np.ndarray) -> sv.Detections:
    model = _get_model()
    result = model(image, conf=CONFIDENCE, iou=IOU, verbose=False)[0]
    return _filter_detections(sv.Detections.from_ultralytics(result))


def detect_image(image: np.ndarray) -> sv.Detections:
    if not USE_SLICER:
        return _detect_tile(image)

    h, w = image.shape[:2]
    # Para imágenes pequeñas no aporta valor partirlas; para galpones grandes,
    # el slicing conserva aves que serían muy pequeñas en la imagen completa.
    if max(h, w) <= SLICE_SIZE:
        return _detect_tile(image)

    slicer = sv.InferenceSlicer(
        callback=_detect_tile,
        slice_wh=(SLICE_SIZE, SLICE_SIZE),
        overlap_wh=(SLICE_OVERLAP, SLICE_OVERLAP),
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold=IOU,
        thread_workers=1,
    )
    return _filter_detections(slicer(image))


def _decode_image(value: str) -> np.ndarray:
    if "," in value and value.lstrip().startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="image_b64 inválida") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="imagen ilegible")
    return image


def _prediction_payload(detections: sv.Detections, image: np.ndarray) -> list[dict[str, Any]]:
    model = _get_model()
    names = getattr(model, "names", {}) or {}
    out: list[dict[str, Any]] = []
    for i, xyxy in enumerate(detections.xyxy):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        confidence = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 1.0
        )
        out.append(
            {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": confidence,
                "class_id": class_id,
                "class": str(names.get(class_id, "ave")),
            }
        )
    return out


@app.get("/salud")
def salud() -> dict[str, Any]:
    return {
        "ok": True,
        "servicio": "avicount-vision",
        "version": APP_VERSION,
        "modelo_version": MODEL_VERSION,
        "modelo_path": MODEL_PATH,
        "modelo_disponible": Path(MODEL_PATH).exists(),
        "supervision": sv.__version__,
        "slicing": USE_SLICER,
    }


@app.post("/contar")
def contar(
    body: CountRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(authorization)
    image = _decode_image(body.image_b64)
    detections = detect_image(image)
    h, w = image.shape[:2]
    return {
        "predictions": _prediction_payload(detections, image),
        "image": {"width": w, "height": h},
        "count": len(detections),
        "model_version": MODEL_VERSION,
        "pipeline": "supervision-inference-slicer" if USE_SLICER else "direct",
    }


def _video_detections(frame: np.ndarray) -> sv.Detections:
    # En video se evita slicing por frame: el tracker necesita latencia estable.
    return _detect_tile(frame)


def _count_door(video_path: str, stride: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="video ilegible")

    tracker = ByteTrackTracker()
    line_zone: sv.LineZone | None = None
    frames_read = 0
    frames_processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_read += 1
            if (frames_read - 1) % stride:
                continue

            if line_zone is None:
                h, w = frame.shape[:2]
                y = int(h * 0.50)
                line_zone = sv.LineZone(
                    start=sv.Point(x=0, y=y),
                    end=sv.Point(x=max(1, w - 1), y=y),
                )

            detections = _video_detections(frame)
            tracked = tracker.update(detections)
            line_zone.trigger(tracked)
            frames_processed += 1
    finally:
        cap.release()

    if line_zone is None:
        raise HTTPException(status_code=400, detail="video sin cuadros")

    in_count = int(line_zone.in_count)
    out_count = int(line_zone.out_count)
    # En una puerta importa cuántas aves cruzaron. Se entregan ambos sentidos
    # por separado para auditoría y el total de cruces como `neto`, contrato
    # que AviCount ya consume para registrar el movimiento.
    return {
        "modo": "puerta",
        "in_count": in_count,
        "out_count": out_count,
        "neto": in_count + out_count,
        "frames_leidos": frames_read,
        "frames_procesados": frames_processed,
        "modelo_version": MODEL_VERSION,
        "valido_inventario": True,
    }


def _track_walkthrough(video_path: str, stride: int) -> dict[str, Any]:
    """Métrica experimental para recorrido con cámara móvil.

    Usa BoT-SORT con compensación de movimiento de cámara para reducir cambios
    artificiales de ID. Aun así, NO declara el resultado como inventario hasta
    que se valide contra conteos humanos reales en galpones.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="video ilegible")

    tracker = BoTSORTTracker(enable_cmc=True, cmc_method="sparseOptFlow")
    seen_ids: set[int] = set()
    frame_counts: list[int] = []
    frames_read = 0
    frames_processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_read += 1
            if (frames_read - 1) % stride:
                continue
            detections = _video_detections(frame)
            tracked = tracker.update(detections, frame=frame)
            if tracked.tracker_id is not None:
                seen_ids.update(int(v) for v in tracked.tracker_id.tolist() if int(v) >= 0)
            frame_counts.append(len(tracked))
            frames_processed += 1
    finally:
        cap.release()

    if frames_processed == 0:
        raise HTTPException(status_code=400, detail="video sin cuadros procesables")

    # Compatibilidad con el contrato de la PWA. `neto` es una medición
    # experimental; el frontend v1.5 deberá respetar valido_inventario=false.
    tracklets = len(seen_ids)
    return {
        "modo": "recorrido",
        "in_count": tracklets,
        "out_count": 0,
        "neto": tracklets,
        "tracklets_unicos": tracklets,
        "max_aves_frame": max(frame_counts, default=0),
        "media_aves_frame": round(float(np.mean(frame_counts)), 2) if frame_counts else 0,
        "frames_leidos": frames_read,
        "frames_procesados": frames_processed,
        "modelo_version": MODEL_VERSION,
        "tracker": "BoT-SORT+CMC",
        "experimental": True,
        "valido_inventario": False,
        "advertencia": "recorrido con cámara móvil requiere validación antes de usar como inventario",
    }


@app.post("/contar_video")
async def contar_video(
    video: UploadFile = File(...),
    modo: str = Query(default="puerta", pattern="^(puerta|recorrido)$"),
    stride: int = Query(default=2, ge=1, le=10),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(authorization)
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await video.read(1024 * 1024):
            tmp.write(chunk)

    try:
        if modo == "puerta":
            return _count_door(tmp_path, stride)
        return _track_walkthrough(tmp_path, stride)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
