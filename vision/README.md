# AviCount Vision — v1.5 (experimental)

Servicio de computer vision independiente del ERP. Su única responsabilidad es convertir fotos y videos de AviCount en detecciones y métricas de conteo.

## Arquitectura

```text
AviCount PWA
   |
   +-- POST /contar -----------------> YOLO + Supervision InferenceSlicer
   |                                      |
   |                                      +--> detecciones + cajas + confianza
   |
   +-- POST /contar_video?modo=puerta -> YOLO + ByteTrack + LineZone
   |                                      |
   |                                      +--> cruces IN / OUT / total
   |
   +-- POST /contar_video?modo=recorrido
                                          |
                                          +--> YOLO + BoT-SORT + CMC
                                               (experimental, NO inventario)
```

## Regla de seguridad de datos

El modo `recorrido` devuelve `valido_inventario=false`. Aunque entrega una métrica de `tracklets_unicos`, no debe usarse para cerrar inventario hasta validarlo contra conteos humanos reales. La cámara se mueve y una misma ave puede perder y recuperar identidad.

El modo `puerta` sí está diseñado para contar cruces de una línea fija. Aun así debe calibrarse y validarse con videos reales antes de uso operativo.

## Modelo

Copiar el peso entrenado a:

```text
vision/models/best.pt
```

O definir otra ruta mediante `AVICOUNT_MODEL_PATH`.

## Variables de entorno

- `AVICOUNT_MODEL_PATH`: ruta al archivo `.pt`. Default: `models/best.pt`.
- `AVICOUNT_MODEL_VERSION`: versión que queda registrada en las respuestas.
- `AVICOUNT_API_TOKEN`: token Bearer opcional. Si está vacío, no exige token.
- `AVICOUNT_CONFIDENCE`: confianza mínima. Default: `0.25`.
- `AVICOUNT_IOU`: IoU para NMS/merge. Default: `0.50`.
- `AVICOUNT_SLICE_SIZE`: tamaño de tile. Default: `640`.
- `AVICOUNT_SLICE_OVERLAP`: solape entre tiles en píxeles. Default: `128`.
- `AVICOUNT_USE_SLICER`: `1`/`0`. Default: `1`.
- `AVICOUNT_CLASS_IDS`: IDs de clase separados por coma. Vacío = todas las clases.

## Ejecutar localmente

```bash
cd vision
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Salud:

```text
GET /salud
```

Conteo de foto (contrato compatible con la PWA actual):

```text
POST /contar
Content-Type: application/json

{"image_b64":"..."}
```

Respuesta principal:

```json
{
  "predictions": [
    {
      "x": 320.0,
      "y": 240.0,
      "width": 80.0,
      "height": 70.0,
      "confidence": 0.91,
      "class_id": 0,
      "class": "gallina"
    }
  ],
  "image": {"width": 1280, "height": 720},
  "count": 1,
  "model_version": "aves-v1"
}
```

Conteo de video:

```text
POST /contar_video?modo=puerta&stride=2
multipart/form-data: video=<archivo>
```

Para recorrido:

```text
POST /contar_video?modo=recorrido&stride=2
```

La respuesta de recorrido siempre incluye:

```json
{
  "experimental": true,
  "valido_inventario": false,
  "tracker": "BoT-SORT+CMC"
}
```

## Docker

```bash
cd vision
docker build -t avicount-vision .
docker run --rm -p 8000:8000 \
  -e AVICOUNT_MODEL_VERSION=aves-v1 \
  -e AVICOUNT_API_TOKEN=CAMBIAR \
  avicount-vision
```

## Activación en AviCount

No publicar `modelo_url` en producción hasta cumplir estos gates:

1. Modelo de aves disponible y versionado.
2. `/salud` responde `modelo_disponible=true`.
3. Fotos: benchmark contra conteo humano en un conjunto separado de entrenamiento.
4. Puerta: benchmark de cruces IN/OUT con videos reales.
5. Recorrido: sigue experimental hasta demostrar error aceptable; `valido_inventario=false` debe respetarse en la PWA.

## Métricas mínimas a registrar en la validación

- error absoluto;
- error porcentual;
- mediana del error;
- P95 del error;
- falsos positivos / falsos negativos en fotos;
- error por galpón, edad y condición de iluminación;
- versión exacta del modelo.
