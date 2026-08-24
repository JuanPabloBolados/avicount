# AviCount

AviCount es un proyecto independiente de visión computacional para conteo avícola.

## Componentes

- `index.html`: PWA móvil de captura, conteo manual, videos y dataset.
- `vision/`: servicio de IA para fotos y videos con Supervision + trackers.
- Supabase: autenticación, sesiones, capturas, videos, dataset y API del proyecto.

## Alcance actual

AviCount se desarrolla y valida de forma independiente. No depende del ERP para funcionar y no debe usar resultados experimentales de visión como inventario automático hasta completar benchmarks contra conteos humanos reales.

La integración con otros sistemas queda fuera del camino crítico de AviCount y puede mantenerse como una API opcional de solo lectura.

## Visión v1.5

La rama `feature/vision-supervision-v1.5` incorpora la base del motor de visión:

- fotos: YOLO + Supervision `InferenceSlicer`;
- puerta fija: ByteTrack + `LineZone`;
- recorrido con cámara móvil: BoT-SORT + compensación de movimiento, todavía experimental.

Ver `vision/README.md` para configuración y gates de validación.
