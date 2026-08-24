# AviCount — entrenamiento `aves-v1`

Este directorio prepara el primer detector avícola de AviCount sin depender del ERP.

## Auditoría del dataset propio — 2026-08-24

Estado verificado directamente en el proyecto Supabase de AviCount:

- 8 fotos almacenadas.
- 1 video almacenado.
- 1 galpón con material (`Galpón 1`).
- 0 fotos con conteo manual.
- 0 fotos con detecciones/modelo.
- 0 fotos con bounding boxes verificadas por humano.
- Las 8 fotos son cuadros extraídos del mismo recorrido y están `pendiente`.

**Conclusión:** esas 8 imágenes todavía no son ground truth para entrenar un detector. Se conservan como material propio para anotar y validar, pero no se inventan etiquetas.

## Base externa recomendada

Para no entrenar desde cero, `aves-v1` parte de un dataset público avícola y luego se adapta con imágenes propias.

### PIO

- 1.487 imágenes reales de galpones.
- 327.289 instancias de pollos anotadas con bounding boxes.
- Formato YOLO.
- Capturas en instalaciones comerciales y prototipo, con variaciones de iluminación, densidad y edad.
- DOI: `10.5281/zenodo.16686320`.

Artículo: Boniche et al., *Scientific Data* 13, 801 (2026).

### ChickenVerse (segunda fuente opcional)

- ChickenDet: 6.539 imágenes.
- 153.764 aves anotadas.
- COCO bounding boxes + máscaras.
- Conviene incorporarlo después de medir PIO, para no mezclar dominios sin saber si mejora Codigua.

## Detector

`aves-v1` usa **RF-DETR Base** como detector inicial.

Motivos:

1. RF-DETR core y sus pesos abiertos están bajo Apache-2.0.
2. Admite fine-tuning con datasets COCO o YOLO.
3. Devuelve `supervision.Detections`, por lo que encaja directamente con Supervision.
4. Evita introducir una dependencia AGPL en el detector del producto.

Supervision sigue siendo la capa de slicing, zonas y postprocesamiento.

## Flujo

```text
PIO (ground truth público)
        +
AviCount (imágenes propias anotadas)
        |
        v
RF-DETR Base
        |
        v
aves-v1.pth
        |
        v
AviCount Vision + Supervision
```

## Herramientas

- `download_pio.py`: descarga los archivos del registro oficial de Zenodo.
- `export_avicount_dataset.py`: exporta imágenes privadas de AviCount mediante el endpoint administrativo existente; nunca convierte predicciones del modelo en ground truth humano.
- `annotate_boxes.py`: anotador local simple para crear bounding boxes YOLO en imágenes propias.
- `build_yolo_dataset.py`: crea splits train/valid/test reproducibles desde imágenes + labels propios.
- `train_aves_v1.py`: fine-tuning RF-DETR.
- `benchmark_counts.py`: compara conteos derivados de boxes contra ground truth y calcula MAE/MAPE.

## Regla de promoción

No publicar `modelo_url` en AviCount hasta que `aves-v1` pase un benchmark con imágenes reales de los galpones objetivo. Como mínimo se debe reportar:

- MAE de conteo por imagen.
- MAPE (ignorando imágenes de ground truth 0).
- precisión/recall/mAP del detector.
- resultados separados por galpón/condición cuando haya material suficiente.

El modo `recorrido` permanece experimental aunque el detector sea bueno; detector y deduplicación temporal son problemas distintos.
