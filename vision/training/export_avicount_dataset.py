from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import unquote

import requests

DEFAULT_API = "https://mgabiviiocowkeohuerc.supabase.co/functions/v1/avicount"


def get_page(api: str, token: str, desde: str | None, desde_id: str | None) -> dict:
    params = {"limite": 200}
    if desde:
        params["desde"] = desde
    if desde_id:
        params["desde_id"] = desde_id
    r = requests.get(
        api.rstrip("/") + "/admin/dataset",
        params=params,
        headers={"x-admin": token},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta material privado de AviCount para anotación/validación")
    ap.add_argument("--api", default=os.getenv("AVICOUNT_API", DEFAULT_API))
    ap.add_argument("--admin-token", default=os.getenv("AVICOUNT_ADMIN_TOKEN", ""))
    ap.add_argument("--out", default="data/avicount-private")
    args = ap.parse_args()

    if not args.admin_token:
        raise SystemExit("Falta AVICOUNT_ADMIN_TOKEN o --admin-token")

    out = Path(args.out)
    images = out / "images"
    pseudo = out / "pseudo_predictions"
    images.mkdir(parents=True, exist_ok=True)
    pseudo.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.jsonl"
    rows: list[dict] = []
    desde = None
    desde_id = None

    while True:
        page = get_page(args.api, args.admin_token, desde, desde_id)
        items = page.get("items", [])
        for item in items:
            cid = str(item["id"])
            url = item.get("url")
            if not url:
                continue
            dest = images / f"{cid}.jpg"
            if not dest.exists():
                download(url, dest)

            # `detecciones` son predicciones del modelo, no boxes verificadas.
            # Se guardan aparte para acelerar una futura revisión humana, pero
            # NUNCA se escriben como labels de entrenamiento automáticamente.
            preds = item.get("detecciones")
            if preds:
                (pseudo / f"{cid}.json").write_text(
                    json.dumps(preds, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            rows.append(
                {
                    "id": cid,
                    "image": str(dest),
                    "galpon_id": item.get("galpon_id"),
                    "zona": item.get("zona"),
                    "conteo_manual": item.get("conteo_manual"),
                    "conteo_detectado": item.get("conteo_detectado"),
                    "modelo_version": item.get("modelo_version"),
                    "revisada_por_humano": item.get("revisada_por_humano"),
                    "corregida_por_humano": item.get("corregida_por_humano"),
                    "created_at": item.get("created_at"),
                    "ground_truth_boxes": False,
                }
            )

        if not page.get("hay_mas") or not items:
            break
        desde = unquote(page.get("siguiente_desde") or "") or None
        desde_id = page.get("siguiente_desde_id")

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exportadas {len(rows)} imagen(es) a {images}")
    print("Ground truth de boxes: 0 generado automáticamente (por diseño).")
    print("Anota las imágenes con annotate_boxes.py antes de usarlas para entrenamiento.")


if __name__ == "__main__":
    main()
