from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests

RECORD_ID = 16686320
API = f"https://zenodo.org/api/records/{RECORD_ID}"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract(path: Path, out: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        return True
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as t:
            t.extractall(out)
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga PIO desde el registro oficial de Zenodo")
    ap.add_argument("--out", default="data/pio", help="Directorio de destino")
    ap.add_argument("--include-videos", action="store_true", help="Incluye archivos cuyo nombre parezca video")
    args = ap.parse_args()

    out = Path(args.out)
    raw = out / "raw"
    extracted = out / "extracted"
    raw.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    record = requests.get(API, timeout=60).json()
    files = record.get("files", [])
    if not files:
        raise RuntimeError("Zenodo no devolvió archivos para el registro PIO")

    manifest = {
        "record_id": RECORD_ID,
        "doi": record.get("doi"),
        "title": record.get("metadata", {}).get("title"),
        "files": [],
    }

    for item in files:
        key = item.get("key") or item.get("filename")
        if not key:
            continue
        low = key.lower()
        if not args.include_videos and any(x in low for x in ("video", ".mp4", ".avi", ".mov")):
            continue

        links = item.get("links", {})
        url = links.get("content") or links.get("self")
        if not url:
            continue

        dest = raw / Path(key).name
        if not dest.exists():
            print(f"Descargando {dest.name}...")
            download(url, dest)

        checksum = str(item.get("checksum") or "")
        if checksum.startswith("md5:"):
            esperado = checksum.split(":", 1)[1]
            real = md5_file(dest)
            if real != esperado:
                raise RuntimeError(f"Checksum inválido para {dest.name}: {real} != {esperado}")

        extracted_ok = extract(dest, extracted)
        manifest["files"].append(
            {
                "name": dest.name,
                "bytes": dest.stat().st_size,
                "checksum": checksum,
                "extracted": extracted_ok,
            }
        )

    (out / "zenodo_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    yamls = list(extracted.rglob("*.yaml")) + list(extracted.rglob("*.yml"))
    print(f"Listo: {len(manifest['files'])} archivo(s).")
    if yamls:
        print("Configs encontradas:")
        for p in yamls[:20]:
            print(" -", p)
    else:
        print("No encontré YAML automáticamente; revisa", extracted)


if __name__ == "__main__":
    main()
