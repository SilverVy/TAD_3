#!/usr/bin/env python3
"""Download a small, attributed, open-license butterfly image subset from GBIF."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image


GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
PAGE_SIZE = 300
MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_LICENSES = {
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
    "https://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
    "http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
}
SPECIES = (
    {
        "folder": "danaus_plexippus",
        "scientific_name": "Danaus plexippus",
        "common_name": "Monarch",
        "gbif_taxon_key": 5133088,
    },
    {
        "folder": "pieris_rapae",
        "scientific_name": "Pieris rapae",
        "common_name": "Cabbage white",
        "gbif_taxon_key": 1920496,
    },
    {
        "folder": "papilio_machaon",
        "scientific_name": "Papilio machaon",
        "common_name": "Old World swallowtail",
        "gbif_taxon_key": 8225376,
    },
)


def _request_bytes(url, accept="application/json"):
    request = Request(
        url,
        headers={
            "User-Agent": "ButterflyMLOpsCoursework/1.0 (educational dataset preparation)",
            "Accept": accept,
        },
    )
    with urlopen(request, timeout=45) as response:
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Ответ превышает лимит размера файла 20 MiB.")
    return data


def _media_extensions(record):
    extensions = record.get("extensions") or {}
    for key in (
        "http://rs.tdwg.org/dwc/terms/Multimedia",
        "http://rs.gbif.org/terms/1.0/Multimedia",
    ):
        value = extensions.get(key)
        if isinstance(value, list):
            return value
    return record.get("media") or []


def _qualified_media(record):
    for media in _media_extensions(record):
        license_url = (
            media.get("http://purl.org/dc/terms/license")
            or media.get("license")
            or ""
        )
        image_url = (
            media.get("http://purl.org/dc/terms/identifier")
            or media.get("identifier")
            or ""
        )
        if license_url not in ALLOWED_LICENSES:
            continue
        if not image_url.startswith("https://"):
            continue
        if "image/" not in (
            media.get("http://purl.org/dc/terms/format")
            or media.get("format")
            or "image/"
        ).lower():
            continue
        return {
            "image_url": image_url,
            "license_url": license_url,
            "license_name": ALLOWED_LICENSES[license_url],
            "creator": media.get("http://purl.org/dc/terms/creator")
            or media.get("creator")
            or "",
            "rights_holder": media.get("http://purl.org/dc/terms/rightsHolder")
            or media.get("rightsHolder")
            or "",
            "reference_url": media.get("http://purl.org/dc/terms/references")
            or media.get("references")
            or "",
            "format": media.get("http://purl.org/dc/terms/format")
            or media.get("format")
            or "",
        }
    return None


def collect_candidates(species, wanted_count, max_records):
    candidates = []
    seen_occurrences = set()
    seen_urls = set()
    offset = 0
    candidate_target = max(wanted_count * 2, wanted_count + 10)

    while len(candidates) < candidate_target and offset < max_records:
        parameters = urlencode(
            {
                "taxonKey": species["gbif_taxon_key"],
                "mediaType": "StillImage",
                "offset": offset,
                "limit": PAGE_SIZE,
            }
        )
        request_url = GBIF_SEARCH_URL + "?" + parameters
        try:
            body = json.loads(_request_bytes(request_url).decode("utf-8"))
        except (HTTPError, URLError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Не удалось получить записи GBIF для {0}: {1}".format(
                    species["scientific_name"], error
                )
            )

        records = body.get("results") or []
        for record in records:
            occurrence_key = record.get("key")
            if occurrence_key is None or occurrence_key in seen_occurrences:
                continue
            media = _qualified_media(record)
            if media is None or media["image_url"] in seen_urls:
                continue

            seen_occurrences.add(occurrence_key)
            seen_urls.add(media["image_url"])
            media.update(
                {
                    "occurrence_key": occurrence_key,
                    "scientific_name": species["scientific_name"],
                    "common_name": species["common_name"],
                    "class_name": species["folder"],
                }
            )
            candidates.append(media)
            if len(candidates) >= candidate_target:
                break

        if body.get("endOfRecords") or not records:
            break
        offset += PAGE_SIZE
        time.sleep(0.15)

    if len(candidates) < wanted_count:
        raise ValueError(
            "Для {0} найдено только {1} изображений с лицензией CC0/CC BY 4.0 "
            "при лимите просмотра {2} записей; нужно {3}. Увеличьте --max-records "
            "или уменьшите --images-per-class.".format(
                species["scientific_name"],
                len(candidates),
                max_records,
                wanted_count,
            )
        )
    return candidates


def _download_image(image_url):
    medium_url = re.sub(r"/original(?=\.[^/?]+(?:\?|$))", "/medium", image_url)
    urls = [medium_url]
    if medium_url != image_url:
        urls.append(image_url)

    last_error = None
    for url in urls:
        try:
            return _request_bytes(url, accept="image/*"), url
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            last_error = error
    raise RuntimeError("Не удалось скачать изображение: {0}".format(last_error))


def _save_jpeg(data, path):
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            rgb_image = image.convert("RGB")
            dimensions = rgb_image.size
            rgb_image.save(str(path), format="JPEG", quality=95, optimize=True)
    except Exception as error:
        raise ValueError("Некорректный файл изображения: {0}".format(error))
    return dimensions


def download_species(species, candidates, temporary_root, images_per_class, used_hashes):
    class_dir = temporary_root / species["folder"]
    class_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    failures = []

    for candidate in candidates:
        if len(manifest_rows) >= images_per_class:
            break
        filename = "{0}_{1:04d}.jpg".format(
            species["folder"], len(manifest_rows) + 1
        )
        output_path = class_dir / filename
        try:
            image_bytes, downloaded_url = _download_image(candidate["image_url"])
            dimensions = _save_jpeg(image_bytes, output_path)
            image_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
            if image_hash in used_hashes:
                output_path.unlink()
                continue
            used_hashes.add(image_hash)
        except Exception as error:
            failures.append(
                "{0}: {1}".format(candidate["image_url"], error)
            )
            if output_path.exists():
                output_path.unlink()
            continue

        manifest_rows.append(
            {
                "local_path": "{0}/{1}".format(species["folder"], filename),
                "class_name": species["folder"],
                "common_name": candidate["common_name"],
                "scientific_name": candidate["scientific_name"],
                "gbif_occurrence_key": candidate["occurrence_key"],
                "creator": candidate["creator"],
                "rights_holder": candidate["rights_holder"],
                "license_name": candidate["license_name"],
                "license_url": candidate["license_url"],
                "reference_url": candidate["reference_url"],
                "source_image_url": candidate["image_url"],
                "downloaded_image_url": downloaded_url,
                "width": dimensions[0],
                "height": dimensions[1],
                "sha256": image_hash,
            }
        )
        time.sleep(0.1)

    if len(manifest_rows) < images_per_class:
        details = "; ".join(failures[:5]) or "доступных файлов с уникальным содержимым не хватило"
        raise RuntimeError(
            "Для {0} загружено {1} из {2} изображений. Первые ошибки: {3}".format(
                species["scientific_name"],
                len(manifest_rows),
                images_per_class,
                details,
            )
        )
    return manifest_rows


def prepare_dataset(output_dir, images_per_class, max_records):
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(
            "Каталог '{0}' уже содержит файлы. Чтобы не перезаписать данные, "
            "укажите другой --output-dir.".format(output_path)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".gbif_butterflies_", dir=str(output_path.parent))
    )
    rows = []
    used_hashes = set()
    try:
        for species in SPECIES:
            print(
                "Поиск лицензированных фото: {0}...".format(
                    species["scientific_name"]
                )
            )
            candidates = collect_candidates(species, images_per_class, max_records)
            rows.extend(
                download_species(
                    species, candidates, temporary_root, images_per_class, used_hashes
                )
            )
            print(
                "Загружено {0} изображений: {1}.".format(
                    images_per_class, species["scientific_name"]
                )
            )

        manifest_path = temporary_root / "sources.csv"
        fieldnames = [
            "local_path",
            "class_name",
            "common_name",
            "scientific_name",
            "gbif_occurrence_key",
            "creator",
            "rights_holder",
            "license_name",
            "license_url",
            "reference_url",
            "source_image_url",
            "downloaded_image_url",
            "width",
            "height",
            "sha256",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if output_path.exists():
            output_path.rmdir()
        os.replace(str(temporary_root), str(output_path))
    except Exception:
        shutil.rmtree(str(temporary_root), ignore_errors=True)
        raise

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Загружает лицензированные фотографии трех видов бабочек из GBIF "
            "и сохраняет атрибуцию в sources.csv."
        )
    )
    parser.add_argument("--output-dir", default="data/raw/butterflies")
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=6000)
    args = parser.parse_args()

    if args.images_per_class < 4 or args.images_per_class > 500:
        parser.error("--images-per-class должен быть от 4 до 500.")
    if args.max_records < PAGE_SIZE or args.max_records > 100000:
        parser.error("--max-records должен быть от 300 до 100000.")

    try:
        rows = prepare_dataset(
            args.output_dir, args.images_per_class, args.max_records
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, "Ошибка: {0}\n".format(error))

    print(
        "Готово: {0} изображений в {1}; атрибуция сохранена в sources.csv.".format(
            len(rows), args.output_dir
        )
    )


if __name__ == "__main__":
    main()
