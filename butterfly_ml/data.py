import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEMO_CLASSES = {
    "blue_morpho": ((35, 95, 220), (105, 175, 255)),
    "monarch": ((235, 105, 25), (255, 190, 45)),
    "swallowtail": ((245, 210, 35), (255, 245, 150)),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_samples(data_dir: str) -> Tuple[List[Dict[str, str]], List[str], str]:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError("Каталог датасета не найден: {0}".format(root))

    class_dirs = sorted(
        [item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")],
        key=lambda item: item.name.lower(),
    )
    if len(class_dirs) < 2:
        raise ValueError(
            "Нужно минимум 2 папки-класса с изображениями внутри каталога {0}.".format(root)
        )

    samples = []
    class_names = []
    for class_dir in class_dirs:
        image_paths = sorted(
            [
                item
                for item in class_dir.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda item: str(item).lower(),
        )
        if len(image_paths) < 2:
            raise ValueError(
                "В классе '{0}' найдено меньше двух изображений.".format(class_dir.name)
            )
        class_names.append(class_dir.name)
        for image_path in image_paths:
            relative_path = image_path.relative_to(root).as_posix()
            content_hash = _sha256_file(image_path)
            source_id = hashlib.sha256(
                (relative_path + "|" + content_hash).encode("utf-8")
            ).hexdigest()[:20]
            try:
                with Image.open(str(image_path)) as image:
                    image.verify()
            except Exception as error:
                raise ValueError(
                    "Не удалось прочитать изображение '{0}': {1}".format(image_path, error)
                )
            samples.append(
                {
                    "path": str(image_path),
                    "relative_path": relative_path,
                    "class_name": class_dir.name,
                    "source_id": source_id,
                    "sha256": content_hash,
                }
            )

    fingerprint_input = "\n".join(
        "{0}|{1}".format(sample["relative_path"], sample["sha256"])
        for sample in samples
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
    return samples, class_names, fingerprint


def extract_features(image_path: str, image_size: int) -> np.ndarray:
    try:
        with Image.open(str(image_path)) as original:
            image = original.convert("RGB").resize(
                (image_size, image_size), resample=Image.BILINEAR
            )
            return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    except Exception as error:
        raise ValueError(
            "Ошибка предобработки изображения '{0}': {1}".format(image_path, error)
        )


def _augmentation_variants(image: Image.Image):
    yield "mirror", ImageOps.mirror(image)
    yield "rotate_plus_12", image.rotate(
        12, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
    )
    yield "rotate_minus_12", image.rotate(
        -12, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
    )
    yield "brightness_up", ImageEnhance.Brightness(image).enhance(1.15)
    yield "color_down", ImageEnhance.Color(image).enhance(0.80)
    yield "contrast_up", ImageEnhance.Contrast(image).enhance(1.15)


def create_augmented_dataset(
    data_dir: str, processed_dir: str, augment_factor: int
) -> Dict[str, str]:
    if augment_factor < 1 or augment_factor > 6:
        raise ValueError("Параметр --augment-factor должен быть от 1 до 6.")

    samples, class_names, raw_fingerprint = collect_samples(data_dir)
    version_material = "{0}|augment-v2|{1}".format(raw_fingerprint, augment_factor)
    dataset_version = hashlib.sha256(version_material.encode("utf-8")).hexdigest()[:12]
    base_dir = Path(processed_dir)
    version_dir = base_dir / ("dataset_" + dataset_version)
    manifest_path = version_dir / "manifest.csv"
    metadata_path = version_dir / "version.json"

    if manifest_path.exists() and metadata_path.exists():
        return {
            "dataset_version": dataset_version,
            "version_dir": str(version_dir),
            "manifest_path": str(manifest_path),
            "raw_fingerprint": raw_fingerprint,
        }
    if version_dir.exists():
        raise RuntimeError(
            "Найдена неполная версия датасета '{0}'. Удалите только эту неполную папку "
            "и запустите команду снова.".format(version_dir)
        )

    base_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".dataset_{0}_".format(dataset_version), dir=str(base_dir))
    )
    rows = []
    try:
        images_dir = temporary_dir / "images"
        for sample in samples:
            class_dir = images_dir / sample["class_name"]
            class_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(sample["path"]) as original:
                image = original.convert("RGB")
                original_name = "{0}__original.jpg".format(sample["source_id"])
                original_path = class_dir / original_name
                image.save(str(original_path), format="JPEG", quality=95)
                rows.append(
                    _manifest_row(
                        dataset_version, sample, "original", original_path, temporary_dir
                    )
                )
                variants = list(_augmentation_variants(image))
                for number, (transform, augmented) in enumerate(
                    variants[:augment_factor], start=1
                ):
                    output_name = "{0}__{1:02d}_{2}.jpg".format(
                        sample["source_id"], number, transform
                    )
                    output_path = class_dir / output_name
                    augmented.save(str(output_path), format="JPEG", quality=95)
                    rows.append(
                        _manifest_row(
                            dataset_version,
                            sample,
                            transform,
                            output_path,
                            temporary_dir,
                        )
                    )

        with (temporary_dir / "manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as manifest_file:
            fieldnames = [
                "dataset_version",
                "source_id",
                "source_file",
                "source_sha256",
                "class_name",
                "transform",
                "relative_image_path",
                "image_sha256",
            ]
            writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        transform_names = [
            item[0]
            for item in _augmentation_variants(Image.new("RGB", (2, 2)))
        ][:augment_factor]
        metadata = {
            "dataset_version": dataset_version,
            "raw_fingerprint": raw_fingerprint,
            "augment_factor": augment_factor,
            "augmentation_transforms": transform_names,
            "classes": class_names,
            "source_image_count": len(samples),
            "saved_image_count": len(rows),
        }
        with (temporary_dir / "version.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

        os.replace(str(temporary_dir), str(version_dir))
    except Exception:
        shutil.rmtree(str(temporary_dir), ignore_errors=True)
        raise

    return {
        "dataset_version": dataset_version,
        "version_dir": str(version_dir),
        "manifest_path": str(manifest_path),
        "raw_fingerprint": raw_fingerprint,
    }


def _manifest_row(dataset_version, sample, transform, image_path, version_dir):
    return {
        "dataset_version": dataset_version,
        "source_id": sample["source_id"],
        "source_file": sample["relative_path"],
        "source_sha256": sample["sha256"],
        "class_name": sample["class_name"],
        "transform": transform,
        "relative_image_path": image_path.relative_to(version_dir).as_posix(),
        "image_sha256": _sha256_file(image_path),
    }


def generate_demo_dataset(
    output_dir: str, samples_per_class: int, seed: int
) -> int:
    if samples_per_class < 4 or samples_per_class > 500:
        raise ValueError(
            "Число демонстрационных изображений на класс должно быть от 4 до 500."
        )

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "Каталог '{0}' уже содержит файлы. Укажите пустой путь, чтобы не "
            "перезаписать пользовательские данные.".format(root)
        )
    root.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    image_size = 64

    for class_name, palette in DEMO_CLASSES.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            background = (
                randomizer.randint(215, 240),
                randomizer.randint(225, 245),
                randomizer.randint(205, 230),
            )
            image = Image.new("RGB", (image_size, image_size), background)
            draw = ImageDraw.Draw(image)
            center_x = image_size // 2 + randomizer.randint(-2, 2)
            center_y = image_size // 2 + randomizer.randint(-2, 2)
            wing_color = tuple(
                max(0, min(255, channel + randomizer.randint(-12, 12)))
                for channel in palette[0]
            )
            accent = palette[1]
            dark = (35, 36, 42)

            left_upper = (6, 8, center_x + 2, center_y + 5)
            right_upper = (
                center_x - 2,
                8,
                image_size - 6,
                center_y + 5,
            )
            left_lower = (11, center_y - 1, center_x + 1, 54)
            right_lower = (
                center_x - 1,
                center_y - 1,
                image_size - 11,
                54,
            )
            for bounds in (left_upper, right_upper, left_lower, right_lower):
                draw.ellipse(bounds, fill=wing_color, outline=dark, width=2)

            for side in (0, 1):
                mirror_x = center_x - 8 if side == 0 else center_x + 8
                for spot_index in range(3):
                    spot_y = 17 + spot_index * 8 + randomizer.randint(-2, 2)
                    radius = randomizer.randint(1, 3)
                    draw.ellipse(
                        (
                            mirror_x - radius,
                            spot_y - radius,
                            mirror_x + radius,
                            spot_y + radius,
                        ),
                        fill=accent,
                    )

            draw.line(
                (center_x, center_y - 14, center_x, center_y + 16),
                fill=dark,
                width=3,
            )
            draw.ellipse(
                (center_x - 3, center_y - 17, center_x + 3, center_y - 11),
                fill=dark,
            )
            draw.line(
                (center_x - 1, center_y - 15, center_x - 7, center_y - 22),
                fill=dark,
                width=1,
            )
            draw.line(
                (center_x + 1, center_y - 15, center_x + 7, center_y - 22),
                fill=dark,
                width=1,
            )
            image.save(str(class_dir / "demo_{0:04d}.png".format(index + 1)))

    return len(DEMO_CLASSES) * samples_per_class
