import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import PIL
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from butterfly_ml.data import collect_samples, create_augmented_dataset, extract_features
from butterfly_ml.models import create_models
from butterfly_ml.tracking import append_experiment, new_experiment_id


def run_pipeline(
    data_dir,
    processed_dir="data/processed",
    artifact_dir="artifacts",
    test_size=0.25,
    augment_factor=3,
    image_size=16,
    seed=42,
):
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("Параметр --test-size должен быть в диапазоне от 0.1 до 0.5.")
    if image_size < 8 or image_size > 32:
        raise ValueError("Параметр --image-size должен быть в диапазоне от 8 до 32.")

    samples, classes, raw_fingerprint = collect_samples(data_dir)
    labels = np.asarray([sample["class_name"] for sample in samples])
    if len(samples) < len(classes) * 2:
        raise ValueError("Недостаточно изображений для обучения и тестирования.")

    test_count = int(math.ceil(len(samples) * test_size))
    train_count = len(samples) - test_count
    if test_count < len(classes) or train_count < len(classes):
        raise ValueError(
            "При текущем --test-size тестовая и обучающая части должны содержать "
            "не меньше одного изображения каждого класса. Увеличьте число изображений "
            "или измените --test-size."
        )
    class_counts = {
        class_name: int(np.sum(labels == class_name)) for class_name in classes
    }
    if min(class_counts.values()) < 2:
        raise ValueError(
            "Для стратифицированного разделения нужно минимум 2 изображения "
            "в каждом классе."
        )

    all_indices = np.arange(len(samples))
    try:
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError("Не удалось разделить датасет: {0}".format(error))

    print("Предобработка {0} исходных изображений...".format(len(samples)))
    features = np.vstack(
        [extract_features(sample["path"], image_size) for sample in samples]
    )
    dataset_info = create_augmented_dataset(
        data_dir, processed_dir, augment_factor
    )
    print(
        "Сохранен версионированный датасет {0}: {1}".format(
            dataset_info["dataset_version"], dataset_info["version_dir"]
        )
    )

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    train_sources = {samples[int(index)]["source_id"] for index in train_indices}
    augmented_features, augmented_labels = _load_training_augmentations(
        dataset_info["version_dir"],
        train_sources,
        image_size,
    )
    x_train_original = features[train_indices]
    y_train_original = labels[train_indices]
    x_test = features[test_indices]
    y_test = labels[test_indices]

    common_record = {
        "seed": int(seed),
        "test_size": float(test_size),
        "image_size": int(image_size),
        "augmentation_factor": int(augment_factor),
        "dataset_version": dataset_info["dataset_version"],
        "raw_fingerprint": raw_fingerprint,
        "raw_image_count": int(len(samples)),
        "class_count": int(len(classes)),
        "classes": classes,
        "original_train_count": int(len(train_indices)),
        "test_count": int(len(test_indices)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    records = []
    for condition in ("original_only", "original_plus_augmentation"):
        if condition == "original_only":
            x_train, y_train = x_train_original, y_train_original
        else:
            x_train = np.vstack((x_train_original, augmented_features))
            y_train = np.concatenate((y_train_original, augmented_labels))

        for model_name, estimator in create_models(seed).items():
            print(
                "Обучение: {0} / {1} ({2} примеров)...".format(
                    condition, model_name, len(y_train)
                )
            )
            try:
                estimator.fit(x_train, y_train)
                predictions = estimator.predict(x_test)
            except Exception as error:
                raise RuntimeError(
                    "Не удалось обучить модель {0}: {1}".format(model_name, error)
                )

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision_weighted": float(
                    precision_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                "recall_weighted": float(
                    recall_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                "f1_weighted": float(
                    f1_score(y_test, predictions, average="weighted", zero_division=0)
                ),
            }
            experiment_id = new_experiment_id()
            experiment_dir = Path(artifact_dir) / "experiments" / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=False)
            model_path = experiment_dir / "model.joblib"
            joblib.dump(estimator, str(model_path))
            model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

            record = dict(common_record)
            record.update(metrics)
            record.update(
                {
                    "experiment_id": experiment_id,
                    "condition": condition,
                    "model": model_name,
                    "train_count": int(len(y_train)),
                    "augmented_train_count": int(
                        len(y_train) - len(y_train_original)
                    ),
                    "model_path": str(model_path),
                    "model_sha256": model_hash,
                    "model_parameters": estimator.get_params(deep=False),
                    "software_versions": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__,
                        "pillow": PIL.__version__,
                        "joblib": joblib.__version__,
                    },
                }
            )
            _save_evaluation_files(
                experiment_dir,
                samples,
                test_indices,
                y_test,
                predictions,
                classes,
            )
            append_experiment(artifact_dir, record)
            records.append(record)

    return records


def _load_training_augmentations(version_dir, source_ids, image_size):
    version_root = Path(version_dir)
    manifest_path = version_root / "manifest.csv"
    vectors = []
    labels = []
    with manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        for row in reader:
            if row["source_id"] not in source_ids or row["transform"] == "original":
                continue
            image_path = version_root / row["relative_image_path"]
            vectors.append(extract_features(str(image_path), image_size))
            labels.append(row["class_name"])
    if not vectors:
        raise ValueError(
            "Не удалось загрузить аугментированные изображения обучающей выборки."
        )
    return np.vstack(vectors), np.asarray(labels)


def _save_evaluation_files(
    experiment_dir, samples, test_indices, expected, predicted, classes
):
    prediction_path = Path(experiment_dir) / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as prediction_file:
        writer = csv.writer(prediction_file)
        writer.writerow(["source_file", "expected_class", "predicted_class"])
        for position, sample_index in enumerate(test_indices):
            writer.writerow(
                [
                    samples[int(sample_index)]["relative_path"],
                    expected[position],
                    predicted[position],
                ]
            )

    report = classification_report(
        expected,
        predicted,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    with (Path(experiment_dir) / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    matrix = confusion_matrix(expected, predicted, labels=classes)
    with (Path(experiment_dir) / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["true/predicted"] + list(classes))
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name] + [int(value) for value in row])
