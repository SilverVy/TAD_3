import csv
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def new_experiment_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return "exp_{0}_{1}".format(stamp, uuid.uuid4().hex[:6])


def append_experiment(artifact_dir, record):
    root = Path(artifact_dir)
    experiment_dir = root / "experiments" / record["experiment_id"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    record["model_path"] = str(experiment_dir / "model.joblib")
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(record, metrics_file, ensure_ascii=False, indent=2)

    history_path = root / "experiments.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    regenerate_results_csv(root, history_path)
    return record


def load_history(artifact_dir):
    history_path = Path(artifact_dir) / "experiments.jsonl"
    if not history_path.exists():
        return []
    records = []
    with history_path.open("r", encoding="utf-8") as history_file:
        for line_number, line in enumerate(history_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as error:
                raise ValueError(
                    "Повреждена строка {0} в {1}: {2}".format(
                        line_number, history_path, error
                    )
                )
    return records


def regenerate_results_csv(root, history_path):
    root = Path(root)
    records = load_history(str(root))
    if not records:
        return
    columns = [
        "experiment_id",
        "condition",
        "model",
        "seed",
        "dataset_version",
        "raw_fingerprint",
        "train_count",
        "test_count",
        "model_sha256",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "created_at",
        "model_path",
    ]
    temporary_path = root / "results.csv.tmp"
    with temporary_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(str(temporary_path), str(root / "results.csv"))


def select_experiment(artifact_dir, experiment_id):
    root = Path(artifact_dir)
    records = load_history(artifact_dir)
    matches = [record for record in records if record["experiment_id"] == experiment_id]
    if not matches:
        raise ValueError("Эксперимент '{0}' отсутствует в истории.".format(experiment_id))
    record = matches[-1]
    source = Path(record["model_path"])
    if not source.exists():
        raise ValueError(
            "Файл модели не найден: {0}. Возможно, артефакты были удалены.".format(source)
        )

    selected_dir = root / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = selected_dir / "model.joblib.tmp"
    shutil.copy2(str(source), str(temporary_model))
    os.replace(str(temporary_model), str(selected_dir / "model.joblib"))
    metadata = {
        "selected_experiment_id": experiment_id,
        "selected_model": record["model"],
        "condition": record["condition"],
        "dataset_version": record["dataset_version"],
        "raw_fingerprint": record["raw_fingerprint"],
        "image_size": record["image_size"],
        "classes": record["classes"],
        "metrics": {
            "accuracy": record["accuracy"],
            "precision_weighted": record["precision_weighted"],
            "recall_weighted": record["recall_weighted"],
            "f1_weighted": record["f1_weighted"],
        },
        "created_at": record["created_at"],
    }
    with (selected_dir / "selected.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return metadata
