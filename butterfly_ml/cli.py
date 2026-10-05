import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from butterfly_ml.data import collect_samples, extract_features, generate_demo_dataset
from butterfly_ml.pipeline import run_pipeline
from butterfly_ml.tracking import load_history, select_experiment


def _add_run_options(parser, include_data_dir):
    if include_data_dir:
        parser.add_argument(
            "--data-dir",
            default="data/raw/butterflies",
            help="Каталог: отдельная подпапка на каждый класс.",
        )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--augment-factor", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)


def build_parser():
    parser = argparse.ArgumentParser(
        description="MLOps-пайплайн классификации видов бабочек."
    )
    commands = parser.add_subparsers(dest="command")
    commands.required = True

    demo = commands.add_parser(
        "demo", help="Создать демонстрационные данные и провести 6 экспериментов."
    )
    demo.add_argument("--data-dir", default="data/raw/demo_butterflies")
    demo.add_argument("--samples-per-class", type=int, default=24)
    _add_run_options(demo, include_data_dir=False)

    run = commands.add_parser(
        "run", help="Обучить три модели на исходных и на расширенных данных."
    )
    _add_run_options(run, include_data_dir=True)

    history = commands.add_parser(
        "history", help="Показать историю экспериментов и метрики."
    )
    history.add_argument("--artifacts-dir", default="artifacts")
    history.add_argument("--limit", type=int, default=20)

    select = commands.add_parser(
        "select", help="Выбрать эксперимент из истории для последующего предсказания."
    )
    select.add_argument("--experiment-id", required=True)
    select.add_argument("--artifacts-dir", default="artifacts")

    predict = commands.add_parser(
        "predict", help="Классифицировать изображение выбранной моделью."
    )
    predict.add_argument("--image", required=True)
    predict.add_argument("--artifacts-dir", default="artifacts")

    info = commands.add_parser(
        "dataset-info", help="Проверить число файлов и классов исходного датасета."
    )
    info.add_argument("--data-dir", default="data/raw/butterflies")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            demo_path = Path(args.data_dir)
            if not demo_path.exists() or not any(demo_path.iterdir()):
                count = generate_demo_dataset(
                    str(demo_path), args.samples_per_class, args.seed
                )
                print(
                    "Создано демонстрационных изображений: {0} ({1}).".format(
                        count, demo_path
                    )
                )
            else:
                print(
                    "Используется существующий датасет без перезаписи: {0}.".format(
                        demo_path
                    )
                )
            _run(args)
        elif args.command == "run":
            _run(args)
        elif args.command == "history":
            if args.limit < 1 or args.limit > 1000:
                raise ValueError("Параметр --limit должен быть от 1 до 1000.")
            _print_history(args.artifacts_dir, args.limit)
        elif args.command == "select":
            metadata = select_experiment(args.artifacts_dir, args.experiment_id)
            print(
                "Выбрана модель {0} из эксперимента {1}.".format(
                    metadata["selected_model"], metadata["selected_experiment_id"]
                )
            )
            print(
                "Артефакт сохранен в {0}.".format(
                    Path(args.artifacts_dir) / "selected"
                )
            )
        elif args.command == "predict":
            _predict(args.image, args.artifacts_dir)
        elif args.command == "dataset-info":
            samples, classes, fingerprint = collect_samples(args.data_dir)
            print("Классы: {0}".format(", ".join(classes)))
            print("Изображений: {0}".format(len(samples)))
            print("SHA-256 набора данных: {0}".format(fingerprint))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print("Ошибка: {0}".format(error), file=sys.stderr)
        return 2


def _run(args):
    records = run_pipeline(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        artifact_dir=args.artifacts_dir,
        test_size=args.test_size,
        augment_factor=args.augment_factor,
        image_size=args.image_size,
        seed=args.seed,
    )
    print("\nЭксперименты завершены. Результаты:")
    for record in records:
        print(
            "{0} | {1:30s} | {2:28s} | accuracy={3:.3f} | F1={4:.3f}".format(
                record["experiment_id"],
                record["condition"],
                record["model"],
                record["accuracy"],
                record["f1_weighted"],
            )
        )
    print("Полная история: {0}".format(Path(args.artifacts_dir) / "experiments.jsonl"))
    print("Таблица результатов: {0}".format(Path(args.artifacts_dir) / "results.csv"))


def _print_history(artifacts_dir, limit):
    records = load_history(artifacts_dir)
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    if not records:
        print("История пока пуста. Сначала запустите команду run или demo.")
        return
    print(
        "{0:34s} {1:28s} {2:20s} {3:>9s} {4:>9s}".format(
            "EXPERIMENT ID", "CONDITION", "MODEL", "ACCURACY", "F1"
        )
    )
    print("-" * 108)
    for record in records[:limit]:
        print(
            "{0:34s} {1:28s} {2:20s} {3:9.3f} {4:9.3f}".format(
                record["experiment_id"],
                record["condition"],
                record["model"],
                record["accuracy"],
                record["f1_weighted"],
            )
        )


def _predict(image_path, artifacts_dir):
    selected_dir = Path(artifacts_dir) / "selected"
    model_path = selected_dir / "model.joblib"
    metadata_path = selected_dir / "selected.json"
    if not model_path.exists() or not metadata_path.exists():
        raise ValueError(
            "Сначала выберите эксперимент командой select --experiment-id <ID>."
        )
    image = Path(image_path)
    if not image.is_file():
        raise ValueError("Изображение не найдено: {0}".format(image))
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    vector = extract_features(str(image), int(metadata["image_size"]))
    estimator = joblib.load(str(model_path))
    predicted = estimator.predict(np.asarray([vector]))[0]
    print("Предсказанный класс: {0}".format(predicted))
    probability_setting = getattr(estimator, "probability", True)
    if hasattr(estimator, "predict_proba") and probability_setting is True:
        probabilities = estimator.predict_proba(np.asarray([vector]))[0]
        labels = estimator.classes_
        confidence = float(max(probabilities))
        print("Уверенность модели: {0:.1%}".format(confidence))
        print(
            "Вероятности: {0}".format(
                ", ".join(
                    "{0}={1:.1%}".format(label, probability)
                    for label, probability in sorted(
                        zip(labels, probabilities),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
            )
        )


if __name__ == "__main__":
    sys.exit(main())
