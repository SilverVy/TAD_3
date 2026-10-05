import tempfile
import unittest
from pathlib import Path

from butterfly_ml.data import collect_samples, generate_demo_dataset
from butterfly_ml.pipeline import run_pipeline
from butterfly_ml.tracking import load_history, select_experiment


class PipelineTests(unittest.TestCase):
    def test_demo_generation_and_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "images"
            count = generate_demo_dataset(str(data_dir), samples_per_class=4, seed=17)
            samples, classes, fingerprint = collect_samples(str(data_dir))

            self.assertEqual(count, 12)
            self.assertEqual(len(samples), 12)
            self.assertEqual(len(classes), 3)
            self.assertEqual(len(fingerprint), 64)

    def test_full_run_records_six_models_and_selects_an_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "images"
            artifact_dir = root / "artifacts"
            processed_dir = root / "processed"
            generate_demo_dataset(str(data_dir), samples_per_class=8, seed=9)

            records = run_pipeline(
                data_dir=str(data_dir),
                processed_dir=str(processed_dir),
                artifact_dir=str(artifact_dir),
                test_size=0.25,
                augment_factor=1,
                image_size=8,
                seed=9,
            )
            history = load_history(str(artifact_dir))

            self.assertEqual(len(records), 6)
            self.assertEqual(len(history), 6)
            self.assertEqual(
                {record["model"] for record in records},
                {"LogisticRegression", "RandomForest", "SVM_RBF"},
            )
            self.assertEqual(
                {record["condition"] for record in records},
                {"original_only", "original_plus_augmentation"},
            )
            self.assertTrue(
                all(Path(record["model_path"]).is_file() for record in records)
            )
            self.assertTrue(all(len(record["model_sha256"]) == 64 for record in records))

            selected = select_experiment(
                str(artifact_dir), records[0]["experiment_id"]
            )
            self.assertEqual(
                selected["selected_experiment_id"], records[0]["experiment_id"]
            )
            self.assertTrue((artifact_dir / "selected" / "model.joblib").is_file())

    def test_invalid_parameters_fail_explicitly(self):
        with self.assertRaises(ValueError):
            run_pipeline(data_dir="does-not-matter", test_size=0.8)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                generate_demo_dataset(temporary, samples_per_class=2, seed=1)


if __name__ == "__main__":
    unittest.main()
