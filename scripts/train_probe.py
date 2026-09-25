"""Train leak-free hallucination probes on explicit dataset splits."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import bootstrap_confidence_intervals
from extraction.signals import SignalProcessor
from probes.probe_classifiers import EntropyBaseline, LogisticProbe, MLPProbe, extract_entropy_features


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination probe")
    parser.add_argument("--type", default="entropy-only", choices=["entropy-only", "logistic", "mlp"])
    parser.add_argument("--signal-dir", default="outputs/signals/qwen2.5-1.5b")
    parser.add_argument("--output-dir", default="outputs/probes")
    parser.add_argument("--train-split", default="data/processed/train.jsonl")
    parser.add_argument("--val-split", default="data/processed/val.jsonl")
    parser.add_argument("--test-split", default="data/processed/test.jsonl")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dims", default="128,64")
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_signal_index(signal_dir):
    index = {}
    for path in sorted(Path(signal_dir).glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        example_id = str(data.get("metadata", {}).get("example_id", path.stem))
        if example_id in index:
            raise ValueError(f"Duplicate signal example ID: {example_id}")
        index[example_id] = {"path": path, "data": data, "label": int(data["label"])}
    if not index:
        raise FileNotFoundError(f"No signal files found in {signal_dir}")
    return index


def select_split(signal_index, split_path):
    selected = []
    missing = []
    for record in read_jsonl(split_path):
        example_id = str(record["id"])
        item = signal_index.get(example_id)
        if item is None:
            missing.append(example_id)
        else:
            selected.append({**item, "id": example_id, "record": record})
    if missing:
        raise ValueError(
            f"{len(missing)} IDs from {split_path} have no signal file: "
            + ", ".join(missing[:5])
        )
    return selected


def validate_splits(splits):
    id_sets = {name: {item["id"] for item in items} for name, items in splits.items()}
    names = list(id_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise ValueError(f"Split leakage between {left} and {right}: {len(overlap)} IDs")
    for name, items in splits.items():
        if len({item["label"] for item in items}) < 2:
            raise ValueError(f"{name} split does not contain both classes")


def split_labels(items):
    return np.asarray([item["label"] for item in items], dtype=np.int64)


def build_features(splits, probe_type, output_dir, pca_dim):
    if probe_type == "entropy-only":
        matrices = {
            name: np.stack([extract_entropy_features(item["data"]) for item in items])
            for name, items in splits.items()
        }
        return matrices, None, "entropy_and_token_probability_statistics"

    processor = SignalProcessor(pca_dim=pca_dim)
    processor.fit([str(item["path"]) for item in splits["train"]])
    matrices = {
        name: np.stack([processor.process_single(item["data"]) for item in items])
        for name, items in splits.items()
    }
    feature_counts = {matrix.shape[1] for matrix in matrices.values()}
    if len(feature_counts) != 1:
        raise ValueError(f"Inconsistent feature dimensions: {sorted(feature_counts)}")
    processor.save(str(output_dir / "signal_processor.joblib"))
    layers = sorted(processor.pca_models)
    description = f"11_uncertainty_statistics_plus_{pca_dim}_pca_components_x_{len(layers)}_layers"
    return matrices, processor, description


def make_prediction_rows(items, probabilities, split_name):
    rows = []
    for item, probability in zip(items, probabilities):
        record = item["record"]
        rows.append({
            "id": item["id"],
            "split": split_name,
            "label": item["label"],
            "probability": float(probability),
            "prediction": int(probability >= 0.5),
            "task_type": record.get("task_type", "Unknown"),
            "model": record.get("model", "Unknown"),
            "question": record.get("question", "")[:500],
            "response": record.get("response", "")[:700],
        })
    return rows


def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    logger.add("outputs/logs/probe_training_{time}.log")

    signal_index = load_signal_index(args.signal_dir)
    splits = {
        "train": select_split(signal_index, args.train_split),
        "val": select_split(signal_index, args.val_split),
        "test": select_split(signal_index, args.test_split),
    }
    validate_splits(splits)
    y = {name: split_labels(items) for name, items in splits.items()}
    logger.info("Explicit ID split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    X, processor, feature_description = build_features(splits, args.type, output_dir, args.pca_dim)
    n_features = X["train"].shape[1]
    logger.info(f"Verified feature matrix: {n_features} columns")

    if args.type == "entropy-only":
        probe = EntropyBaseline()
        model_path = output_dir / "entropy_baseline.joblib"
        probe.fit(X["train"], y["train"])
        history = None
    elif args.type == "logistic":
        probe = LogisticProbe()
        model_path = output_dir / "logistic_probe.joblib"
        probe.fit(X["train"], y["train"])
        history = None
    else:
        hidden_dims = [int(value) for value in args.hidden_dims.split(",")]
        probe = MLPProbe(
            hidden_dims=hidden_dims,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        model_path = output_dir / "mlp_probe.pt"
        history = probe.fit(X["train"], y["train"], X["val"], y["val"])

    val_metrics = probe.evaluate(X["val"], y["val"])
    test_metrics = probe.evaluate(X["test"], y["test"])
    val_probs = probe.predict_proba(X["val"])
    test_probs = probe.predict_proba(X["test"])
    probe.save(str(model_path))

    results = {
        "type": args.type,
        "status": "completed",
        "split_strategy": "explicit_dataset_ids",
        "feature_description": feature_description,
        "n_features": n_features,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "class_counts": {
            name: {"grounded": int((values == 0).sum()), "hallucinated": int((values == 1).sum())}
            for name, values in y.items()
        },
        "val": val_metrics,
        "test": test_metrics,
        "test_confidence_intervals": bootstrap_confidence_intervals(
            y["test"], test_probs, n_bootstrap=1000
        ),
        "model_path": str(model_path),
        "processor_path": str(output_dir / "signal_processor.joblib") if processor else None,
    }
    if history is not None:
        results["epochs_trained"] = len(history["train_loss"])
        results["training_history"] = history

    result_path = output_dir / f"results_{args.type}.json"
    prediction_path = output_dir / f"predictions_{args.type}.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    prediction_rows = make_prediction_rows(splits["val"], val_probs, "val")
    prediction_rows += make_prediction_rows(splits["test"], test_probs, "test")
    prediction_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")

    print(f"Probe: {args.type}")
    print(f"Features: {n_features} ({feature_description})")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
