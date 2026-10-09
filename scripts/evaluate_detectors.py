"""Rebuild detector metrics from saved predictions without retraining."""

import argparse
import json
import math
import random
from pathlib import Path


def binary_metrics(labels, probabilities, threshold=0.5):
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    positives = [p for y, p in zip(labels, probabilities) if y == 1]
    negatives = [p for y, p in zip(labels, probabilities) if y == 0]
    wins = sum(a > b for a in positives for b in negatives)
    ties = sum(a == b for a in positives for b in negatives)
    auroc = (wins + 0.5 * ties) / max(len(positives) * len(negatives), 1)
    ece = 0.0
    for start in (value / 10 for value in range(10)):
        bucket = [(y, p) for y, p in zip(labels, probabilities) if start <= p < start + 0.1 or start == 0.9 and p == 1]
        if bucket:
            ece += len(bucket) / len(labels) * abs(sum(p for _, p in bucket) / len(bucket) -
                                                    sum(y for y, _ in bucket) / len(bucket))
    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc, "ece": ece}


def bootstrap(labels, probabilities, repeats=1000, seed=42):
    rng, values = random.Random(seed), []
    for _ in range(repeats):
        sample = [rng.randrange(len(labels)) for _ in labels]
        sample_labels = [labels[index] for index in sample]
        if len(set(sample_labels)) < 2:
            continue
        values.append(binary_metrics(sample_labels, [probabilities[index] for index in sample]))
    result = {}
    for metric in ("precision", "recall", "f1", "auroc", "ece"):
        ordered = sorted(item[metric] for item in values)
        result[metric] = {"mean": sum(ordered) / len(ordered),
                          "lower": ordered[math.floor(0.025 * (len(ordered) - 1))],
                          "upper": ordered[math.floor(0.975 * (len(ordered) - 1))]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", default="outputs/probes")
    parser.add_argument("--output", default="outputs/results/detector_evaluation.json")
    args = parser.parse_args()
    probe_dir, detectors = Path(args.probe_dir), {}
    for name in ("entropy-only", "logistic", "mlp"):
        path = probe_dir / f"predictions_{name}.json"
        if not path.exists():
            continue
        rows = [row for row in json.loads(path.read_text(encoding="utf-8")) if row.get("split") == "test"]
        labels, probabilities = [row["label"] for row in rows], [row["probability"] for row in rows]
        detectors[name] = {"status": "evaluated", "samples": len(rows),
                           "metrics": binary_metrics(labels, probabilities),
                           "confidence_intervals": bootstrap(labels, probabilities)}
    for name in ("lexical-live", "hybrid-live", "selfcheckgpt"):
        detectors[name] = {"status": "not_evaluated", "reason":
                           "No aligned labeled prediction artifact is available; no score is fabricated."}
    output = {"schema_version": "detector-eval-v1", "detectors": detectors,
              "note": "Threshold-independent and calibration metrics are rebuilt from saved held-out predictions."}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
