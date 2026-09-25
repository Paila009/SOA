"""Consolidate completed probe runs into one honest benchmark artifact."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build benchmark summary")
    parser.add_argument("--probe-dir", default="outputs/probes")
    parser.add_argument("--output-dir", default="outputs/results")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    probe_dir = Path(args.probe_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "Entropy baseline": "results_entropy-only.json",
        "Full logistic probe": "results_logistic.json",
        "MLP probe": "results_mlp.json",
    }
    methods = {}
    for display_name, filename in candidates.items():
        path = probe_dir / filename
        if path.exists():
            result = load_json(path)
            methods[display_name] = {
                **result["test"],
                "confidence_intervals": result.get("test_confidence_intervals", {}),
                "n_features": result.get("n_features"),
                "n_test": result.get("n_test"),
                "forward_passes": 1,
                "feature_description": result.get("feature_description"),
                "status": "completed",
            }
    pending = {
        "SelfCheckGPT": "Requires repeated Qwen generation and NLI scoring on the same 115 test IDs.",
        "Semantic entropy": "Requires stochastic Qwen samples and semantic clustering on the same test IDs.",
        "Activation patching": "Requires the Qwen base model and controlled matched-pair interventions.",
        "Gemma transfer": "Requires a second signal extraction run with architecture-aligned features.",
        "Activation steering": "Requires the Qwen base model, a validated target layer, and factuality review.",
    }
    best = max(methods, key=lambda name: methods[name]["auroc"]) if methods else None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_strategy": "explicit_dataset_ids",
        "methods": methods,
        "best_completed_method": best,
        "pending_experiments": pending,
        "interpretation": (
            "The full logistic probe improves ranking AUROC over entropy, but the gain is modest. "
            "The MLP does not improve on the simpler models. Causal, transfer, and mitigation claims "
            "remain unverified until their dedicated inference runs finish."
        ),
    }
    destination = output_dir / "benchmark_results.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Completed benchmark methods")
    for name, metrics in methods.items():
        print(f"  {name:22s} AUROC={metrics['auroc']:.4f} F1={metrics['f1']:.4f}")
    print(f"Best completed method: {best}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
