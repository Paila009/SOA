"""
Script: Run Activation Patching Experiments.

Performs causal validation by swapping internal activations between
grounded and hallucinating generations.

Usage:
    python scripts/run_patching.py --model qwen2.5-1.5b --num-pairs 50
"""

import argparse
import json
import sys
import random
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.loader import ModelLoader, MODEL_REGISTRY
from causal.patching import ActivationPatcher
from data.download import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Run activation patching experiments")
    parser.add_argument(
        "--model", type=str, default="qwen2.5-1.5b",
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--input", type=str, default="data/processed/test.jsonl",
        help="Dataset file with grounded and hallucinated examples",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/causal",
    )
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--flip-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    logger.add("outputs/logs/patching_{time}.log")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    records = load_jsonl(args.input)

    # Separate grounded and hallucinated
    grounded = [r for r in records if not r["is_hallucinated"]]
    hallucinated = [r for r in records if r["is_hallucinated"]]

    logger.info(f"Found {len(grounded)} grounded, {len(hallucinated)} hallucinated examples")

    # Create matched pairs (same task type where possible)
    pairs = []
    for task_type in set(r["task_type"] for r in records):
        g_task = [r for r in grounded if r["task_type"] == task_type]
        h_task = [r for r in hallucinated if r["task_type"] == task_type]

        n_pairs = min(len(g_task), len(h_task), args.num_pairs // 3 + 1)
        random.shuffle(g_task)
        random.shuffle(h_task)

        for i in range(n_pairs):
            g = g_task[i]
            h = h_task[i]
            # Format as full prompts
            g_prompt = f"Context:\n{g['context'][:500]}\n\nQuestion: {g['question'][:200]}\n\nAnswer:"
            h_prompt = f"Context:\n{h['context'][:500]}\n\nQuestion: {h['question'][:200]}\n\nAnswer:"
            pairs.append((g_prompt, h_prompt))

    pairs = pairs[:args.num_pairs]
    logger.info(f"Created {len(pairs)} matched pairs for patching")

    # Setup patcher
    model_loader = ModelLoader(model_name=args.model, quant_type="4bit")
    patcher = ActivationPatcher(
        model_loader=model_loader,
        flip_threshold=args.flip_threshold,
    )

    # Run experiments
    results = patcher.run_batch_experiments(pairs)

    # Save results
    save_data = {
        "model": args.model,
        "num_pairs": len(pairs),
        "flip_threshold": args.flip_threshold,
        "flip_rates": results["flip_rates"],
        "overall_flip_rate": results["overall_flip_rate"],
        "total_experiments": results["total_experiments"],
    }

    results_path = output_dir / f"patching_results_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Activation Patching Results ({args.model})")
    print(f"{'='*60}")
    print(f"Total pairs tested: {len(pairs)}")
    print(f"Overall flip rate: {results['overall_flip_rate']:.1%}")
    print(f"\nPer-layer flip rates:")
    for layer, rate in sorted(results["flip_rates"].items()):
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  Layer {layer:3d}: {bar} {rate:.1%}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
