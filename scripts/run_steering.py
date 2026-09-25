"""
Script: Run Activation Steering Experiments.

Tests hallucination mitigation by applying activation steering during
generation. Compares steered vs unsteered outputs.

Usage:
    python scripts/run_steering.py --model qwen2.5-1.5b --alpha 2.0
"""

import argparse
import json
import sys
import random
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.loader import ModelLoader, MODEL_REGISTRY
from causal.steering import ActivationSteerer
from data.download import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Run activation steering experiments")
    parser.add_argument("--model", type=str, default="qwen2.5-1.5b",
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--input", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/causal")
    parser.add_argument("--alpha", type=float, nargs="+",
                       default=[0.5, 1.0, 2.0, 3.0, 5.0],
                       help="Steering strength values to test")
    parser.add_argument("--num-direction-examples", type=int, default=50,
                       help="Number of examples for computing steering direction")
    parser.add_argument("--num-test-examples", type=int, default=20,
                       help="Number of examples to test steering on")
    parser.add_argument("--target-layer", type=int, default=None,
                       help="Layer to apply steering at (default: 75%% depth)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    logger.add("outputs/logs/steering_{time}.log")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    records = load_jsonl(args.input)
    grounded = [r for r in records if not r["is_hallucinated"]]
    hallucinated = [r for r in records if r["is_hallucinated"]]

    logger.info(f"Dataset: {len(grounded)} grounded, {len(hallucinated)} hallucinated")

    # Setup steerer
    model_loader = ModelLoader(model_name=args.model, quant_type="4bit")
    steerer = ActivationSteerer(
        model_loader=model_loader,
        target_layer=args.target_layer,
    )

    # Prepare texts for direction computation
    n_dir = min(args.num_direction_examples, len(grounded), len(hallucinated))
    random.shuffle(grounded)
    random.shuffle(hallucinated)

    grounded_texts = []
    for r in grounded[:n_dir]:
        prompt = model_loader.format_prompt(r["question"], r["context"][:500])
        grounded_texts.append(prompt)

    hallucinated_texts = []
    for r in hallucinated[:n_dir]:
        prompt = model_loader.format_prompt(r["question"], r["context"][:500])
        hallucinated_texts.append(prompt)

    # Compute steering direction
    logger.info(f"Computing steering direction from {n_dir} pairs...")
    steerer.compute_direction(grounded_texts, hallucinated_texts)

    # Test steering on hallucinated examples
    test_examples = hallucinated[n_dir:n_dir + args.num_test_examples]
    results = []

    for i, example in enumerate(test_examples):
        prompt = model_loader.format_prompt(example["question"], example["context"][:500])
        logger.info(f"Testing example {i+1}/{len(test_examples)}")

        # Generate with alpha sweep
        try:
            alpha_outputs = steerer.sweep_alpha(
                prompt=prompt,
                alpha_values=[0.0] + args.alpha,
                max_new_tokens=128,
            )

            result = {
                "example_id": example.get("id", f"test_{i}"),
                "question": example["question"][:200],
                "original_response": example["response"][:300],
                "outputs_by_alpha": {str(k): v for k, v in alpha_outputs.items()},
            }
            results.append(result)
        except Exception as e:
            logger.error(f"Error on example {i}: {e}")
            continue

    # Save results
    results_path = output_dir / f"steering_results_{args.model}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Activation Steering Results ({args.model})")
    print(f"{'='*60}")
    print(f"Target layer: {steerer.target_layer}")
    print(f"Alpha values tested: {[0.0] + args.alpha}")
    print(f"Examples tested: {len(results)}")
    print(f"\nResults saved to: {results_path}")

    # Show sample comparison
    if results:
        sample = results[0]
        print(f"\n--- Sample Comparison ---")
        print(f"Question: {sample['question'][:100]}...")
        for alpha, text in sample["outputs_by_alpha"].items():
            print(f"\n  Alpha={alpha}: {text[:150]}...")


if __name__ == "__main__":
    main()
