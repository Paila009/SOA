"""
Script: Run Signal Extraction Pipeline.

Extracts internal model signals (attention, hidden states, entropy) from
the base language model over the RAG dataset. Saves signals to disk for
subsequent probe training.

Usage:
    python scripts/run_extraction.py --model qwen2.5-1.5b --subset-size 750
    python scripts/run_extraction.py --model gemma-2b --input data/processed/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from models.loader import ModelLoader, MODEL_REGISTRY
from extraction.extractor import SignalExtractor
from data.download import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Run signal extraction pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-1.5b",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use for extraction",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/subset_full.jsonl",
        help="Input dataset file (JSONL format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for signals (default: outputs/signals/<model>)",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization type",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per example",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="strategic",
        help="Layer indices to extract ('strategic' or comma-separated: '0,7,14,21,27')",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume extraction from this example index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Skip attention weight extraction (saves memory)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    logger.add("outputs/logs/extraction_{time}.log", rotation="100 MB")

    # Setup output dir
    if args.output_dir is None:
        args.output_dir = f"outputs/signals/{args.model}"

    # Parse layer indices
    if args.layers == "strategic":
        layer_indices = None  # Will auto-select
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    # Load model
    logger.info(f"Setting up model: {args.model} ({args.quant} quantization)")
    model_loader = ModelLoader(
        model_name=args.model,
        quant_type=args.quant,
    )

    # Load dataset
    logger.info(f"Loading dataset from {args.input}")
    dataset = load_jsonl(args.input)

    if args.limit is not None:
        dataset = dataset[:args.limit]
        logger.info(f"Limited to {len(dataset)} examples")

    # Create extractor
    extractor = SignalExtractor(
        model_loader=model_loader,
        layer_indices=layer_indices,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        capture_attention=not args.no_attention,
    )

    # Run extraction
    logger.info(f"Starting extraction: {len(dataset)} examples -> {args.output_dir}")
    output_files = extractor.run(
        dataset=dataset,
        resume_from=args.resume_from,
    )

    logger.info(f"Extraction complete! {len(output_files)} signal files saved.")
    print(f"\n[OK] Extracted signals for {len(output_files)} examples")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
