"""
Stratified Dataset Subsetting.

Creates balanced subsets of the RAGTruth dataset for training and evaluation,
preserving the distribution across task types, source models, and hallucination
labels. Also handles train/validation/test splitting at the source_id level
to prevent context leakage.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from loguru import logger


def create_stratified_subset(
    records: List[Dict[str, Any]],
    target_size: int = 750,
    stratify_keys: List[str] = None,
    balance_hallucination: bool = True,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create a stratified subset of the dataset.

    Stratifies by task_type and hallucination status to ensure the subset
    is representative. Optionally enforces 50/50 balance between grounded
    and hallucinated examples.

    Args:
        records: Full dataset records.
        target_size: Desired subset size (~500-1000).
        stratify_keys: Keys to stratify by. Default: ["task_type"].
        balance_hallucination: If True, enforce ~50/50 grounded/hallucinated.
        random_seed: Random seed for reproducibility.

    Returns:
        Stratified subset of records.
    """
    if stratify_keys is None:
        stratify_keys = ["task_type"]

    random.seed(random_seed)

    # Group records by stratification key
    def get_stratum(record):
        parts = [str(record.get(k, "unknown")) for k in stratify_keys]
        parts.append(str(record["is_hallucinated"]))
        return "__".join(parts)

    strata = defaultdict(list)
    for record in records:
        strata[get_stratum(record)].append(record)

    # Calculate per-stratum sample size
    if balance_hallucination:
        # Split target evenly between hallucinated and grounded
        half_target = target_size // 2

        # Separate strata by hallucination status
        hal_strata = {k: v for k, v in strata.items() if k.endswith("__True")}
        ground_strata = {k: v for k, v in strata.items() if k.endswith("__False")}

        def sample_from_strata(strata_dict, total_target):
            """Sample proportionally from strata."""
            total_available = sum(len(v) for v in strata_dict.values())
            sampled = []
            for key, records_in_stratum in strata_dict.items():
                # Proportional allocation
                n_sample = max(1, round(len(records_in_stratum) / total_available * total_target))
                n_sample = min(n_sample, len(records_in_stratum))
                sampled.extend(random.sample(records_in_stratum, n_sample))
            return sampled

        subset = sample_from_strata(hal_strata, half_target)
        subset.extend(sample_from_strata(ground_strata, half_target))
    else:
        # Proportional sampling across all strata
        total_available = sum(len(v) for v in strata.values())
        subset = []
        for key, records_in_stratum in strata.items():
            n_sample = max(1, round(len(records_in_stratum) / total_available * target_size))
            n_sample = min(n_sample, len(records_in_stratum))
            subset.extend(random.sample(records_in_stratum, n_sample))

    # Shuffle final subset
    random.shuffle(subset)

    # Log distribution
    hal_count = sum(1 for r in subset if r["is_hallucinated"])
    ground_count = len(subset) - hal_count
    logger.info(
        f"Created subset: {len(subset)} examples "
        f"({hal_count} hallucinated, {ground_count} grounded)"
    )

    task_dist = Counter(r["task_type"] for r in subset)
    for task, count in task_dist.most_common():
        logger.info(f"  Task '{task}': {count} examples")

    return subset


def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_by: str = "source_id",
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into train/validation/test sets.

    IMPORTANT: Splits at the source_id level to prevent context leakage —
    all responses from the same source context go into the same split.

    Args:
        records: Dataset records.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        split_by: Key to group records by for splitting (prevents leakage).
        random_seed: Random seed.

    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(random_seed)

    # Group records by split key
    groups = defaultdict(list)
    for record in records:
        key = record.get(split_by, record.get("id", id(record)))
        groups[key].append(record)

    # Shuffle group keys
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    # Calculate split points
    n_groups = len(group_keys)
    train_end = int(n_groups * train_ratio)
    val_end = train_end + int(n_groups * val_ratio)

    # Assign groups to splits
    train_keys = set(group_keys[:train_end])
    val_keys = set(group_keys[train_end:val_end])
    test_keys = set(group_keys[val_end:])

    train_records = [r for r in records if r.get(split_by) in train_keys]
    val_records = [r for r in records if r.get(split_by) in val_keys]
    test_records = [r for r in records if r.get(split_by) in test_keys]

    logger.info(
        f"Split dataset ({split_by}-level): "
        f"train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
    )

    # Verify no leakage
    train_sources = set(r.get(split_by) for r in train_records)
    val_sources = set(r.get(split_by) for r in val_records)
    test_sources = set(r.get(split_by) for r in test_records)
    assert not (train_sources & val_sources), "Data leakage detected: train ∩ val"
    assert not (train_sources & test_sources), "Data leakage detected: train ∩ test"
    assert not (val_sources & test_sources), "Data leakage detected: val ∩ test"
    logger.info("✓ No data leakage across splits")

    return train_records, val_records, test_records


def select_gold_subset(
    records: List[Dict[str, Any]],
    target_size: int = 65,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select a gold subset for manual verification.

    Selects diverse examples across task types and hallucination types,
    prioritizing clear-cut examples that are easiest to verify manually.

    Args:
        records: Dataset records (typically from test split).
        target_size: Target gold subset size (~50-80).
        random_seed: Random seed.

    Returns:
        Gold subset records with an added "gold_verified" field.
    """
    random.seed(random_seed)

    # Separate by hallucination status
    hallucinated = [r for r in records if r["is_hallucinated"]]
    grounded = [r for r in records if not r["is_hallucinated"]]

    # For hallucinated examples, prefer those with "Evident" types (easier to verify)
    evident_hal = [
        r for r in hallucinated
        if any("Evident" in ht for ht in r.get("hallucination_types", []))
    ]
    subtle_hal = [
        r for r in hallucinated
        if not any("Evident" in ht for ht in r.get("hallucination_types", []))
    ]

    # Allocate: ~50% grounded, ~35% evident hallucination, ~15% subtle hallucination
    n_grounded = target_size // 2
    n_evident = int(target_size * 0.35)
    n_subtle = target_size - n_grounded - n_evident

    gold = []
    gold.extend(random.sample(grounded, min(n_grounded, len(grounded))))
    gold.extend(random.sample(evident_hal, min(n_evident, len(evident_hal))))
    gold.extend(random.sample(subtle_hal, min(n_subtle, len(subtle_hal))))

    # Add verification placeholder
    for record in gold:
        record["gold_verified"] = False
        record["gold_notes"] = ""

    random.shuffle(gold)
    logger.info(
        f"Selected {len(gold)} gold examples: "
        f"{sum(1 for r in gold if not r['is_hallucinated'])} grounded, "
        f"{sum(1 for r in gold if r['is_hallucinated'])} hallucinated"
    )

    return gold


def main():
    """Create stratified subset and splits from merged RAGTruth data."""
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset subset and splits")
    parser.add_argument("--input", default="data/raw/ragtruth_merged.jsonl", help="Merged dataset")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--subset-size", type=int, default=750, help="Target subset size")
    parser.add_argument("--gold-size", type=int, default=65, help="Gold subset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    from data.download import load_jsonl, save_dataset

    # Load merged data
    records = load_jsonl(args.input)

    # Create subset
    subset = create_stratified_subset(
        records, target_size=args.subset_size, random_seed=args.seed
    )

    # Split into train/val/test
    train, val, test = split_dataset(subset, random_seed=args.seed)

    # Select gold subset from test
    gold = select_gold_subset(test, target_size=args.gold_size, random_seed=args.seed)

    # Save everything
    output_dir = Path(args.output_dir)
    save_dataset(subset, str(output_dir / "subset_full.jsonl"))
    save_dataset(train, str(output_dir / "train.jsonl"))
    save_dataset(val, str(output_dir / "val.jsonl"))
    save_dataset(test, str(output_dir / "test.jsonl"))
    save_dataset(gold, str(output_dir / "gold_subset.jsonl"))

    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset Preparation Complete")
    print(f"{'='*60}")
    print(f"Full subset:  {len(subset)} examples")
    print(f"  Train:      {len(train)}")
    print(f"  Validation: {len(val)}")
    print(f"  Test:       {len(test)}")
    print(f"  Gold:       {len(gold)} (from test, for manual verification)")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
