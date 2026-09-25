"""
RAGTruth Dataset Download, Parsing, and Subsetting.

Downloads RAGTruth from GitHub, merges source contexts with response labels,
creates a stratified subset of ~500-1000 examples balanced by task type,
model, and hallucination label.

Sources:
    - GitHub: https://github.com/ParticleMedia/RAGTruth
    - Paper: RAGTruth: A Hallucination Corpus (ACL 2024 Findings)
"""

import json
import urllib.request
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger


# ── URLs ────────────────────────────────────────────────────────────────────
RAGTRUTH_SOURCE_URL = (
    "https://github.com/ParticleMedia/RAGTruth/raw/refs/heads/main/dataset/source_info.jsonl"
)
RAGTRUTH_RESPONSE_URL = (
    "https://github.com/ParticleMedia/RAGTruth/raw/refs/heads/main/dataset/response.jsonl"
)


def download_file(url: str, dest_path: str, force: bool = False) -> str:
    """Download a file from URL to local path.

    Args:
        url: Source URL.
        dest_path: Local destination path.
        force: If True, re-download even if file exists.

    Returns:
        Path to downloaded file.
    """
    dest = Path(dest_path)
    if dest.exists() and not force:
        logger.info(f"File already exists: {dest}")
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {dest}")

    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.info(f"Downloaded successfully: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

    return str(dest)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")
    logger.info(f"Loaded {len(records)} records from {filepath}")
    return records


def download_ragtruth(
    output_dir: str = "data/raw",
    force: bool = False,
) -> Tuple[str, str]:
    """Download RAGTruth dataset files.

    Args:
        output_dir: Directory to save downloaded files.
        force: Re-download even if files exist.

    Returns:
        Tuple of (source_info_path, response_path).
    """
    output_dir = Path(output_dir)
    source_path = download_file(
        RAGTRUTH_SOURCE_URL, str(output_dir / "source_info.jsonl"), force=force
    )
    response_path = download_file(
        RAGTRUTH_RESPONSE_URL, str(output_dir / "response.jsonl"), force=force
    )
    return source_path, response_path


def merge_ragtruth(
    source_path: str,
    response_path: str,
) -> List[Dict[str, Any]]:
    """Merge source contexts with response labels.

    Creates a unified dataset where each record has:
    - question/prompt, context, response, hallucination labels
    - Binary is_hallucinated flag
    - Task type and model metadata

    Args:
        source_path: Path to source_info.jsonl.
        response_path: Path to response.jsonl.

    Returns:
        List of merged records.
    """
    sources = load_jsonl(source_path)
    responses = load_jsonl(response_path)

    # Build source lookup
    source_map = {}
    for src in sources:
        source_map[src["source_id"]] = src

    # Merge
    merged = []
    missing_sources = 0

    for resp in responses:
        source_id = resp.get("source_id")
        if source_id not in source_map:
            missing_sources += 1
            continue

        src = source_map[source_id]
        labels = resp.get("labels", [])

        # Handle source_info polymorphism (string for Summary, dict for QA/Data2txt)
        context = src.get("source_info", "")
        if isinstance(context, dict):
            # Convert structured context to readable string
            context = json.dumps(context, indent=2, ensure_ascii=False)

        # Build unified record
        record = {
            "id": f"{source_id}__{resp.get('model', 'unknown')}__{resp.get('temperature', 0)}",
            "source_id": source_id,
            "task_type": src.get("task_type", "unknown"),
            "source_dataset": src.get("source", "unknown"),
            "prompt": src.get("prompt", ""),
            "context": context,
            "question": src.get("prompt", ""),  # Prompt serves as the question
            "response": resp.get("response", ""),
            "model": resp.get("model", "unknown"),
            "temperature": resp.get("temperature", 0.0),
            "split": resp.get("split", "train"),
            # Labels
            "labels": labels,
            "num_hallucinations": len(labels),
            "is_hallucinated": len(labels) > 0,
            "hallucination_types": list(set(
                lbl.get("label_type", "unknown") for lbl in labels
            )) if labels else [],
            # Metadata flags
            "has_implicit_true": any(
                lbl.get("implicit_true", False) for lbl in labels
            ),
            "has_due_to_null": any(
                lbl.get("due_to_null", False) for lbl in labels
            ),
        }
        merged.append(record)

    if missing_sources > 0:
        logger.warning(f"{missing_sources} responses had no matching source context")

    # Stats
    total = len(merged)
    hallucinated = sum(1 for r in merged if r["is_hallucinated"])
    logger.info(
        f"Merged dataset: {total} total examples, "
        f"{hallucinated} hallucinated ({100*hallucinated/total:.1f}%), "
        f"{total - hallucinated} grounded ({100*(total-hallucinated)/total:.1f}%)"
    )

    return merged


def save_dataset(
    records: List[Dict[str, Any]],
    output_path: str,
    format: str = "jsonl",
):
    """Save dataset records to file.

    Args:
        records: List of record dicts.
        output_path: Output file path.
        format: "jsonl" or "json".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved {len(records)} records to {output_path}")


def main():
    """Download and prepare RAGTruth dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Download RAGTruth dataset")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    # Download
    source_path, response_path = download_ragtruth(args.output_dir, force=args.force)

    # Merge
    merged = merge_ragtruth(source_path, response_path)

    # Save merged dataset
    save_dataset(merged, f"{args.output_dir}/ragtruth_merged.jsonl")

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"RAGTruth Dataset Summary")
    print(f"{'='*60}")
    print(f"Total examples: {len(merged)}")

    # By task type
    from collections import Counter
    task_counts = Counter(r["task_type"] for r in merged)
    print(f"\nBy task type:")
    for task, count in task_counts.most_common():
        hal = sum(1 for r in merged if r["task_type"] == task and r["is_hallucinated"])
        print(f"  {task}: {count} ({hal} hallucinated, {100*hal/count:.1f}%)")

    # By model
    model_counts = Counter(r["model"] for r in merged)
    print(f"\nBy model:")
    for model, count in model_counts.most_common():
        hal = sum(1 for r in merged if r["model"] == model and r["is_hallucinated"])
        print(f"  {model}: {count} ({hal} hallucinated, {100*hal/count:.1f}%)")


if __name__ == "__main__":
    main()
