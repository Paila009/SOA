"""
Gold Subset Verification Tool.

Provides utilities for manual verification of the gold subset labels.
Outputs a review form (CSV/JSON) that team members can fill in, and
computes inter-annotator agreement metrics.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


def create_review_form(
    gold_records: List[Dict[str, Any]],
    output_path: str,
    format: str = "csv",
    max_context_chars: int = 500,
    max_response_chars: int = 500,
):
    """Create a human-readable review form from gold subset records.

    Generates a spreadsheet/document that reviewers can annotate with:
    - Their own grounded/hallucinated judgment
    - Confidence level (1-5)
    - Notes on disagreements with original label

    Args:
        gold_records: Gold subset records.
        output_path: Output file path.
        format: "csv" or "json".
        max_context_chars: Max characters of context to show.
        max_response_chars: Max characters of response to show.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "example_id",
                "task_type",
                "original_label",
                "hallucination_types",
                "context_preview",
                "response_preview",
                "hallucination_spans",
                # Reviewer fills these in:
                "reviewer_label",
                "reviewer_confidence_1_5",
                "reviewer_notes",
            ])

            for record in gold_records:
                # Format hallucination spans for display
                spans = []
                for lbl in record.get("labels", []):
                    span_text = lbl.get("text", "")[:100]
                    span_type = lbl.get("label_type", "unknown")
                    spans.append(f"[{span_type}] {span_text}")

                writer.writerow([
                    record.get("id", ""),
                    record.get("task_type", ""),
                    "hallucinated" if record.get("is_hallucinated") else "grounded",
                    "; ".join(record.get("hallucination_types", [])),
                    record.get("context", "")[:max_context_chars],
                    record.get("response", "")[:max_response_chars],
                    " || ".join(spans) if spans else "NONE",
                    "",  # reviewer_label
                    "",  # reviewer_confidence
                    "",  # reviewer_notes
                ])

    elif format == "json":
        review_items = []
        for record in gold_records:
            item = {
                "example_id": record.get("id", ""),
                "task_type": record.get("task_type", ""),
                "original_label": "hallucinated" if record.get("is_hallucinated") else "grounded",
                "hallucination_types": record.get("hallucination_types", []),
                "context": record.get("context", "")[:max_context_chars],
                "response": record.get("response", "")[:max_response_chars],
                "hallucination_spans": [
                    {"text": lbl.get("text", ""), "type": lbl.get("label_type", "")}
                    for lbl in record.get("labels", [])
                ],
                # Reviewer fills these:
                "reviewer_label": None,
                "reviewer_confidence": None,
                "reviewer_notes": "",
            }
            review_items.append(item)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(review_items, f, indent=2, ensure_ascii=False)

    logger.info(f"Created review form with {len(gold_records)} examples at {output_path}")


def load_review_results(review_path: str) -> List[Dict[str, Any]]:
    """Load completed review form.

    Args:
        review_path: Path to completed review form (CSV or JSON).

    Returns:
        List of review records with reviewer annotations.
    """
    path = Path(review_path)

    if path.suffix == ".csv":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def compute_agreement_metrics(
    reviews: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute agreement between original labels and reviewer labels.

    Args:
        reviews: List of review records with both original and reviewer labels.

    Returns:
        Dict with agreement metrics.
    """
    agreed = 0
    disagreed = 0
    skipped = 0

    for review in reviews:
        original = review.get("original_label", "")
        reviewer = review.get("reviewer_label", "")

        if not reviewer or reviewer.strip() == "":
            skipped += 1
            continue

        # Normalize labels
        orig_is_hal = original.lower() in ["hallucinated", "1", "true", "yes"]
        rev_is_hal = reviewer.lower() in ["hallucinated", "1", "true", "yes"]

        if orig_is_hal == rev_is_hal:
            agreed += 1
        else:
            disagreed += 1

    total = agreed + disagreed
    agreement_rate = agreed / max(total, 1)

    metrics = {
        "total_reviewed": total,
        "agreed": agreed,
        "disagreed": disagreed,
        "skipped": skipped,
        "agreement_rate": agreement_rate,
    }

    logger.info(
        f"Label agreement: {agreement_rate:.1%} "
        f"({agreed}/{total} agreed, {disagreed} disagreed, {skipped} skipped)"
    )

    return metrics


def main():
    """Generate review form from gold subset."""
    import argparse

    parser = argparse.ArgumentParser(description="Gold subset verification tool")
    parser.add_argument("--input", default="data/processed/gold_subset.jsonl", help="Gold subset file")
    parser.add_argument("--output", default="data/gold/review_form.csv", help="Review form output")
    parser.add_argument("--format", default="csv", choices=["csv", "json"], help="Output format")
    args = parser.parse_args()

    from data.download import load_jsonl
    records = load_jsonl(args.input)
    create_review_form(records, args.output, format=args.format)
    print(f"✓ Review form created at {args.output} ({len(records)} examples)")


if __name__ == "__main__":
    main()
