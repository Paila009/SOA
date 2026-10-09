"""Reject training labels that belong to a different answer."""


def validate_response_labels(splits):
    for split, items in splits.items():
        for item in items:
            signal, record = item['data'], item['record']
            generated = ' '.join(str(signal.get('generated_text', '')).split())
            annotated = ' '.join(str(record.get('response', '')).split())
            if not generated or not annotated or generated != annotated:
                raise ValueError(
                    f"Response-label mismatch in {split}, example {item['id']}. "
                    "Extract signals on the exact annotated answer or obtain labels "
                    "for the generated answer; inherited labels are not valid.")
            expected = int(record.get('is_hallucinated', record.get('label', -1)))
            if expected not in (0, 1) or item['label'] != expected:
                raise ValueError(f"Missing or inconsistent annotation for {item['id']}")
