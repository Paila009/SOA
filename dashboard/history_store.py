"""Privacy-safe local experiment ledger stored outside tracked source files."""

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
SCORE_VERSION = "nli-claims-v3-full-history"


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, question, model, runtime, analysis, sources, timings, mode,
               answer=None, retrieval=None, search_error=None):
        record = {
            "run_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_id": hashlib.sha256(question.encode("utf-8")).hexdigest()[:12],
            "model_id": model,
            "quantization": (runtime or {}).get("quantization"),
            "mode": mode,
            "decision": analysis.get("decision"),
            "evidence_coverage": analysis.get("evidence_coverage", analysis.get("support")),
            "weakest_claim_risk": analysis.get("weakest_claim_risk", analysis.get("risk")),
            "status_counts": analysis.get("status_counts", {}),
            "source_urls": [source.get("url", "") for source in sources],
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "sources": sources,
            "retrieval": retrieval,
            "search_error": search_error,
            "timings_ms": timings,
            "score_version": SCORE_VERSION,
            "storage_notice": "Full question, answer, evidence and analysis stored locally at user request.",
        }
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self, limit=200):
        if not self.path.exists():
            return []
        with _LOCK:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-max(1, min(int(limit), 1000)):]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(records))
