"""Dependency-free localhost server for the research dashboard."""

import argparse
import base64
import io
import json
import os
import mimetypes
import re
import time
import uuid
import sys
import threading
from difflib import SequenceMatcher
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Optional ML packages are installed only in the ignored project runtime.
sys.path.append(str(Path(__file__).resolve().parents[1] / '.runtime'))
from claim_verifier import split_claims, verify_claim
from nli_runtime import NLI
from jobs import JOBS, Cancelled
from chat_pipeline import ClaimStream, evidence_id, resolve_followup
from history_store import HistoryStore
from local_model import MODEL_RUNTIME, MODEL_RUNTIMES, MODEL_SPECS
from probe_adapter import ProbeAdapter
from search_runtime import retrieval_health, search_web_detailed, sources_to_context
from research_runtime import RESEARCH_RUNTIME, SPEC as RESEARCH_SPEC

MODEL_RUNTIMES['qwen-research'] = RESEARCH_RUNTIME
MODEL_SPECS['qwen-research'] = RESEARCH_SPEC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = Path(__file__).resolve().parent
PROBE_DIR = PROJECT_ROOT / "outputs" / "probes"
ALIGNED_PROBE_DIR = PROJECT_ROOT / "outputs" / "probes-aligned"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
HISTORY = HistoryStore(PROJECT_ROOT / "outputs" / "history" / "experiments.jsonl")
ACTIVE_PROBE_DIR = (ALIGNED_PROBE_DIR if
                    (ALIGNED_PROBE_DIR / "results_logistic.json").exists()
                    else PROBE_DIR)
PROBE_ADAPTER = ProbeAdapter(ACTIVE_PROBE_DIR)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "you", "your",
}


def content_tokens(text):
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def split_sentences(text):
    return split_claims(text)


def is_casual_message(text):
    """Identify non-factual chat that should not trigger retrieval or grounding alarms."""
    normalized = re.sub(r"[^a-z0-9\s']", " ", text.lower())
    normalized = " ".join(normalized.split())
    exact = {
        "hi", "hello", "hey", "hii", "hiii", "hola", "namaste",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "how are u", "how r u", "how are you doing", "what's up", "whats up",
        "thanks", "thank you", "okay", "ok", "cool", "bye", "goodbye",
    }
    if normalized in exact:
        return True
    greeting_prefixes = ("hi ", "hello ", "hey ", "thanks ", "thank you ")
    return len(normalized.split()) <= 6 and normalized.startswith(greeting_prefixes)


def build_search_query(question):
    """Turn conversational questions into concise entity/topic search queries."""
    cleaned = " ".join(question.strip().split())
    capital_match = re.search(r"\bcapital\s+of\s+(.+?)(?:\?|$)", cleaned, re.IGNORECASE)
    if capital_match:
        return capital_match.group(1).strip(" .?!")
    entity_match = re.match(r"^(?:who|where)\s+(?:is|was|are|were)\s+(.+?)(?:\?|$)", cleaned, re.IGNORECASE)
    if entity_match:
        return entity_match.group(1).strip(" .?!")
    return " ".join(content_tokens(question)) or question


def plan_search_queries(question):
    """Fetch evidence for both sides of an explicit comparison."""
    cleaned = " ".join(question.strip().split()).strip(" .?!")
    patterns = (
        r"^(?:compare|contrast)\s+(.+?)\s+(?:and|with|versus|vs\.?|v\.)\s+(.+)$",
        r"^(?:what(?:'s| is)?\s+(?:the\s+)?(?:difference|similarit(?:y|ies))\s+between)\s+(.+?)\s+and\s+(.+)$",
        r"^how\s+(?:does|do)\s+(.+?)\s+(?:differ from|compare (?:with|to))\s+(.+)$",
        r"^(.+?)\s+(?:versus|vs\.?|v\.)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            subjects = []
            for part in match.groups():
                part = re.sub(r"\s+(?:in terms of|on|regarding)\s+.+$", "", part, flags=re.IGNORECASE)
                part = re.sub(r"\s+(?:and\s+)?(?:explain|describe|summarize|list|state|tell me)\b.*$",
                              "", part, flags=re.IGNORECASE)
                subjects.append(part.strip(" .?!"))
            if all(len(subject) >= 2 for subject in subjects):
                return subjects
    return [build_search_query(question)]


def best_evidence_match(sentence, evidence_sources):
    """Find one attributable passage; never combine unrelated source vocabularies."""
    result = verify_claim(sentence, evidence_sources)
    return result["score"], result["source_index"], result["excerpt"]


def sentence_support(context, sentence):
    """Compatibility helper for callers that supply one private evidence block."""
    return best_evidence_match(sentence, [{"snippet": context}])[0]


def question_relevance(context, question):
    question_terms = set(content_tokens(question))
    if not question_terms:
        return 0.0
    context_terms = set(content_tokens(context))
    return len(question_terms & context_terms) / len(question_terms)


def analyze_grounding(context, answer, risk_threshold=0.5, question=None, sources=None,
                      supplied_context=""):
    evidence_sources = list(sources or [])
    if supplied_context:
        evidence_sources.append({"title": "Private evidence", "url": "", "snippet": supplied_context})
    elif sources is None and context:
        evidence_sources.append({"title": "Provided evidence", "url": "", "snippet": context})
    sentences = split_sentences(answer)
    details = []
    weights = []
    reviews = [{
        "title": source.get("title", "Evidence"), "url": source.get("url", ""),
        "matched_claims": 0, "strong_matches": 0,
    } for source in evidence_sources]
    for sentence in sentences:
        verification = verify_claim(sentence, evidence_sources)
        support, source_index, excerpt = (verification["score"], verification["source_index"],
                                          verification["excerpt"])
        weight = max(len(content_tokens(sentence)), 1)
        weights.append(weight)
        if source_index is not None:
            reviews[source_index - 1]["matched_claims"] += 1
            if verification["status"] == "supported":
                reviews[source_index - 1]["strong_matches"] += 1
        matched_source = evidence_sources[source_index - 1] if source_index is not None else {}
        details.append({
            "sentence": sentence,
            "support": round(support, 4),
            "status": verification["status"],
            "source_index": source_index,
            "source_title": matched_source.get("title", ""),
            "source_url": matched_source.get("url", ""),
            "evidence_excerpt": excerpt,
            "conflicting_excerpt": verification.get('conflicting_excerpt'),
            "signals": verification["signals"],
            "reason": verification["reason"],
        })
    total_weight = sum(weights) or 1
    support_score = sum(item["support"] * weight for item, weight in zip(details, weights)) / total_weight
    relevance = question_relevance(context, question) if question is not None else None
    # Query wording overlap is a retrieval diagnostic, not proof of false claims.
    coverage_risk = 1.0 - support_score
    substantive_claims = [item for item in details if len(content_tokens(item["sentence"])) >= 3]
    weakest_claim_risk = max((1.0 if item["status"] == "contradicted" else 1.0 - item["support"]
                              for item in substantive_claims), default=0.0)
    risk = max(coverage_risk, weakest_claim_risk)
    status_counts = {status: sum(item["status"] == status for item in details)
                     for status in ("supported", "partial", "contradicted", "unverified")}
    if not evidence_sources:
        guard_reason = "No evidence was available, so the claims are unverified rather than proven false."
    elif status_counts["contradicted"]:
        guard_reason = f"{status_counts['contradicted']} claim(s) conflict with a reviewed passage."
    elif status_counts["unverified"]:
        guard_reason = f"{status_counts['unverified']} claim(s) have no sufficiently matching passage."
    elif status_counts["partial"]:
        guard_reason = f"{status_counts['partial']} claim(s) have only partial passage support."
    else:
        guard_reason = "Every reviewed claim has a strong match in one attributable passage."
    return {
        "risk": round(risk, 4),
        "weakest_claim_risk": round(weakest_claim_risk, 4),
        "coverage_risk": round(coverage_risk, 4),
        "support": round(support_score, 4),
        "evidence_coverage": round(support_score, 4),
        "question_relevance": round(relevance, 4) if relevance is not None else None,
        "decision": "high-risk" if risk > risk_threshold else "grounded",
        "threshold": risk_threshold,
        "sentences": details,
        "source_reviews": reviews,
        "evidence_available": bool(evidence_sources),
        "status_counts": status_counts,
        "guard_reason": guard_reason,
        "unsupported_count": sum(item["status"] in {"contradicted", "unverified"} for item in details),
    }


def evidence_answer(context, question):
    sentences = split_sentences(context)
    if not sentences:
        return "I cannot answer from evidence because no retrieval context was provided."
    question_terms = set(content_tokens(question))
    ranked = []
    for index, sentence in enumerate(sentences):
        tokens = set(content_tokens(sentence))
        overlap = len(tokens & question_terms)
        density = overlap / max(len(question_terms), 1)
        ranked.append((density, overlap, -index, sentence))
    ranked.sort(reverse=True)
    selected = [item[3] for item in ranked[:3] if item[1] > 0]
    if not selected:
        return "The supplied context does not contain enough matching evidence to answer this question safely."
    return "Based on the supplied evidence: " + " ".join(selected)


def available_models():
    models = []
    for model_id, spec in MODEL_SPECS.items():
        runtime = MODEL_RUNTIMES[model_id].status()
        models.append({
            "id": model_id, "name": spec["name"],
            "family": f"{spec['parameters']} parameters · {spec.get('quantization','Q4_K_M')} · local CPU · {spec['license']}",
            "available": runtime["installed"], "ready": runtime["ready"],
            "loading": runtime["loading"], "installed": runtime["installed"],
            "detail": (
                "Loaded locally. Retrieves sources, generates an answer, then reviews claims."
                if runtime["ready"] else
                "Installed locally. Loads on first use; the first answer may take longer."
                if runtime["installed"] else runtime["error"] or "Model weights are not installed."
            ),
        })
    models.append({
            "id": "evidence-guard",
            "name": "Evidence-only fallback",
            "family": "Deterministic extractive fallback",
            "available": True,
            "ready": True,
            "loading": False,
            "installed": True,
            "detail": "Ranks retrieved sentences without generative inference.",
    })
    return models


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def calibrated_probe_threshold(rows):
    """Choose a validation-only F1 threshold; never reuse test labels for tuning."""
    validation = [row for row in rows if row.get("split") == "val" and
                  row.get("probability") is not None and row.get("label") in (0, 1)]
    if not validation:
        return None
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in (value / 100 for value in range(5, 96)):
        tp = sum(row["label"] == 1 and row["probability"] >= threshold for row in validation)
        fp = sum(row["label"] == 0 and row["probability"] >= threshold for row in validation)
        fn = sum(row["label"] == 1 and row["probability"] < threshold for row in validation)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best[0]:
            best = (f1, threshold, precision, recall)
    return {"threshold": round(best[1], 2), "f1": round(best[0], 4),
            "precision": round(best[2], 4), "recall": round(best[3], 4),
            "samples": len(validation), "split": "validation"}


def build_summary():
    benchmark = read_json(RESULTS_DIR / "benchmark_results.json", {})
    aligned_dir = PROJECT_ROOT / 'outputs' / 'probes-aligned'
    result_dir = aligned_dir if (aligned_dir/'results_logistic.json').exists() else PROBE_DIR
    result_files = {
        "entropy": result_dir / "results_entropy-only.json",
        "logistic": result_dir / "results_logistic.json",
        "mlp": result_dir / "results_mlp.json",
    }
    results = {name: read_json(path) for name, path in result_files.items() if path.exists()}
    logistic_predictions = read_json(result_dir / "predictions_logistic.json", [])
    signal_root = PROJECT_ROOT/'outputs/signals'/('qwen2.5-1.5b-aligned' if result_dir == aligned_dir else 'qwen2.5-1.5b')
    signal_count = len(list(signal_root.glob("*.pt")))
    aligned_signal_count = len(list((PROJECT_ROOT/'outputs/signals/qwen2.5-1.5b-aligned').glob("*.pt")))
    aligned = result_dir == aligned_dir
    stages = [
        {"name": "Dataset and labels", "status": "complete" if aligned else "needs-repair", "detail": "Exact response-label alignment" if aligned else "Legacy response-label mismatch"},
        {"name": "Qwen signal extraction", "status": "complete" if aligned_signal_count == 749 else "running", "detail": f"{aligned_signal_count}/749 exact-response signal tensors"},
        {"name": "Entropy baseline", "status": "complete" if "entropy" in results else "pending", "detail": "Single-pass uncertainty baseline"},
        {"name": "Full probes", "status": "complete" if {"logistic", "mlp"} <= results.keys() else "pending", "detail": "331 features with train-only PCA"},
        {"name": "Activation patching", "status": "pending", "detail": "Needs Qwen inference environment"},
        {"name": "Gemma transfer", "status": "pending", "detail": "Needs secondary-model extraction"},
        {"name": "Steering and baselines", "status": "pending", "detail": "Needs controlled generation runs"},
    ]
    return {
        "project": "Causal Uncertainty Signatures",
        "result_provenance": "aligned exact-response experiment" if aligned else "legacy mismatched-label experiment",
        "signal_count": signal_count,
        "aligned_signal_count": aligned_signal_count,
        "results": results,
        "probe_calibration": calibrated_probe_threshold(logistic_predictions),
        "probe_status": PROBE_ADAPTER.status(),
        "benchmark": benchmark,
        "stages": stages,
        "completed_stages": sum(stage["status"] == "complete" for stage in stages),
        "total_stages": len(stages),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def handle_one_request(self):
        self.request_id = uuid.uuid4().hex[:12]
        return super().handle_one_request()

    def log_message(self, format, *args):
        print(json.dumps({"time": int(time.time()), "request_id": getattr(self, "request_id", None),
                          "client": self.address_string(), "message": format % args}), flush=True)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Request-ID", getattr(self, "request_id", "unknown"))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self.send_json({"status": "ok", "retrieval": retrieval_health(), 'verifier': NLI.status()})
        if parsed.path.startswith('/api/jobs/'):
            job = JOBS.get(parsed.path.rsplit('/', 1)[-1])
            if not job:
                return self.send_json({'error':'Unknown or expired job'}, 404)
            try:
                cursor = max(0, int(parse_qs(parsed.query).get('cursor', ['0'])[0]))
            except ValueError:
                return self.send_json({'error':'Invalid cursor'}, 400)
            return self.send_json(job.snapshot(cursor))
        if parsed.path == "/api/retrieval-health":
            return self.send_json(retrieval_health())
        if parsed.path == "/api/history":
            limit = parse_qs(parsed.query).get("limit", ["200"])[0]
            return self.send_json(HISTORY.read(limit))
        if parsed.path == "/api/probe-status":
            return self.send_json(PROBE_ADAPTER.status())
        if parsed.path == "/api/summary":
            return self.send_json(build_summary())
        if parsed.path == "/api/models":
            return self.send_json(available_models())
        if parsed.path == "/api/runtime":
            model_id = parse_qs(parsed.query).get("model", ["qwen2.5-1.5b"])[0]
            runtime = MODEL_RUNTIMES.get(model_id, MODEL_RUNTIME)
            return self.send_json(runtime.status())
        if parsed.path == "/api/predictions":
            params = parse_qs(parsed.query)
            model = params.get("model", ["logistic"])[0]
            split = params.get("split", ["test"])[0]
            filenames = {
                "entropy": "predictions_entropy-only.json",
                "logistic": "predictions_logistic.json",
                "mlp": "predictions_mlp.json",
            }
            if model not in filenames:
                return self.send_json({"error": "Unknown model"}, 400)
            rows = read_json(PROBE_DIR / filenames[model], [])
            return self.send_json([row for row in rows if row.get("split") == split])

        relative = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
        target = (DASHBOARD_ROOT / relative).resolve()
        try:
            target.relative_to(DASHBOARD_ROOT)
        except ValueError:
            return self.send_error(403)
        if not target.exists() or not target.is_file():
            return self.send_error(404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 12 * 1024 * 1024:
                return None
            return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, json.JSONDecodeError):
            return None

    def do_POST(self):
        parsed = urlparse(self.path)
        payload = self.read_json_body()
        if payload is None:
            return self.send_json({"error": "Invalid JSON body"}, 400)
        return self.dispatch_post(parsed.path, payload)

    def dispatch_post(self, path, payload):
        if path.startswith('/api/jobs/') and path.endswith('/cancel'):
            job = JOBS.get(path.split('/')[-2])
            if not job:
                return self.send_json({'error':'Unknown job'}, 404)
            job.cancel()
            return self.send_json({'cancelled':True})
        if path == '/api/jobs':
            try:
                job = JOBS.create()
            except ValueError as exc:
                return self.send_json({'error':str(exc)}, 429)
            threading.Thread(target=run_chat_job, args=(job, payload), daemon=True).start()
            return self.send_json({'id':job.id}, 202)
        if path == '/api/evidence':
            question = resolve_followup(str(payload.get('question','')), payload.get('history', []))
            context = str(payload.get('context',''))[:120000]
            sources, attempts = [], []
            queries = plan_search_queries(question)
            limit = min(6, max(2, int(payload.get('max_sources',4))))
            if payload.get('search_enabled',True):
                seen = set()
                for query in queries:
                    found, meta = search_web_detailed(query, max(1, limit//len(queries)))
                    attempts.append(meta)
                    for source in found:
                        if source['url'] not in seen:
                            seen.add(source['url'])
                            sources.append(source)
            return self.send_json({'sources':sources, 'context':context, 'question':question,
                                   'snapshot_id':evidence_id(sources, context), 'attempts':attempts})

        if path == "/api/analyze":
            context = str(payload.get("context", "")).strip()
            answer = str(payload.get("answer", "")).strip()
            threshold = float(payload.get("threshold", 0.5))
            if not answer:
                return self.send_json({"error": "Answer is required"}, 400)
            return self.send_json(analyze_grounding(context, answer, threshold))

        if path == "/api/documents/extract":
            filename = Path(str(payload.get("filename", "document"))).name
            encoded = str(payload.get("content_base64", ""))
            try:
                raw = base64.b64decode(encoded, validate=True)
            except ValueError:
                return self.send_json({"error": "Invalid document encoding"}, 400)
            if len(raw) > 8 * 1024 * 1024:
                return self.send_json({"error": "Document exceeds the 8 MB local limit"}, 413)
            extension = Path(filename).suffix.lower()
            if extension in {".txt", ".md"}:
                text = raw.decode("utf-8", errors="replace")
            elif extension == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(raw))
                    if len(reader.pages) > 100:
                        return self.send_json({"error": "PDF exceeds the 100-page local limit"}, 413)
                    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                except ImportError:
                    return self.send_json({"error": "PDF extraction dependency is unavailable"}, 503)
                except Exception as exc:
                    return self.send_json({"error": f"Could not read PDF: {exc}"}, 400)
            else:
                return self.send_json({"error": "Only .txt, .md, and .pdf evidence files are accepted"}, 415)
            text = text.strip()[:120000]
            if not text:
                return self.send_json({"error": "The document contains no extractable text"}, 400)
            return self.send_json({"filename": filename, "characters": len(text), "text": text,
                                   "stored": False, "privacy": "Processed in memory and not uploaded externally."})

        if path == "/api/chat":
            job = getattr(self, 'job', None)
            if job:
                job.check()
            started = time.perf_counter()
            model = str(payload.get("model", "qwen2.5-1.5b"))
            supplied_context = str(payload.get("context", "")).strip()
            question = str(payload.get("question", "")).strip()
            threshold = min(max(float(payload.get("threshold", 0.5)), 0.05), 0.95)
            guardrail = bool(payload.get("guardrail", True))
            search_enabled = bool(payload.get("search_enabled", True))
            max_sources = min(max(int(payload.get("max_sources", 4)), 1), 6)
            model_info = next((item for item in available_models() if item["id"] == model), None)
            if model_info is None:
                return self.send_json({"error": "Unknown model"}, 400)
            if model_info.get("loading"):
                return self.send_json({
                    "error": f"{model_info['name']} is still loading.",
                    "detail": "Wait a few seconds; the model selector will turn green when inference is ready.",
                    "code": "MODEL_LOADING",
                }, 503)
            if not model_info["available"]:
                return self.send_json({
                    "error": f"{model_info['name']} is currently offline.",
                    "detail": model_info["detail"],
                    "code": "MODEL_OFFLINE",
                }, 503)
            if not question:
                return self.send_json({"error": "Question is required"}, 400)

            history = payload.get("history", [])
            original_question = question
            question = resolve_followup(question, history)
            frozen = payload.get('evidence_snapshot')
            if frozen:
                if not isinstance(frozen, dict) or frozen.get('snapshot_id') != evidence_id(frozen.get('sources',[]), frozen.get('context','')):
                    return self.send_json({'error':'Invalid evidence snapshot'}, 400)
                question = frozen['question']
                supplied_context = frozen['context']
            runtime = MODEL_RUNTIMES.get(model)
            routing_started = time.perf_counter()
            route_method = "local-model-intent" if runtime else "heuristic"
            try:
                intent = "chat" if is_casual_message(question) and not frozen else 'search'
                route_method = 'conversation-aware-routing'
            except Exception:
                intent = "chat" if is_casual_message(question) else "search"
                route_method = "heuristic-fallback"
            routing_ms = (time.perf_counter() - routing_started) * 1000

            if intent == "chat":
                generation_started = time.perf_counter()
                try:
                    if runtime:
                        answer, usage = runtime.chat_general(question, history, job=job,
                            on_token=(lambda token: job.publish('token', text=token)) if job else None)
                    else:
                        answer, usage = "Hello! Ask me anything and I can search for sources before answering.", {}
                except Exception as exc:
                    return self.send_json({
                        "error": "Local model generation failed.",
                        "detail": str(exc),
                        "code": "MODEL_ERROR",
                    }, 503)
                generation_ms = (time.perf_counter() - generation_started) * 1000
                latency_ms = (time.perf_counter() - started) * 1000
                response_payload = {
                    "mode": "conversation",
                    "search_skipped": True,
                    "routing": {"intent": intent, "method": route_method, "latency_ms": round(routing_ms, 2)},
                    "runtime": runtime.status() if runtime else None,
                    "model": model_info,
                    "answer": answer,
                    "analysis": {
                        "risk": 0.0,
                        "support": None,
                        "question_relevance": None,
                        "decision": "conversational",
                        "threshold": threshold,
                        "sentences": [{"sentence": answer, "support": None, "status": "neutral"}],
                        "unsupported_count": 0,
                    },
                    "guardrail_triggered": False,
                    "latency_ms": round(latency_ms, 2),
                    "search_ms": 0.0,
                    "generation_ms": round(generation_ms, 2),
                    "sources": [],
                    "search_error": None,
                    "usage": usage,
                    "detector": {
                        "name": "Conversational intent router",
                        "research_probe": "Not applicable",
                        "test_auroc": None,
                        "test_f1": None,
                        "note": "Greetings contain no factual claims, so source-grounding risk is not applied.",
                    },
                }
                HISTORY.append(question, model, response_payload["runtime"], response_payload["analysis"], [],
                               {"total": response_payload["latency_ms"], "search": 0.0,
                                "generation": response_payload["generation_ms"]}, "conversation",
                               answer=answer)
                return self.send_json(response_payload)

            search_started = time.perf_counter()
            sources = []
            search_error = None
            retrieval_attempts = []
            search_queries = plan_search_queries(question)
            comparison = len(search_queries) > 1
            search_query = " | ".join(search_queries)
            original_search_query = search_query
            if frozen:
                sources = frozen['sources']
                retrieval_attempts = frozen.get('attempts', [])
            if search_enabled and not frozen:
                try:
                    if comparison:
                        seen_urls = set()
                        for query_index, query in enumerate(search_queries):
                            quota = max(1, max_sources // len(search_queries)
                                        + (query_index < max_sources % len(search_queries)))
                            query_sources, query_meta = search_web_detailed(query, quota)
                            retrieval_attempts.append(query_meta)
                            for source in query_sources:
                                if source.get("url") not in seen_urls:
                                    sources.append({**source, "search_topic": query})
                                    seen_urls.add(source.get("url"))
                    else:
                        sources, query_meta = search_web_detailed(search_query, max_sources)
                        retrieval_attempts.append(query_meta)
                    if not sources and not comparison:
                        first_term = original_search_query.split()[0] if original_search_query.split() else ""
                        if first_term and len(first_term) >= 4:
                            candidates, query_meta = search_web_detailed(first_term, 6)
                            retrieval_attempts.append(query_meta)
                            ranked = sorted(
                                candidates,
                                key=lambda source: SequenceMatcher(
                                    None, original_search_query.lower(), source["title"].lower()
                                ).ratio(),
                                reverse=True,
                            )
                            if ranked and SequenceMatcher(
                                None, original_search_query.lower(), ranked[0]["title"].lower()
                            ).ratio() >= 0.74:
                                search_query = ranked[0]["title"]
                                sources, query_meta = search_web_detailed(search_query, max_sources)
                                retrieval_attempts.append(query_meta)
                    if not sources and runtime and hasattr(runtime,'rewrite_search_query') and not comparison and not job:
                        try:
                            suggestion = runtime.rewrite_search_query(question)
                            if suggestion and suggestion.lower() != search_query.lower():
                                search_query = suggestion
                                sources, query_meta = search_web_detailed(search_query, max_sources)
                                retrieval_attempts.append(query_meta)
                        except Exception as exc:
                            search_error = f"Search retry failed: {exc}"
                except Exception as exc:
                    search_error = str(exc)
                if not search_error:
                    failed = [item for item in retrieval_attempts if item.get("status") == "error"]
                    if failed and not sources:
                        search_error = failed[-1].get("error")
            if job:
                job.check()
                job.publish('sources', sources=sources, search_error=search_error,
                            resolved_question=question)
            retrieved_context = sources_to_context(sources)
            context_parts = [part for part in (retrieved_context, supplied_context) if part]
            context = retrieved_context
            if supplied_context:
                context += f'\n\n[{len(sources)+1}] Private evidence\n{supplied_context}'
            search_ms = (time.perf_counter() - search_started) * 1000

            generation_started = time.perf_counter()
            usage = {}
            evidence_sources = sources + ([{'title':'Private evidence','snippet':supplied_context,'url':''}] if supplied_context else [])
            monitor = ClaimStream(job, evidence_sources, threshold, guardrail) if job else None
            if runtime:
                try:
                    answer, usage = runtime.chat(question, context, comparison=comparison,
                        history=history, job=job, on_token=monitor,
                        max_tokens=min(512,max(32,int(payload.get('max_tokens',240)))),
                        temperature=min(1.0,max(0.0,float(payload.get('temperature',.15)))))
                    if monitor:
                        monitor.finish()
                except Exception as exc:
                    return self.send_json({
                        "error": "Local model generation failed.",
                        "detail": str(exc),
                        "code": "MODEL_ERROR",
                    }, 503)
            else:
                answer = evidence_answer(context, question)
            if job:
                job.check()
            generation_ms = (time.perf_counter() - generation_started) * 1000

            relevance_query = search_query if sources and search_query != original_search_query else question
            analysis = analyze_grounding(context, answer, threshold, relevance_query,
                                         sources=sources, supplied_context=supplied_context)
            stopped = guardrail and analysis["risk"] > threshold
            if stopped:
                safe_claims = [item["sentence"] for item in analysis["sentences"]
                               if item["status"] == "supported" and item["source_index"] is not None]
                if safe_claims:
                    answer = ("Only these claims had a strong source-text match; I hid the rest for review:\n"
                              + "\n".join(f"• {claim}" for claim in safe_claims))
                else:
                    answer = (
                        "I could not retrieve a source to review this answer. Try a clearer search or add private evidence."
                        if not analysis["evidence_available"] else
                        "The guard hid the generated answer because its claims did not match a "
                        "reviewed source passage strongly enough. Open Analysis to inspect the claims and passages."
                    )
                analysis["safe_claims_shown"] = len(safe_claims)
                analysis["decision"] = "stopped"
            latency_ms = (time.perf_counter() - started) * 1000
            response_payload = {
                "model": model_info,
                "answer": answer,
                "analysis": analysis,
                'resolved_question': question,
                'original_question': original_question,
                'snapshot_id': evidence_id(sources, supplied_context),
                'inference_stopped_by_guard': bool(monitor and monitor.stopped),
                'verifier': NLI.status(),
                'research_probe': usage.get('research_probe'),
                "mode": "search",
                "search_skipped": not search_enabled,
                "routing": {"intent": intent, "method": route_method, "latency_ms": round(routing_ms, 2)},
                "runtime": runtime.status() if runtime else None,
                "guardrail_triggered": stopped,
                "latency_ms": round(latency_ms, 2),
                "search_ms": round(search_ms, 2),
                "search_query": search_query,
                "search_queries": search_queries,
                "original_search_query": original_search_query,
                "generation_ms": round(generation_ms, 2),
                "sources": sources,
                "search_error": search_error,
                "retrieval": {
                    "status": "ok" if sources else "error" if search_error else "no-results",
                    "attempts": retrieval_attempts,
                    "health": retrieval_health(),
                },
                "usage": usage,
                "detector": {
                    "name": "Search-grounding guard",
                    "research_probe": "Full logistic probe",
                    "test_auroc": 0.7154,
                    "test_f1": 0.6552,
                    "note": "Live answers use local NLI with lexical fallback. Offline probe scores have a known response-label mismatch and are unvalidated.",
                },
            }
            HISTORY.append(question, model, response_payload["runtime"], analysis, sources,
                           {"total": response_payload["latency_ms"], "search": response_payload["search_ms"],
                            "generation": response_payload["generation_ms"]}, "search",
                           answer=answer, retrieval=response_payload['retrieval'],
                           search_error=search_error)
            return self.send_json(response_payload)

        return self.send_json({"error": "Not found"}, 404)


def run_chat_job(job, payload):
    class Worker(DashboardHandler):
        def __init__(self):
            self.job = job
        def send_json(self, data, status=200):
            job.check()
            job.publish('result' if status < 400 else 'error', data=data)
    try:
        Worker().dispatch_post('/api/chat', payload)
    except Cancelled:
        job.publish('cancelled')
    except Exception as exc:
        if job.cancelled.is_set():
            job.publish('cancelled')
        else:
            job.publish('error', data={'error':str(exc)})
    finally:
        job.done = True


def main():
    parser = argparse.ArgumentParser(description="Serve the hallucination research dashboard")
    parser.add_argument("--host", default=os.getenv("GROUNDED_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GROUNDED_PORT", "8766")))
    args = parser.parse_args()
    MODEL_RUNTIME.ensure_started(background=True)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
