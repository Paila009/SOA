"""Dependency-free localhost server for the research dashboard."""

import argparse
import json
import mimetypes
import re
import time
from difflib import SequenceMatcher
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from local_model import MODEL_RUNTIME, MODEL_RUNTIMES, MODEL_SPECS
from search_runtime import search_web, sources_to_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = Path(__file__).resolve().parent
PROBE_DIR = PROJECT_ROOT / "outputs" / "probes"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

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
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    sentences = []
    for part in parts:
        if sentences and re.fullmatch(r"(?:\[\d+\]\s*)+", part):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


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
    citations = {int(number) for number in re.findall(r"\[(\d+)\]", sentence)}
    if citations and not citations <= set(range(1, len(evidence_sources) + 1)):
        return 0.0, None, ""
    claim = re.sub(r"\[\d+\]", "", sentence).strip()
    answer_tokens = set(content_tokens(claim))
    if not answer_tokens:
        return 0.0, None, ""
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    answer_negated = bool(re.search(r"\b(?:not|never|no|without)\b|n't\b", claim, re.IGNORECASE))
    best = (0.0, None, "")
    for index, source in enumerate(evidence_sources, start=1):
        if citations and index not in citations:
            continue
        source_sentences = split_sentences(source.get("snippet", ""))
        passages = []
        for start in range(len(source_sentences)):
            for width in (1, 2, 3):
                if start + width <= len(source_sentences):
                    passages.append(" ".join(source_sentences[start:start + width]))
        for passage in passages:
            passage_tokens = set(content_tokens(passage))
            if not passage_tokens:
                continue
            overlap = len(answer_tokens & passage_tokens) / len(answer_tokens)
            if answer_numbers - set(re.findall(r"\b\d+(?:\.\d+)?\b", passage)):
                overlap = min(overlap, 0.25)
            passage_negated = bool(re.search(r"\b(?:not|never|no|without)\b|n't\b", passage, re.IGNORECASE))
            if answer_negated != passage_negated:
                overlap = min(overlap, 0.25)
            if overlap > best[0]:
                best = (overlap, index, passage[:420])
    return best


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
        support, source_index, excerpt = best_evidence_match(sentence, evidence_sources)
        weight = max(len(content_tokens(sentence)), 1)
        weights.append(weight)
        if source_index is not None:
            reviews[source_index - 1]["matched_claims"] += 1
            if support >= 0.55:
                reviews[source_index - 1]["strong_matches"] += 1
        matched_source = evidence_sources[source_index - 1] if source_index is not None else {}
        details.append({
            "sentence": sentence,
            "support": round(support, 4),
            "status": "supported" if support >= 0.55 else "uncertain" if support >= 0.3 else "unsupported",
            "source_index": source_index,
            "source_title": matched_source.get("title", ""),
            "source_url": matched_source.get("url", ""),
            "evidence_excerpt": excerpt,
        })
    total_weight = sum(weights) or 1
    support_score = sum(item["support"] * weight for item, weight in zip(details, weights)) / total_weight
    relevance = question_relevance(context, question) if question is not None else None
    if relevance is None:
        risk = 1.0 - support_score
    elif relevance < 0.25:
        risk = max(1.0 - support_score, 0.75)
    else:
        risk = 1.0 - (0.8 * support_score + 0.2 * relevance)
    substantive_claims = [item for item in details if len(content_tokens(item["sentence"])) >= 3]
    if substantive_claims:
        risk = max(risk, max(1.0 - item["support"] for item in substantive_claims))
    return {
        "risk": round(risk, 4),
        "support": round(support_score, 4),
        "question_relevance": round(relevance, 4) if relevance is not None else None,
        "decision": "high-risk" if risk > risk_threshold else "grounded",
        "threshold": risk_threshold,
        "sentences": details,
        "source_reviews": reviews,
        "evidence_available": bool(evidence_sources),
        "unsupported_count": sum(item["status"] == "unsupported" for item in details),
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
            "family": f"{spec['parameters']} parameters · Q4_K_M · local CPU · {spec['license']}",
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


def build_summary():
    benchmark = read_json(RESULTS_DIR / "benchmark_results.json", {})
    result_files = {
        "entropy": PROBE_DIR / "results_entropy-only.json",
        "logistic": PROBE_DIR / "results_logistic.json",
        "mlp": PROBE_DIR / "results_mlp.json",
    }
    results = {name: read_json(path) for name, path in result_files.items() if path.exists()}
    signal_count = len(list((PROJECT_ROOT / "outputs" / "signals" / "qwen2.5-1.5b").glob("*.pt")))
    stages = [
        {"name": "Dataset and labels", "status": "complete", "detail": "749 balanced RAGTruth records"},
        {"name": "Qwen signal extraction", "status": "complete", "detail": f"{signal_count} saved signal tensors"},
        {"name": "Entropy baseline", "status": "complete" if "entropy" in results else "pending", "detail": "Single-pass uncertainty baseline"},
        {"name": "Full probes", "status": "complete" if {"logistic", "mlp"} <= results.keys() else "pending", "detail": "331 features with train-only PCA"},
        {"name": "Activation patching", "status": "pending", "detail": "Needs Qwen inference environment"},
        {"name": "Gemma transfer", "status": "pending", "detail": "Needs secondary-model extraction"},
        {"name": "Steering and baselines", "status": "pending", "detail": "Needs controlled generation runs"},
    ]
    return {
        "project": "Causal Uncertainty Signatures",
        "signal_count": signal_count,
        "results": results,
        "benchmark": benchmark,
        "stages": stages,
        "completed_stages": sum(stage["status"] == "complete" for stage in stages),
        "total_stages": len(stages),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[dashboard] {self.address_string()} - {format % args}")

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self.send_json({"status": "ok"})
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
            return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, json.JSONDecodeError):
            return None

    def do_POST(self):
        parsed = urlparse(self.path)
        payload = self.read_json_body()
        if payload is None:
            return self.send_json({"error": "Invalid JSON body"}, 400)

        if parsed.path == "/api/analyze":
            context = str(payload.get("context", "")).strip()
            answer = str(payload.get("answer", "")).strip()
            threshold = float(payload.get("threshold", 0.5))
            if not answer:
                return self.send_json({"error": "Answer is required"}, 400)
            return self.send_json(analyze_grounding(context, answer, threshold))

        if parsed.path == "/api/chat":
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
            runtime = MODEL_RUNTIMES.get(model)
            routing_started = time.perf_counter()
            route_method = "local-model-intent" if runtime else "heuristic"
            try:
                intent = "chat" if is_casual_message(question) else (
                    runtime.classify_intent(question) if runtime else "search")
            except Exception:
                intent = "chat" if is_casual_message(question) else "search"
                route_method = "heuristic-fallback"
            routing_ms = (time.perf_counter() - routing_started) * 1000

            if intent == "chat":
                generation_started = time.perf_counter()
                try:
                    if runtime:
                        answer, usage = runtime.chat_general(question, history)
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
                return self.send_json({
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
                })

            search_started = time.perf_counter()
            sources = []
            search_error = None
            search_queries = plan_search_queries(question)
            comparison = len(search_queries) > 1
            search_query = " | ".join(search_queries)
            original_search_query = search_query
            if search_enabled:
                try:
                    if comparison:
                        seen_urls = set()
                        for query_index, query in enumerate(search_queries):
                            quota = max(1, max_sources // len(search_queries)
                                        + (query_index < max_sources % len(search_queries)))
                            for source in search_web(query, quota):
                                if source.get("url") not in seen_urls:
                                    sources.append({**source, "search_topic": query})
                                    seen_urls.add(source.get("url"))
                    else:
                        sources = search_web(search_query, max_sources)
                    if not sources and not comparison:
                        first_term = original_search_query.split()[0] if original_search_query.split() else ""
                        if first_term and len(first_term) >= 4:
                            candidates = search_web(first_term, 6)
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
                                sources = search_web(search_query, max_sources)
                    if not sources and runtime and not comparison:
                        try:
                            suggestion = runtime.rewrite_search_query(question)
                            if suggestion and suggestion.lower() != search_query.lower():
                                search_query = suggestion
                                sources = search_web(search_query, max_sources)
                        except Exception as exc:
                            search_error = f"Search retry failed: {exc}"
                except Exception as exc:
                    search_error = str(exc)
            retrieved_context = sources_to_context(sources)
            context_parts = [part for part in (retrieved_context, supplied_context) if part]
            context = "\n\n[USER-PROVIDED EVIDENCE]\n".join(context_parts)
            search_ms = (time.perf_counter() - search_started) * 1000

            generation_started = time.perf_counter()
            usage = {}
            if runtime:
                try:
                    answer, usage = runtime.chat(question, context, comparison=comparison)
                except Exception as exc:
                    return self.send_json({
                        "error": "Local model generation failed.",
                        "detail": str(exc),
                        "code": "MODEL_ERROR",
                    }, 503)
            else:
                answer = evidence_answer(context, question)
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
            return self.send_json({
                "model": model_info,
                "answer": answer,
                "analysis": analysis,
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
                "usage": usage,
                "detector": {
                    "name": "Search-grounding guard",
                    "research_probe": "Full logistic probe",
                    "test_auroc": 0.7154,
                    "test_f1": 0.6552,
                    "note": "Live answers use single-passage text overlap and query relevance, not the trained internal-signal probe or factual entailment.",
                },
            })

        return self.send_json({"error": "Not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="Serve the hallucination research dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
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
