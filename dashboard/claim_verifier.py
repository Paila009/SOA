"""Explainable, claim-level verification against attributable passages."""

from difflib import SequenceMatcher
import re
try:
    from .nli_runtime import NLI
except ImportError:  # dashboard/server.py also supports direct script execution.
    from nli_runtime import NLI

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
             "have", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
             "to", "was", "were", "what", "when", "where", "which", "who", "why", "with"}
NEGATIONS = re.compile(r"\b(?:not|never|no|without|neither|nor)\b|n't\b", re.IGNORECASE)
ENTITY_IGNORE = {"The", "This", "That", "It", "He", "She", "They", "His", "Her", "Their",
                 "A", "An", "Based", "According"}


def content_tokens(text):
    return [token for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in STOPWORDS]


def split_claims(text):
    """Split prose/bullets into reviewable claims and keep standalone citations attached."""
    text = re.sub(r"^\s*[•*-]\s*", "", str(text), flags=re.MULTILINE)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+|;\s+", text) if part.strip()]
    claims = []
    for part in parts:
        if claims and re.fullmatch(r"(?:\[\d+\]\s*)+", part):
            claims[-1] += " " + part
        else:
            claims.append(part)
    return claims


def _entities(text):
    entities = set()
    for match in re.findall(r"\b(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)*", text):
        cleaned = match.strip()
        if cleaned not in ENTITY_IGNORE and len(cleaned) > 1:
            entities.add(cleaned.casefold())
    return entities


def _numbers(text):
    return set(re.findall(r"\b(?:\d{1,4}(?:[,.]\d+)*|\d+(?:st|nd|rd|th))\b", text.lower()))


def _semantic_fallback(claim, passage):
    """Dependency-free paraphrase hint; never treated as entailment by itself."""
    claim_norm = " ".join(content_tokens(claim))
    passage_norm = " ".join(content_tokens(passage))
    if not claim_norm or not passage_norm:
        return 0.0
    return SequenceMatcher(None, claim_norm, passage_norm).ratio()


def score_passage(claim, passage):
    claim_tokens, passage_tokens = set(content_tokens(claim)), set(content_tokens(passage))
    lexical = len(claim_tokens & passage_tokens) / max(len(claim_tokens), 1)
    semantic = _semantic_fallback(claim, passage)
    claim_numbers, passage_numbers = _numbers(claim), _numbers(passage)
    number_consistent = not claim_numbers or claim_numbers <= passage_numbers
    claim_negated, passage_negated = bool(NEGATIONS.search(claim)), bool(NEGATIONS.search(passage))
    negation_consistent = claim_negated == passage_negated
    claim_entities, passage_entities = _entities(claim), _entities(passage)
    missing_entities = {entity for entity in claim_entities
                        if not any(entity in candidate or candidate in entity for candidate in passage_entities)}
    entity_consistent = not missing_entities
    contradiction_reasons = []
    # Absence alone is not a contradiction. Only compare the same relation.
    number_skeleton = lambda text: re.sub(r'\d+(?:[,.]\d+)*', '#', text.lower()).strip(' .')
    if (claim_numbers and passage_numbers and not number_consistent
            and number_skeleton(claim) == number_skeleton(passage)):
        contradiction_reasons.append("number/date differs from the passage")
    affirmative = lambda text: set(content_tokens(NEGATIONS.sub('', text)))
    if (not negation_consistent and affirmative(claim) == affirmative(passage)
            and bool(affirmative(claim)) and entity_consistent):
        contradiction_reasons.append("negation reverses the passage")
    combined = 0.78 * lexical + 0.22 * semantic
    if not negation_consistent:
        combined = min(combined, 0.29)
    if not number_consistent or not entity_consistent:
        combined = min(combined, 0.24)
    if contradiction_reasons:
        combined = min(combined, 0.24)
    if contradiction_reasons:
        status = "contradicted"
    elif combined >= 0.58 and lexical >= 0.5:
        status = "supported"
    elif combined >= 0.3:
        status = "partial"
    else:
        status = "unverified"
    return {
        "score": round(combined, 4), "status": status,
        "signals": {"lexical": round(lexical, 4), "semantic": round(semantic, 4),
                    "entity_consistent": entity_consistent, "number_consistent": number_consistent,
                    "negation_consistent": negation_consistent,
                    "semantic_backend": "lexical-fallback"},
        "reason": "; ".join(contradiction_reasons) if contradiction_reasons else
                  "strong single-passage match" if status == "supported" else
                  "some wording matches, but support is incomplete" if status == "partial" else
                  "no sufficiently matching passage",
    }


def verify_claim(claim, evidence_sources, use_nli=True):
    citations = {int(number) for number in re.findall(r"\[(\d+)\]", claim)}
    if citations and not citations <= set(range(1, len(evidence_sources) + 1)):
        return {"score": 0.0, "status": "unverified", "source_index": None,
                "excerpt": "", "signals": {}, "reason": "citation does not identify an available source"}
    clean_claim = re.sub(r"\[\d+\]", "", claim).strip()
    best = None
    candidates = []
    for index, source in enumerate(evidence_sources, start=1):
        if citations and index not in citations:
            continue
        sentences = split_claims(source.get("snippet", ""))
        for start in range(len(sentences)):
            for width in (1, 2, 3):
                if start + width > len(sentences):
                    continue
                passage = " ".join(sentences[start:start + width])
                result = score_passage(clean_claim, passage)
                candidate = {**result, "source_index": index, "excerpt": passage[:520]}
                candidates.append((result['signals']['lexical'], passage, candidate))
                priority = {"supported": 3, "contradicted": 2, "partial": 1, "unverified": 0}
                rank = (priority[result["status"]], result["score"])
                if best is None or rank > best[0]:
                    best = (rank, candidate)
    if use_nli and candidates:
        # Inspect relevant passages, preserving both supporting and opposing evidence.
        candidates.sort(key=lambda item: (-item[0], len(item[1])))
        judged = []
        for relevance, passage, candidate in candidates[:6]:
            probabilities = NLI.predict(passage, clean_claim)
            if probabilities is None:
                break
            entail = probabilities['entailment']
            contra = probabilities['contradiction']
            status = ('supported' if entail >= .8 else
                      'contradicted' if contra >= .85 and relevance >= .25 else
                      'partial' if entail >= .45 else 'unverified')
            judged.append({**candidate, 'status': status, 'score': round(entail, 4),
                           'signals': {**candidate['signals'], 'nli': probabilities,
                                       'semantic_backend': NLI.status()['model']},
                           'reason': {'supported':'NLI model finds passage support',
                                      'contradicted':'NLI model identifies a conflicting claim',
                                      'partial':'Evidence only partially supports the claim',
                                      'unverified':'Passage does not establish this claim'}[status]})
        if judged:
            support = [item for item in judged if item['status'] == 'supported']
            conflict = [item for item in judged if item['status'] == 'contradicted']
            # A short unrelated sentence in the same document must not override
            # direct support (e.g. museum admission versus cafe admission).
            if support:
                conflict = [item for item in conflict if not any(
                    item['source_index'] == match['source_index']
                    and item['signals']['lexical'] < .75
                    for match in support)]
            if support and conflict:
                return {**support[0], 'status':'partial', 'score': .45,
                        'reason':'Sources disagree; both passages require review',
                        'conflicting_excerpt':conflict[0]['excerpt']}
            selected = max(judged, key=lambda item: (item['status'] == 'supported',
                           item['status'] == 'contradicted', item['score']))
            if selected['status'] == 'unverified':
                # A passage rejected by the verifier is not evidence for the claim.
                # Keeping its text in the UI makes incidental word overlap look like
                # attribution, so preserve diagnostics but remove the false linkage.
                return {**selected, 'score': 0.0, 'source_index': None, 'excerpt': ''}
            return selected
    if best is None or best[1]["status"] == "unverified":
        return {"score": 0.0, "status": "unverified", "source_index": None,
                "excerpt": "", "signals": {}, "reason": "no sufficiently matching passage"}
    return best[1]
