"""Shared evidence preparation and incremental claim review."""
import hashlib
import json
import re
from claim_verifier import verify_claim
from jobs import Cancelled


def resolve_followup(question, history):
    if not re.search(r'\b(he|she|him|her|his|their|them|it|that|they)\b', question, re.I):
        return question
    for item in reversed(history or []):
        if item.get('role') != 'user':
            continue
        previous = str(item.get('content', '')).strip(' ?.!')
        match = re.match(r'(?:who|what)\s+(?:is|was|are)\s+(.+)', previous, re.I)
        if match:
            subject = match.group(1)
            resolved = re.sub(r'\b(he|she|him|her|them|it|they|that)\b', lambda _:subject, question, flags=re.I)
            resolved = re.sub(r'\b(his|their)\b', lambda _:subject + "'s", resolved, flags=re.I)
            return resolved
    # Explicitly preserve unresolved context instead of guessing a person's name.
    return question


def evidence_id(sources, context):
    return hashlib.sha256(json.dumps({'sources':sources, 'private':context},
                                    sort_keys=True).encode()).hexdigest()[:16]


class ClaimStream:
    def __init__(self, job, sources, threshold=.5, guard=True):
        self.job, self.sources, self.threshold, self.guard = job, sources, threshold, guard
        self.pending = ''
        self.stopped = False
        self.reviewed = []

    def review(self, sentence):
        if self.job:
            self.job.check()
        review = verify_claim(sentence, self.sources)
        risk = 1 - review['score']
        blocked = self.guard and risk > self.threshold
        self.reviewed.append({'sentence':sentence, **review, 'blocked':blocked})
        if self.job:
            self.job.publish('claim', claim=self.reviewed[-1])
        self.stopped = blocked
        return not blocked

    def __call__(self, token):
        if self.job:
            self.job.check()
            self.job.publish('progress', stage='generating', received_characters=len(token))
        self.pending += token
        # Wait for citation suffixes and a subsequent token before finalizing a sentence.
        while True:
            match = re.search(r'([.!?](?:\s*\[\d+\])*)\s+(?=[A-Z•*\-])|\n(?=\s*[•*\-])', self.pending)
            if not match:
                break
            end = match.end()
            sentence, self.pending = self.pending[:end].strip(), self.pending[end:]
            if sentence and not self.review(sentence):
                return False
        return True

    def finish(self):
        if not self.stopped and self.pending.strip():
            self.review(self.pending.strip())
            self.pending = ''
