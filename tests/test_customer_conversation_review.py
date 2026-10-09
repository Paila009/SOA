"""Conversational text remains visible without weakening factual claim review."""
import time

import pytest
from fastapi.testclient import TestClient

from customer.app import create_app
from customer.config import Settings
from customer.review import is_nonfactual, review_answer, review_local_answer


@pytest.mark.parametrize("answer", ["Hello!", "Thank you.", "How can I assist you today?"])
def test_exact_greetings_and_questions_need_no_sources(answer):
    class NoAPI:
        def complete(self, *args, **kwargs):
            pytest.fail("Conversation needs no API evidence review")
    for review in (review_local_answer(answer, []), review_answer(NoAPI(), "unused", answer, [], None)):
        assert review["counts"] == {"not_factual": 1}
        assert review["coverage"] is None
        assert review["claims"][0]["source"] is None


def test_greeting_with_factual_followon_reviews_each_sentence():
    result = review_local_answer("Hello! Mars has cities.", [])
    assert [c["status"] for c in result["claims"]] == ["not_factual", "unverified"]
    assert not is_nonfactual("Hi Mars has cities")
    assert not is_nonfactual("History is a scientific discipline.")


def test_conversation_skips_local_matching_even_when_sources_exist(monkeypatch):
    monkeypatch.setattr("dashboard.claim_verifier.verify_claim", lambda *args, **kwargs: pytest.fail("Greeting must not be matched to unrelated evidence"))
    result = review_local_answer("Hello! How can I assist you today?", [{"id": 1, "snippet": "Unrelated evidence."}])
    assert result["counts"] == {"not_factual": 2}


def test_strict_local_chat_keeps_conversation_and_filters_unverified_facts(tmp_path, monkeypatch):
    monkeypatch.setattr("customer.config.local_model_catalog", lambda: [
        {"id": "test", "chatId": "local:test", "name": "Test", "installed": True}])
    class Conversation:
        def complete(self, model, messages, job, on_token=None, **kwargs):
            answer = "Hello! How can I assist you today? Mars has cities."
            if on_token:
                on_token(answer)
            return answer
    settings = Settings(database=tmp_path / "workspace.sqlite3", firebase={"projectId": "test"}, local_models_enabled=True)
    client = TestClient(create_app(settings, token_verifier=lambda _: {"uid": "alice", "email_verified": True},
                                   model_api=Conversation()), base_url=settings.origin,
                        headers={"Authorization": "Bearer alice"})
    started = client.post("/api/research", json={"question": "Hi", "model": "local:test", "strict": True, "search": False}).json()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get("/api/jobs/" + started["jobId"]).json()
        if response["done"]:
            assert response["text"] == "Hello!\n\nHow can I assist you today?"
            assert response["detail"]["review"]["counts"] == {"not_factual": 2, "unverified": 1}
            assert response["detail"]["original"].endswith("Mars has cities.")
            return
        time.sleep(.02)
    pytest.fail("Strict conversation job did not finish")
