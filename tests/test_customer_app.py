"""Customer boundary tests. No model/network calls or credentials are used."""
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from customer.app import create_app
from customer.config import Settings
from customer.provider import Job, ModelAPI, ProviderError, RateLimitError, parse_sse
from customer.review import split_claims, validate_review, summarize
from customer.store import Store


def config(tmp_path, live=False):
    return Settings(database=tmp_path / "workspace.sqlite3", firebase={"projectId": "unit-test"} if live else {},
                    models={"groq:test-model": {"id": "groq:test-model", "model": "test-model", "provider": "groq", "providerName": "Groq"}},
                    keys={"groq": "test-only-secret"}, groq_free_plan_confirmed=True)


def token(token):
    if token == "invalid":
        raise ValueError()
    return {"uid": token, "email_verified": token != "unverified"}


def client(app, uid=None):
    c = TestClient(app, base_url="http://127.0.0.1:8770")
    if uid:
        c.headers["Authorization"] = "Bearer " + uid
    return c


def test_preview_data_is_isolated_and_no_paid_calls(tmp_path):
    app = create_app(config(tmp_path))
    a, b = client(app), client(app)
    sample = a.post("/api/sample").json()
    assert a.get("/api/workspace").json()["chats"][0]["id"] == sample["id"]
    assert not b.get("/api/workspace").json()["chats"]
    assert b.get("/api/chats/" + sample["id"]).status_code == 404
    assert b.get("/api/chats/" + sample["id"] + "/export").status_code == 404
    assert b.delete("/api/chats/" + sample["id"]).status_code == 404
    assert a.post("/api/research", json={"question": "Hello", "model": "groq:test-model", "consent": True}).status_code == 503


def test_auth_fail_closed_and_user_boundaries(tmp_path):
    app = create_app(config(tmp_path, True), token_verifier=token)
    anon, bad, pending, a, b = [client(app, uid) for uid in [None, "invalid", "unverified", "alice", "bob"]]
    assert anon.get("/api/workspace").status_code == 401
    assert bad.get("/api/workspace").status_code == 401
    assert pending.get("/api/workspace").status_code == 403
    sample = a.post("/api/sample").json()
    assert b.get("/api/chats/" + sample["id"]).status_code == 404
    assert a.get("/api/chats/" + sample["id"]).status_code == 200
    job = Job("alice", sample["id"])
    app.state.jobs[job.id] = job
    assert b.get("/api/jobs/" + job.id).status_code == 404
    assert b.post("/api/jobs/" + job.id + "/stop").status_code == 404
    assert not job.cancelled.is_set()
    assert a.post("/api/jobs/" + job.id + "/stop").status_code == 200
    assert a.delete("/api/workspace").status_code == 409
    job.update(done=True)
    assert a.delete("/api/workspace").status_code == 200
    assert not a.get("/api/workspace").json()["chats"]


def test_documents_scope_type_limits_and_removal(tmp_path):
    app = create_app(config(tmp_path, True), token_verifier=token)
    a, b = client(app, "alice"), client(app, "bob")
    item = a.post("/api/documents", headers={"X-Filename": "notes.md"}, content=b"A source passage for research.").json()
    assert len(a.get("/api/workspace").json()["documents"]) == 1
    assert not b.get("/api/workspace").json()["documents"]
    assert b.delete("/api/documents/" + item["id"]).status_code == 404
    assert b.post("/api/research", json={"question": "Review this", "model": "groq:test-model", "documents": [item["id"]], "consent": True}).status_code == 404
    assert a.post("/api/documents", headers={"X-Filename": "script.html"}, content=b"<script>").status_code == 415
    assert a.post("/api/documents", content=b"a" * 40001).status_code == 400
    assert a.post("/api/documents", content=b"a" * (2 * 1024 * 1024 + 1)).status_code == 413
    assert a.delete("/api/documents/" + item["id"]).status_code == 200


def test_origin_host_security_and_static_allowlist(tmp_path):
    c = client(create_app(config(tmp_path)))
    assert c.post("/api/sample", headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.get("/api/config", headers={"Host": "evil.example"}).status_code == 400
    assert "test-only-secret" not in c.get("/api/config").text
    assert c.get("/assets/.env").status_code == 404
    assert c.get("/assets/app.py").status_code == 404
    html = c.get("/")
    assert html.status_code == 200
    assert "frame-ancestors 'none'" in html.headers["content-security-policy"]
    assert html.headers["cache-control"] == "no-store"


def test_public_model_catalog_separates_live_local_and_disabled_providers(tmp_path):
    public = config(tmp_path, True).public()
    assert public["models"][0]["provider"] == "groq"
    assert [item["id"] for item in public["localModels"]] == ["qwen2.5-1.5b", "phi-3-mini", "qwen3-4b"]
    assert all(item["chatId"].startswith("local:") for item in public["localModels"])
    assert all(item["downloadUrl"].startswith("https://huggingface.co/") for item in public["localModels"])
    policies = {item["id"]: item for item in public["providers"]}
    assert policies["groq"]["enabled"] is True
    assert policies["gemini"]["enabled"] is True
    assert policies["openrouter"]["enabled"] is True
    assert "test-only-secret" not in json.dumps(public)


def test_installed_local_model_is_available_in_customer_chat(tmp_path, monkeypatch):
    import customer.config as customer_config
    item = {"id": "qwen", "chatId": "local:qwen", "name": "Local Qwen", "parameters": "1B",
            "quantization": "Q4", "license": "Apache-2.0", "downloadUrl": "https://example.test/model", "installed": True}
    monkeypatch.setattr(customer_config, "local_model_catalog", lambda: [item])
    settings = Settings(database=tmp_path / "workspace.sqlite3", firebase={"projectId": "unit-test"},
                        local_models_enabled=True)
    model = settings.require_model("local:qwen")
    assert model["provider"] == "local"
    assert settings.available_models() == [model]


def test_local_model_adapter_streams_without_api_key(tmp_path, monkeypatch):
    import customer.config as customer_config
    import dashboard.local_model as local_model
    item = {"id": "qwen", "chatId": "local:qwen", "name": "Local Qwen", "parameters": "1B",
            "quantization": "Q4", "license": "Apache-2.0", "downloadUrl": "https://example.test/model", "installed": True}
    monkeypatch.setattr(customer_config, "local_model_catalog", lambda: [item])

    class Runtime:
        def _complete(self, messages, **kwargs):
            kwargs["on_token"]("Local ")
            kwargs["on_token"]("answer")
            return "Local answer", {"completion_tokens": 2}

    monkeypatch.setitem(local_model.MODEL_RUNTIMES, "qwen", Runtime())
    settings = Settings(database=tmp_path / "workspace.sqlite3", local_models_enabled=True)
    updates = []
    answer = ModelAPI(settings).complete("local:qwen", [{"role": "user", "content": "Hello"}], Job("alice", "chat"), updates.append)
    assert answer == "Local answer"
    assert updates == ["Local ", "Local answer"]


def test_production_and_remote_preview_require_auth(tmp_path):
    with pytest.raises(ValueError):
        Settings(environment="production", database=tmp_path / "x").validate()
    with pytest.raises(ValueError):
        Settings(origin="http://example.com", database=tmp_path / "x").validate()


def test_review_cannot_cite_missing_or_invented_evidence():
    sources = [{"id": 1, "snippet": "The experiment enrolled twenty adult participants."}]
    claims = ["The experiment enrolled twenty participants."]
    valid = {"claims": [{"id": 1, "status": "supported", "source": 1, "quote": sources[0]["snippet"], "reason": "Matches."}]}
    assert validate_review(claims, sources, valid)[0]["status"] == "supported"
    valid["claims"][0]["quote"] = "The experiment enrolled two hundred participants."
    assert validate_review(claims, sources, valid)[0]["status"] == "unverified"
    valid["claims"][0].update(quote=sources[0]["snippet"], source=50)
    assert validate_review(claims, sources, valid)[0]["status"] == "unverified"
    assert validate_review(claims, sources, {"claims": [None, "invalid", {"id": 1, "status": "certain"}]})[0]["status"] == "unverified"
    assert summarize([])["coverage"] is None


def test_sse_parsing():
    lines = [": keepalive", "", 'data: {"choices":[{"delta":{"content":"Hello"}}]}', 'data: {"choices":[{"delta":{"content":" world"}}]}', "data: [DONE]", 'data: {"choices":[{"delta":{"content":"ignored"}}]}']
    assert "".join(parse_sse(lines)) == "Hello world"
    with pytest.raises(ProviderError):
        list(parse_sse(['data: {"error":{"message":"private upstream detail"}}']))


class FakeAPI:
    def __init__(self):
        self.calls = []

    def complete(self, model, messages, job, on_token=None, max_tokens=None):
        self.calls.append((model, messages))
        if on_token:
            answer = "The experiment enrolled twenty adult participants. [1]"
            on_token(answer)
            return answer
        return json.dumps({"claims": [{"id": 1, "status": "supported", "source": 1, "quote": "The experiment enrolled twenty adult participants.", "reason": "The document gives this participant count."}]})


def wait_for_job(c, identity):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = c.get("/api/jobs/" + identity).json()
        if result["done"]:
            return result
        time.sleep(.02)
    pytest.fail("Job did not finish")


def test_local_chat_route_uses_same_workspace_without_api_consent(tmp_path, monkeypatch):
    import customer.config as customer_config
    item = {"id": "qwen", "chatId": "local:qwen", "name": "Local Qwen", "parameters": "1B",
            "quantization": "Q4", "license": "Apache-2.0", "downloadUrl": "https://example.test/model", "installed": True}
    monkeypatch.setattr(customer_config, "local_model_catalog", lambda: [item])
    settings = Settings(database=tmp_path / "workspace.sqlite3", firebase={"projectId": "unit-test"},
                        local_models_enabled=True)
    fake = FakeAPI()
    c = client(create_app(settings, token_verifier=token, model_api=fake), "alice")
    started = c.post("/api/research", json={"question": "Hello", "model": "local:qwen", "search": False})
    assert started.status_code == 200
    result = wait_for_job(c, started.json()["jobId"])
    assert result["done"] is True and result["error"] is None
    assert fake.calls[0][0] == "local:qwen"


def test_complete_api_pipeline_saved_review_reload_and_export(tmp_path):
    fake = FakeAPI()
    app = create_app(config(tmp_path, True), token_verifier=token, model_api=fake)
    c = client(app, "alice")
    doc = c.post("/api/documents", headers={"X-Filename": "study.txt"}, content=b"The experiment enrolled twenty adult participants.").json()
    payload = {"question": "How many participants were enrolled?", "model": "groq:test-model", "documents": [doc["id"]], "search": False, "consent": True}
    started = c.post("/api/research", json=payload)
    assert started.status_code == 200
    ids = started.json()
    result = wait_for_job(c, ids["jobId"])
    assert not result["error"]
    assert result["detail"]["review"]["counts"]["supported"] == 1
    assert len(fake.calls) == 2
    saved = c.get("/api/chats/" + ids["chatId"]).json()
    assert saved["messages"][1]["detail"]["sources"][0]["title"] == "study.txt"
    assert saved["messages"][1]["content"].startswith("The experiment")
    reloaded = client(create_app(config(tmp_path, True), token_verifier=token, model_api=fake), "alice")
    assert len(reloaded.get("/api/chats/" + ids["chatId"]).json()["messages"]) == 2
    assert "SUPPORTED" in reloaded.get("/api/chats/" + ids["chatId"] + "/export").text
    assert not client(app, "bob").get("/api/workspace").json()["chats"]


def test_cancellation_and_single_active_request(tmp_path):
    entered = threading.Event()
    class SlowAPI:
        def complete(self, model, messages, job, on_token=None, **kwargs):
            entered.set()
            while True:
                job.check()
                time.sleep(.01)
    app = create_app(config(tmp_path, True), token_verifier=token, model_api=SlowAPI())
    c = client(app, "alice")
    payload = {"question": "What is an experiment?", "model": "groq:test-model", "search": False, "consent": True}
    ids = c.post("/api/research", json=payload).json()
    assert entered.wait(3)
    assert c.post("/api/research", json=payload).status_code == 429
    assert c.post("/api/jobs/" + ids["jobId"] + "/stop").status_code == 200
    assert wait_for_job(c, ids["jobId"])["phase"] == "stopped"
    assert c.get("/api/chats/" + ids["chatId"]).json()["messages"][1]["detail"]["cancelled"]


def test_usage_limit_is_persistent(tmp_path):
    db = tmp_path / "usage.sqlite3"
    store = Store(db)
    store.reserve_usage("alice", 1, 2)
    with pytest.raises(ValueError):
        Store(db).reserve_usage("alice", 1, 2)
    store.reserve_usage("bob", 1, 2)
    with pytest.raises(ValueError):
        store.reserve_usage("cora", 1, 2)


def test_consent_and_configuration_validation(tmp_path):
    c = client(create_app(config(tmp_path, True), token_verifier=token), "alice")
    assert c.post("/api/research", json={"question": "Hello", "model": "groq:test-model"}).status_code == 400
    assert c.post("/api/research", json={"question": "Hello", "model": "arbitrary-url", "consent": True}).status_code == 400
    assert c.post("/api/research", json={"question": " " * 3, "model": "groq:test-model", "consent": True}).status_code == 400


def test_short_claims_and_malformed_verdict_fail_closed():
    assert split_claims("Earth is flat.") == ["Earth is flat."]
    claims = ["A short but factual claim."]
    assert validate_review(claims, [], {"claims": [{"id": 1, "status": {"supported": True}}]})[0]["status"] == "unverified"
    assert validate_review(claims, [], ["not an object"])[0]["status"] == "unverified"


def test_model_adapter_sends_server_key_and_streams(tmp_path, monkeypatch):
    import httpx
    import customer.provider as provider
    original_client = httpx.Client
    seen = []
    def respond(request):
        seen.append(request)
        assert request.url == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-only-secret"
        body = json.loads(request.content)
        assert body["stream"] is True and body["model"] == "test-model"
        assert "test-only-secret" not in request.content.decode()
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"A real "}}]}\n\ndata: {"choices":[{"delta":{"content":"stream shape."}}]}\n\ndata: [DONE]\n\n')
    monkeypatch.setattr(provider.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    updates = []
    job = Job("alice", "test")
    answer = ModelAPI(config(tmp_path, True)).complete("groq:test-model", [{"role": "user", "content": "question"}], job, updates.append)
    assert answer == "A real stream shape."
    assert updates == ["A real ", "A real stream shape."]
    assert len(seen) == 1 and job.closer is None


def test_provider_errors_do_not_leak_raw_payload_or_retry(tmp_path, monkeypatch):
    import httpx
    import customer.provider as provider
    original_client = httpx.Client
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(401, text="Sensitive diagnostic test-only-secret")
    monkeypatch.setattr(provider.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(ProviderError) as error:
        ModelAPI(config(tmp_path, True)).complete("groq:test-model", [], Job("alice", "test"))
    assert "HTTP 401" in str(error.value)
    assert "test-only-secret" not in str(error.value)
    assert len(requests) == 1


def test_missing_evidence_is_unverified_not_contradicted(tmp_path):
    fake = FakeAPI()
    c = client(create_app(config(tmp_path, True), token_verifier=token, model_api=fake), "alice")
    ids = c.post("/api/research", json={"question": "Explain experiments", "model": "groq:test-model", "search": False, "consent": True}).json()
    result = wait_for_job(c, ids["jobId"])
    assert result["detail"]["review"]["counts"] == {"unverified": 1}
    assert len(fake.calls) == 1


def test_strict_mode_keeps_original_in_review_and_filters_unsupported(tmp_path):
    fake = FakeAPI()
    c = client(create_app(config(tmp_path, True), token_verifier=token, model_api=fake), "alice")
    ids = c.post("/api/research", json={"question": "Explain experiments", "model": "groq:test-model", "search": False, "strict": True, "consent": True}).json()
    result = wait_for_job(c, ids["jobId"])
    assert result["text"].startswith("No claims passed")
    assert result["detail"]["original"].startswith("The experiment enrolled")


def test_two_answers_keep_separate_saved_reviews(tmp_path):
    fake = FakeAPI()
    c = client(create_app(config(tmp_path, True), token_verifier=token, model_api=fake), "alice")
    document = c.post("/api/documents", content=b"The experiment enrolled twenty adult participants.").json()
    payload = {"question": "First question", "model": "groq:test-model", "search": False, "consent": True, "documents": [document["id"]]}
    first = c.post("/api/research", json=payload).json()
    wait_for_job(c, first["jobId"])
    payload.update(question="Second question without evidence", chatId=first["chatId"], documents=[])
    second = c.post("/api/research", json=payload).json()
    wait_for_job(c, second["jobId"])
    messages = c.get("/api/chats/" + first["chatId"]).json()["messages"]
    assert len(messages) == 4
    assert messages[1]["detail"]["review"]["counts"] == {"supported": 1}
    assert messages[3]["detail"]["review"]["counts"] == {"unverified": 1}
    assert messages[1]["id"] != messages[3]["id"]


def test_provider_failure_is_saved_without_fabricated_review(tmp_path):
    class FailedAPI:
        def complete(self, *args, **kwargs):
            raise ProviderError("Model provider is unavailable.")
    c = client(create_app(config(tmp_path, True), token_verifier=token, model_api=FailedAPI()), "alice")
    ids = c.post("/api/research", json={"question": "Hello there", "model": "groq:test-model", "search": False, "consent": True}).json()
    result = wait_for_job(c, ids["jobId"])
    assert result["phase"] == "error"
    messages = c.get("/api/chats/" + ids["chatId"]).json()["messages"]
    assert messages[1]["detail"]["error"] is True
    assert not messages[1]["detail"]["review"]["claims"]


def test_free_plan_confirmation_is_required_on_client_and_server(tmp_path, monkeypatch):
    settings = config(tmp_path, True)
    settings.groq_free_plan_confirmed = False
    fake = FakeAPI()
    c = client(create_app(settings, token_verifier=token, model_api=fake), "alice")
    assert c.get("/api/config").json()["models"] == []
    assert c.post("/api/research", json={"question": "Hello", "model": "groq:test-model", "consent": True}).status_code == 400
    assert not fake.calls and not c.get("/api/workspace").json()["chats"]
    monkeypatch.setattr("customer.provider.httpx.Client", lambda **kwargs: pytest.fail("Network must not be opened"))
    with pytest.raises(ProviderError, match="free-only workspace"):
        ModelAPI(settings).complete("groq:test-model", [], Job("alice", "test"))


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_api_providers_require_explicit_free_only_confirmation(tmp_path, provider):
    settings = config(tmp_path, True)
    identity = provider + ":test-model"
    settings.models[identity] = {"id": identity, "model": "test-model", "provider": provider, "providerName": provider}
    settings.keys[provider] = "another-test-only-secret"
    c = client(create_app(settings, token_verifier=token, model_api=FakeAPI()), "alice")
    assert [m["provider"] for m in c.get("/api/config").json()["models"]] == ["groq"]
    assert c.post("/api/research", json={"question": "Hello", "model": identity, "consent": True}).status_code == 400
    settings.review_model = identity
    with pytest.raises(ValueError, match="free-only workspace"):
        settings.validate()


def test_confirmed_gemini_and_free_openrouter_models_can_be_configured(tmp_path):
    settings = config(tmp_path, True)
    settings.models.update({
        "gemini:test-flash": {"id": "gemini:test-flash", "model": "test-flash", "provider": "gemini", "providerName": "Google Gemini"},
        "openrouter:test:free": {"id": "openrouter:test:free", "model": "test:free", "provider": "openrouter", "providerName": "OpenRouter"},
    })
    settings.keys.update({"gemini": "test-gemini-secret", "openrouter": "test-openrouter-secret"})
    settings.gemini_free_tier_confirmed = True
    settings.openrouter_free_only_confirmed = True
    providers = {item["provider"] for item in settings.available_models()}
    assert providers == {"groq", "gemini", "openrouter"}


@pytest.mark.parametrize("header, seconds", [("15", 15), ("bad", 60), ("0", 1), ("999999", 86400)])
def test_quota_stops_without_retry_or_fallback(tmp_path, monkeypatch, header, seconds):
    import httpx
    original_client = httpx.Client
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(429, headers={"retry-after": header}, text="private upstream payload")
    monkeypatch.setattr("customer.provider.httpx.Client", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    api = ModelAPI(config(tmp_path, True))
    with pytest.raises(RateLimitError) as exc:
        api.complete("groq:test-model", [], Job("alice", "test"))
    assert exc.value.retry_after == seconds
    assert "private upstream" not in str(exc.value)
    with pytest.raises(RateLimitError):
        api.complete("groq:test-model", [], Job("bob", "other"))
    assert len(seen) == 1  # Shared cooldown; no automatic retry or model switch.


def test_cooldown_does_not_create_chat_or_use_job_quota(tmp_path):
    settings = config(tmp_path, True)
    api = ModelAPI(settings)
    api.cooldown_until = time.monotonic() + 60
    c = client(create_app(settings, token_verifier=token, model_api=api), "alice")
    result = c.post("/api/research", json={"question": "Hello", "model": "groq:test-model", "consent": True})
    assert result.status_code == 429 and int(result.headers["retry-after"]) > 0
    assert not c.get("/api/workspace").json()["chats"]


def test_review_quota_preserves_answer_without_inventing_verdicts(tmp_path):
    class ReviewLimited(FakeAPI):
        def complete(self, model, messages, job, on_token=None, **kwargs):
            if on_token:
                return super().complete(model, messages, job, on_token)
            raise RateLimitError(60)
    c = client(create_app(config(tmp_path, True), token_verifier=token, model_api=ReviewLimited()), "alice")
    doc = c.post("/api/documents", content=b"The experiment enrolled twenty adult participants.").json()
    ids = c.post("/api/research", json={"question": "How many participants?", "model": "groq:test-model", "documents": [doc["id"]], "search": False, "consent": True}).json()
    result = wait_for_job(c, ids["jobId"])
    assert result["text"].startswith("The experiment enrolled")
    assert result["detail"]["review"]["counts"] == {"unverified": 1}
    assert "free-tier limit" in result["detail"]["review"]["note"]
    assert "free-tier limit" in c.get("/api/chats/" + ids["chatId"]).json()["messages"][1]["detail"]["review"]["note"]


def test_owner_readiness_is_offline_and_never_prints_secrets(tmp_path):
    from customer.check_setup import readiness
    settings = config(tmp_path)
    checks = readiness(settings)
    assert checks["Groq API key saved (optional)"]
    assert checks["Usable chat model available"]
    assert not checks["Installed local chat model available"]
    assert not checks["Firebase web configuration complete"]
    assert not checks["Firebase Admin credential path configured"]
    assert "test-only-secret" not in json.dumps(checks)
