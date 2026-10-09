"""Provider catalog, per-account isolation, and captured credential tests."""
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from customer.app import create_app
from customer.config import Settings
from customer.connections import Connections, list_provider_models
from customer.provider import ModelAPI, ProviderError


def settings(tmp_path):
    return Settings(database=tmp_path / "connections.sqlite3", firebase={"projectId": "test-project"})


def token(value):
    return {"uid": value, "email_verified": True}


def client(app, identity="alice"):
    return TestClient(app, base_url="http://127.0.0.1:8770",
                      headers={"Authorization": "Bearer " + identity} if identity else {})


def transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr("customer.connections.httpx.Client",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_catalog_uses_headers_and_filters_non_chat_models(monkeypatch):
    seen = []
    def handler(request):
        seen.append(request)
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer test-secret-key"
        assert "test-secret-key" not in str(request.url)
        return httpx.Response(200, json={"data": [
            {"id": "chat-model", "active": True}, {"id": "whisper-large"},
            {"id": "embedding-model"}, {"id": "disabled", "active": False},
            {"id": "speech-model", "architecture": {"output_modalities": ["audio"]}},
            {"id": "other-model", "name": "Other"}, {"id": "bad\nmodel"}]})
    transport(monkeypatch, handler)
    assert list_provider_models("groq", "test-secret-key") == [
        {"id": "chat-model", "name": "chat-model"}, {"id": "other-model", "name": "Other"}]
    assert str(seen[0].url) == "https://api.groq.com/openai/v1/models"


def test_openrouter_checks_key_and_allows_only_free_text_models(monkeypatch):
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/v1/key":
            return httpx.Response(200, json={"data": {"label": "private-account-label"}})
        return httpx.Response(200, json={"data": [
            {"id": "org/chat:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "org/chat", "pricing": {"prompt": "0"}},
            {"id": "org/wrong:free", "pricing": {"prompt": "0.1"}},
            {"id": "org/audio:free", "architecture": {"output_modalities": ["audio"]}}]})
    transport(monkeypatch, handler)
    assert list_provider_models("openrouter", "test-secret-key") == [{"id": "org/chat:free", "name": "org/chat:free"}]
    assert paths == ["/api/v1/key", "/api/v1/models"]


def test_gemini_catalog_excludes_non_chat_generators(monkeypatch):
    transport(monkeypatch, lambda request: httpx.Response(200, json={"data": [
        {"id": "gemini-test-flash"}, {"id": "gemini-native-audio-latest"},
        {"id": "gemini-image-preview"}, {"id": "embedding-001"}]}))
    assert list_provider_models("gemini", "test-secret-key") == [{"id": "gemini-test-flash", "name": "gemini-test-flash"}]


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_provider_error_redacts_credentials_and_account_payload(monkeypatch, status):
    transport(monkeypatch, lambda request: httpx.Response(status, text="test-secret-key and private upstream diagnostic"))
    with pytest.raises(ProviderError) as exc:
        list_provider_models("groq", "test-secret-key")
    assert "test-secret-key" not in str(exc.value)
    assert "private upstream" not in str(exc.value)


def test_temporary_account_connections_do_not_mutate_owner_or_other_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr("customer.connections.list_provider_models", lambda provider, key: [{"id": "chat-model"}])
    base, vault = settings(tmp_path), Connections()
    owner_api = ModelAPI(base)
    with pytest.raises(ValueError, match="free plan"):
        vault.connect("alice", base, "groq", "first-test-secret", "chat-model", False)
    with pytest.raises(ValueError, match="exact"):
        vault.connect("alice", base, "groq", "first-test-secret", "invented-model", True)
    vault.connect("alice", base, "groq", "first-test-secret", "chat-model", True)
    captured_settings, captured_api = vault.resolve("alice", base, owner_api, "groq:chat-model")
    assert captured_settings.keys == {"groq": "first-test-secret"}
    assert captured_api is not owner_api
    assert base.keys == {} and base.models == {}
    assert vault.public("bob", base)["models"] == []
    with pytest.raises(ValueError):
        vault.resolve("bob", base, owner_api, "groq:chat-model")
    public = vault.public("alice", base)
    assert public["models"][0]["connection"] == "personal"
    assert "first-test-secret" not in json.dumps(public)
    vault.connect("alice", base, "groq", "second-test-secret", "chat-model", True)
    assert captured_settings.keys == {"groq": "first-test-secret"}
    assert vault.resolve("alice", base, owner_api, "groq:chat-model")[0].keys == {"groq": "second-test-secret"}
    vault.disconnect("alice", "groq")
    assert vault.public("alice", base)["models"] == []
    assert captured_api.settings.keys["groq"] == "first-test-secret"


def test_connect_routes_require_auth_and_return_scoped_config(tmp_path, monkeypatch):
    app = create_app(settings(tmp_path), token_verifier=token)
    alice, bob, anonymous = client(app), client(app, "bob"), client(app, None)
    monkeypatch.setattr("customer.connections.list_provider_models", lambda provider, key: [{"id": "chat-model"}])
    body = {"provider": "groq", "key": "temporary-test-secret", "model": "chat-model", "freeConfirmed": True}
    assert anonymous.post("/api/providers/connect", json=body).status_code == 401
    connected = alice.post("/api/providers/connect", json=body)
    assert connected.status_code == 200
    assert connected.json()["models"][0]["id"] == "groq:chat-model"
    assert "temporary-test-secret" not in connected.text
    assert alice.get("/api/workspace").json()["config"]["models"][0]["id"] == "groq:chat-model"
    assert bob.get("/api/workspace").json()["config"]["models"] == []
    assert app.state.store.chats("alice") == []
    disconnected = alice.post("/api/providers/disconnect", json={"provider": "groq"})
    assert disconnected.json()["models"] == []


def test_preview_cannot_connect_and_invalid_inputs_do_not_echo_key(tmp_path):
    app = create_app(Settings(database=tmp_path / "preview.sqlite3"))
    c = client(app, None)
    assert c.post("/api/providers/models", json={"provider": "groq", "key": "test-secret-key"}).status_code == 503
    live = client(create_app(settings(tmp_path), token_verifier=token))
    invalid = live.post("/api/providers/connect", json={"provider": "groq", "key": {"secret": "do-not-echo-secret"}})
    assert invalid.status_code == 422
    assert "do-not-echo-secret" not in invalid.text


def test_connected_key_runs_real_provider_adapter_in_scoped_job(tmp_path, monkeypatch):
    base = settings(tmp_path)
    c = client(create_app(base, token_verifier=token))
    seen_keys = []
    def handler(request):
        seen_keys.append(request.headers["authorization"])
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "chat-model"}]})
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"Hello from the connected model."}}]}\n\ndata: [DONE]\n\n')
    transport(monkeypatch, handler)
    body = {"provider": "groq", "key": "temporary-test-secret", "model": "chat-model", "freeConfirmed": True}
    assert c.post("/api/providers/connect", json=body).status_code == 200
    started = c.post("/api/research", json={"question": "hi", "model": "groq:chat-model", "search": False, "consent": True})
    assert started.status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = c.get("/api/jobs/" + started.json()["jobId"]).json()
        if result["done"]:
            break
        time.sleep(.02)
    assert result["phase"] == "complete" and result["text"] == "Hello from the connected model."
    saved = c.get("/api/chats/" + started.json()["chatId"])
    assert "temporary-test-secret" not in saved.text
    assert seen_keys == ["Bearer temporary-test-secret", "Bearer temporary-test-secret"]
    assert base.keys == {} and base.models == {}
