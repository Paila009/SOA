"""Per-account API connections held only in the running server's memory."""
import re
import threading
from dataclasses import dataclass, field, replace

import httpx

from .config import PROVIDERS
from .provider import ModelAPI, ProviderError

MODEL_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
}


def validate_credentials(provider, key):
    if provider not in PROVIDERS:
        raise ValueError("Choose Groq, Google Gemini, or OpenRouter.")
    key = key.strip()
    if not 10 <= len(key) <= 512 or not key.isascii() or any(c.isspace() for c in key):
        raise ValueError("Enter a valid API key without spaces.")
    return key


def _json_get(client, url, headers, label):
    try:
        response = client.get(url, headers=headers)
        if response.status_code != 200:
            reasons = {401: "the API key was not accepted", 403: "access was denied",
                       429: "the provider's rate limit was reached"}
            raise ProviderError(f"{label}: {reasons.get(response.status_code, 'the model list could not be loaded')} (HTTP {response.status_code}).")
        if len(response.content) > 4 * 1024 * 1024:
            raise ProviderError(f"{label}: the model list was too large to inspect.")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError()
        return payload
    except (httpx.HTTPError, ValueError):
        # Upstream diagnostics and HTTP exceptions can contain credential headers.
        raise ProviderError(f"{label}: could not verify the connection. Check the key and try again.") from None


def list_provider_models(provider, key):
    """Read catalog/auth endpoints only; never generate an answer to test a key."""
    key = validate_credentials(provider, key)
    label = PROVIDERS[provider][0]
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    with httpx.Client(timeout=httpx.Timeout(15, connect=8), follow_redirects=False) as client:
        if provider == "openrouter":
            # The public catalog may succeed even when a supplied key is invalid.
            account = _json_get(client, "https://openrouter.ai/api/v1/key", headers, label)
            if not isinstance(account.get("data"), dict):
                raise ProviderError("OpenRouter: the API key could not be verified.")
        payload = _json_get(client, MODEL_ENDPOINTS[provider], headers, label)
    data = payload.get("data")
    if not isinstance(data, list):
        raise ProviderError(f"{label}: the model list returned an unsupported format.")
    models = {}
    for item in data[:2000]:
        if not isinstance(item, dict) or item.get("active") is False:
            continue
        identity = item.get("id")
        if not isinstance(identity, str) or not 1 <= len(identity) <= 180 or not re.fullmatch(r"[A-Za-z0-9._/:-]+", identity):
            continue
        if any(word in identity.lower() for word in ("whisper", "orpheus", "audio", "tts", "embed", "rerank", "image", "vision-only", "prompt-guard", "safeguard", "transcrib", "live-", "native-audio")):
            continue
        architecture = item.get("architecture")
        if isinstance(architecture, dict):
            outputs = architecture.get("output_modalities")
            if outputs is not None and (not isinstance(outputs, list) or "text" not in outputs):
                continue
        if provider == "gemini" and not identity.startswith("gemini-"):
            continue
        if provider == "openrouter":
            if not identity.endswith(":free"):
                continue
            pricing = item.get("pricing", {})
            try:
                if any(float(pricing.get(name, "0")) != 0 for name in ("prompt", "completion", "request")):
                    continue
            except (ValueError, TypeError, AttributeError):
                continue
        name = item.get("name")
        models[identity] = {"id": identity, "name": name[:180] if isinstance(name, str) and name else identity}
    return sorted(models.values(), key=lambda item: item["id"])


@dataclass
class Connection:
    model: dict
    settings: object = field(repr=False)
    api: object = field(repr=False)


class Connections:
    def __init__(self):
        self.lock = threading.RLock()
        self.accounts = {}

    def connect(self, uid, base, provider, key, model, free_confirmed):
        if not free_confirmed:
            raise ValueError("Confirm that this key uses your provider's free plan and that the selected model is available on that plan.")
        key = validate_credentials(provider, key)
        if not isinstance(model, str) or not model or (provider == "openrouter" and not model.endswith(":free")):
            raise ValueError("Select an available chat model. OpenRouter supports only :free models here.")
        catalog = list_provider_models(provider, key)
        if model not in {item["id"] for item in catalog}:
            raise ValueError("This exact chat model was not returned by your provider. Refresh the model list and select one of its models.")
        identity = provider + ":" + model
        selected = {"id": identity, "model": model, "provider": provider,
                    "providerName": PROVIDERS[provider][0], "connection": "personal", "execution": "api"}
        # One immutable settings/client snapshot per provider preserves cooldowns,
        # while jobs already running retain their exact connection if it is changed.
        models = {identity: selected}
        flags = {"groq_free_plan_confirmed": provider == "groq",
                 "gemini_free_tier_confirmed": provider == "gemini",
                 "openrouter_free_only_confirmed": provider == "openrouter"}
        scoped = replace(base, models=models, keys={provider: key}, review_model="", **flags)
        connection = Connection(selected, scoped, ModelAPI(scoped))
        with self.lock:
            self.accounts.setdefault(uid, {})[provider] = connection

    def disconnect(self, uid, provider=None):
        if provider is not None and provider not in PROVIDERS:
            raise ValueError("Choose Groq, Google Gemini, or OpenRouter.")
        with self.lock:
            if provider is None:
                self.accounts.pop(uid, None)
            else:
                account = self.accounts.get(uid, {})
                account.pop(provider, None)
                if not account:
                    self.accounts.pop(uid, None)

    def public(self, uid, base):
        result = base.public()
        with self.lock:
            account = dict(self.accounts.get(uid, {}))
        result["models"] = [model for model in result["models"] if model["provider"] not in account]
        result["models"] += [dict(item.model) for item in account.values()]
        for provider in result["providers"]:
            personal = account.get(provider["id"])
            provider["ready"] = any(model["provider"] == provider["id"] for model in result["models"])
            provider["connection"] = "personal" if personal else "workspace" if provider["ready"] else "none"
            if personal:
                provider["model"] = personal.model["model"]
        result["connectionStorage"] = "API keys are kept in server memory for your account until disconnected or the server restarts."
        return result

    def resolve(self, uid, base, default_api, identity):
        provider = identity.split(":", 1)[0]
        with self.lock:
            personal = self.accounts.get(uid, {}).get(provider)
        if personal:
            personal.settings.require_model(identity)
            return personal.settings, personal.api
        base.require_model(identity)
        return base, default_api
