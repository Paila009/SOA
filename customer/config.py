import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = {
    "groq": ("Groq", "https://api.groq.com/openai/v1/chat/completions"),
    "gemini": ("Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1/chat/completions"),
}

LOCAL_MODEL_PAGES = {
    "qwen2.5-1.5b": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "phi-3-mini": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf",
    "qwen3-4b": "https://huggingface.co/Qwen/Qwen3-4B-GGUF",
}


def local_model_catalog():
    """Return public metadata only; this never starts a local model."""
    try:
        from dashboard.local_model import MODEL_ROOT, MODEL_SPECS, RUNTIME_ROOT
        engine_installed = any(RUNTIME_ROOT.rglob("llama-server.exe"))
        return [{"id": identity, "chatId": f"local:{identity}", "name": spec["name"],
                 "parameters": spec["parameters"],
                 "quantization": "Q4_K_M", "license": spec["license"],
                 "downloadUrl": LOCAL_MODEL_PAGES.get(identity, ""),
                 "installed": bool(engine_installed and (MODEL_ROOT / spec["file"]).exists()
                                   and (MODEL_ROOT / spec["file"]).stat().st_size >= spec["min_bytes"])}
                for identity, spec in MODEL_SPECS.items()]
    except (ImportError, OSError):
        return []


@dataclass
class Settings:
    environment: str = "development"
    origin: str = "http://127.0.0.1:8770"
    database: Path = ROOT / "outputs/customer/workspace.sqlite3"
    firebase: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    keys: dict = field(default_factory=dict, repr=False)
    review_model: str = ""
    daily_user: int = 30
    daily_total: int = 200
    concurrency: int = 4
    max_tokens: int = 1400
    local_models_enabled: bool = False
    # Owner acknowledgements, not billing-plan checks. Keep each provider free-only.
    groq_free_plan_confirmed: bool = False
    gemini_free_tier_confirmed: bool = False
    openrouter_free_only_confirmed: bool = False

    def require_model(self, identity):
        if identity.startswith("local:"):
            if not self.local_models_enabled:
                raise ValueError("Local models are not enabled in this workspace.")
            runtime_id = identity.split(":", 1)[1]
            item = next((entry for entry in local_model_catalog() if entry["id"] == runtime_id), None)
            if not item or not item["installed"]:
                raise ValueError("This local model and its llama.cpp runtime are not installed on the server computer.")
            return {"id": identity, "model": item["name"], "provider": "local",
                    "providerName": "On this computer", "execution": "local",
                    "runtimeId": runtime_id}
        model = self.models.get(identity)
        if not model:
            raise ValueError("This model is not available in this workspace.")
        provider = model.get("provider")
        confirmed = {
            "groq": self.groq_free_plan_confirmed,
            "gemini": self.gemini_free_tier_confirmed,
            "openrouter": self.openrouter_free_only_confirmed and str(model.get("model", "")).endswith(":free"),
        }.get(provider, False)
        if not confirmed or not self.keys.get(provider):
            raise ValueError("This API model is not enabled for the owner's free-only workspace.")
        return model

    def available_models(self):
        available = []
        if self.local_models_enabled:
            for item in local_model_catalog():
                if item["installed"]:
                    available.append(self.require_model(item["chatId"]))
        for identity in self.models:
            try:
                available.append(self.require_model(identity))
            except ValueError:
                continue
        return available

    @property
    def preview(self):
        return self.environment == "development" and not self.firebase.get("projectId")

    def validate(self):
        parsed = urlparse(self.origin)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
            raise ValueError("PUBLIC_ORIGIN must be an HTTP(S) origin, without a path.")
        if parsed.username or parsed.password:
            raise ValueError("PUBLIC_ORIGIN cannot contain credentials.")
        if self.environment not in {"development", "production"}:
            raise ValueError("APP_ENV must be development or production.")
        firebase_complete = all(self.firebase.get(key) for key in ("apiKey", "authDomain", "projectId", "appId"))
        if self.environment == "production" and (parsed.scheme != "https" or not firebase_complete):
            raise ValueError("Production requires HTTPS and the complete Firebase web configuration.")
        if self.preview and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Unconfigured preview is restricted to localhost.")
        if self.review_model and self.review_model not in self.models:
            raise ValueError("REVIEW_MODEL must match a configured provider:model-id.")
        if self.review_model:
            self.require_model(self.review_model)
        return self

    def public(self):
        available = self.available_models()
        local = local_model_catalog()
        return {"preview": self.preview, "firebase": self.firebase,
                "models": available, "localModels": local,
                "providers": [{"id": p, "name": name, "enabled": True,
                               "ready": any(m["provider"] == p for m in available)}
                              for p, (name, _) in PROVIDERS.items()],
                "limits": {"daily": self.daily_user, "maxTokens": self.max_tokens},
                "review": "API evidence review; not a calibrated hallucination detector."}


def load_settings():
    from dotenv import load_dotenv
    load_dotenv(ROOT / "customer/.env", override=False)
    firebase = {"apiKey": os.getenv("FIREBASE_API_KEY", ""),
                "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
                "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
                "appId": os.getenv("FIREBASE_APP_ID", "")}
    keys, models = {}, {}
    for provider, (label, _) in PROVIDERS.items():
        key = os.getenv(f"{provider.upper()}_API_KEY", "").strip()
        keys[provider] = key
        # Model access changes: require an explicit ID from the owner's Free Plan console.
        default = ""
        for model in os.getenv(f"{provider.upper()}_MODELS", default).split(","):
            model = model.strip()
            if key and model:
                identity = f"{provider}:{model}"
                models[identity] = {"id": identity, "model": model, "provider": provider, "providerName": label}
    db = Path(os.getenv("CUSTOMER_DB", "outputs/customer/workspace.sqlite3"))
    def bounded(name, default, upper):
        return max(1, min(upper, int(os.getenv(name, str(default)))))
    return Settings(environment=os.getenv("APP_ENV", "development"),
                    origin=os.getenv("PUBLIC_ORIGIN", "http://127.0.0.1:8770").rstrip("/"),
                    database=db if db.is_absolute() else ROOT / db,
                    firebase=firebase, models=models, keys=keys,
                    local_models_enabled=os.getenv("ENABLE_LOCAL_MODELS", "true").strip().lower() == "true",
                    groq_free_plan_confirmed=os.getenv("GROQ_FREE_PLAN_CONFIRMED", "false").strip().lower() == "true",
                    gemini_free_tier_confirmed=os.getenv("GEMINI_FREE_TIER_CONFIRMED", "false").strip().lower() == "true",
                    openrouter_free_only_confirmed=os.getenv("OPENROUTER_FREE_ONLY_CONFIRMED", "false").strip().lower() == "true",
                    review_model=os.getenv("REVIEW_MODEL", ""),
                    daily_user=bounded("DAILY_REQUESTS_PER_USER", 30, 10000),
                    daily_total=bounded("DAILY_REQUESTS_TOTAL", 200, 100000),
                    concurrency=bounded("MAX_CONCURRENT_REQUESTS", 4, 16),
                    max_tokens=bounded("MAX_OUTPUT_TOKENS", 1400, 4000)).validate()
