"""Owner-only readiness check. No network requests, secrets, or account changes."""
import os

from .config import load_settings


def readiness(settings):
    available = settings.available_models()
    return {
        "Usable chat model available": bool(available),
        "Installed local chat model available": any(model["provider"] == "local" for model in available),
        "Groq API key saved (optional)": bool(settings.keys.get("groq")),
        "Gemini API key saved (optional)": bool(settings.keys.get("gemini")),
        "OpenRouter API key saved (optional)": bool(settings.keys.get("openrouter")),
        "Firebase web configuration complete": all(settings.firebase.get(k) for k in ("apiKey", "authDomain", "projectId", "appId")),
        "Firebase Admin credential path configured": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()),
    }


def main():
    try:
        checks = readiness(load_settings())
    except ValueError:
        print("Configuration is invalid. Check customer/.env using docs/CUSTOMER_SETUP.md.")
        return 2
    for label, ready in checks.items():
        print(f"{'READY' if ready else 'NEEDED'}: {label}")
    print("No model was called. Credential contents, billing plan and real login are not verified by this check.")
    print("Local chat needs no API key. API providers require their matching free-only confirmation; no paid fallback is enabled.")
    essential = checks["Usable chat model available"] and checks["Firebase web configuration complete"]
    return 0 if essential else 2


if __name__ == "__main__":
    raise SystemExit(main())
