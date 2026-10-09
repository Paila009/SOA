"""Streaming provider adapter. No automatic retries of billable requests."""
import json
import re
import threading
import time
import uuid
import httpx
from .config import PROVIDERS


class Cancelled(Exception):
    pass


class ProviderError(Exception):
    pass


class RateLimitError(ProviderError):
    def __init__(self, seconds):
        self.retry_after = max(1, int(seconds))
        super().__init__(f"The model's free-tier limit was reached. Try again in {self.retry_after} seconds. No paid fallback was used.")


class Job:
    def __init__(self, owner, chat_id):
        self.id = uuid.uuid4().hex
        self.owner, self.chat_id = owner, chat_id
        self.created = time.time()
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.closer = None
        self.state = {"id": self.id, "chatId": chat_id, "phase": "queued", "text": "", "done": False,
                      "sources": [], "error": None, "createdAt": self.created, "updatedAt": self.created}

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled()

    def update(self, **values):
        with self.lock:
            values["updatedAt"] = time.time()
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            closer = self.closer
        if closer:
            try:
                closer()
            except Exception:
                pass

    def attach(self, closer):
        """Compatibility hook for cancellable local llama.cpp streams."""
        with self.lock:
            self.closer = closer

    def detach(self):
        with self.lock:
            self.closer = None


def parse_sse(lines):
    """OpenAI-compatible providers use one JSON object per data line."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            return
        try:
            payload = json.loads(value)
        except ValueError:
            continue
        if payload.get("error"):
            raise ProviderError("The model provider rejected the response. Check your provider console.")
        for choice in payload.get("choices", []):
            text = choice.get("delta", {}).get("content")
            if isinstance(text, str) and text:
                yield text


class ModelAPI:
    def __init__(self, settings):
        self.settings = settings
        self.cooldown_until = 0
        self.cooldown_lock = threading.Lock()

    def check_ready(self):
        with self.cooldown_lock:
            remaining = self.cooldown_until - time.monotonic()
        if remaining > 0:
            raise RateLimitError(int(remaining) + 1)

    def complete(self, model_id, messages, job, on_token=None, max_tokens=None):
        try:
            model = self.settings.require_model(model_id)
        except ValueError as exc:
            raise ProviderError(str(exc)) from None
        provider = model["provider"]
        if provider == "local":
            return self._complete_local(model, messages, job, on_token, max_tokens)
        self.check_ready()
        headers = {"Authorization": "Bearer " + self.settings.keys[provider], "Content-Type": "application/json"}
        if provider == "gemini":
            headers["x-goog-api-client"] = "grounded-research-oai/1.0"
        if provider == "openrouter":
            headers["X-Title"] = "Grounded Research Workspace"
        body = {"model": model["model"], "messages": messages, "stream": True,
                "temperature": 0.2, "max_tokens": max_tokens or self.settings.max_tokens}
        content, started = "", time.monotonic()
        try:
            job.check()
            with httpx.Client(timeout=httpx.Timeout(40, connect=10), follow_redirects=False) as client:
                with client.stream("POST", PROVIDERS[provider][1], headers=headers, json=body) as response:
                    if response.status_code == 429:
                        # A shared account quota applies across customers and both API calls.
                        raw = response.headers.get("retry-after", "60")
                        seconds = min(86400, max(1, int(raw))) if raw.isascii() and raw.isdecimal() and len(raw) < 8 else 60
                        with self.cooldown_lock:
                            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
                        raise RateLimitError(seconds)
                    if response.status_code != 200:
                        labels = {401: "API key was not accepted", 403: "model access was denied", 404: "model ID was not found", 429: "rate limit or credit limit was reached"}
                        raise ProviderError(f"{model['providerName']}: {labels.get(response.status_code, 'request failed')} (HTTP {response.status_code}). Check the server configuration and provider account.")
                    with job.lock:
                        job.closer = response.close
                    job.check()
                    for token in parse_sse(response.iter_lines()):
                        job.check()
                        if time.monotonic() - started > 150 or len(content) + len(token) > 40000:
                            raise ProviderError("The model response exceeded this workspace's time or size limit.")
                        content += token
                        if on_token:
                            on_token(content)
            job.check()
            if not content.strip():
                raise ProviderError("The provider returned an empty answer. Try another configured model.")
            return content
        except httpx.HTTPError:
            job.check()
            raise ProviderError("The model provider could not be reached. Please try again later.") from None
        finally:
            with job.lock:
                job.closer = None

    def _complete_local(self, model, messages, job, on_token=None, max_tokens=None):
        """Use the same installed llama.cpp runtimes without leaving this workspace."""
        try:
            from dashboard.local_model import MODEL_RUNTIMES
            runtime = MODEL_RUNTIMES[model["runtimeId"]]
            chunks = []

            def publish(token):
                chunks.append(token)
                if on_token:
                    on_token("".join(chunks))

            answer, _usage = runtime._complete(
                messages,
                max_tokens=min(max_tokens or self.settings.max_tokens, 700),
                temperature=0.2,
                timeout=240,
                on_token=publish,
                job=job,
            )
            if model["runtimeId"] == "qwen3-4b":
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            if not answer:
                raise ProviderError("The local model returned an empty answer.")
            return answer
        except Cancelled:
            raise
        except (KeyError, RuntimeError, OSError) as exc:
            raise ProviderError(f"The local model could not answer: {exc}") from None
