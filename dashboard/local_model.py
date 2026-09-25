"""Run actual local GGUF chat models using the bundled llama.cpp server."""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "runtime" / "llama-cpp"
MODEL_ROOT = ROOT / "models-local"
MODEL_SPECS = {
    "qwen2.5-1.5b": {"name": "Qwen2.5 1.5B Instruct", "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "parameters": "1.54B", "port": 8081, "license": "Apache-2.0", "min_bytes": 1_100_000_000},
    "phi-3-mini": {"name": "Phi-3 Mini 3.8B", "file": "Phi-3-mini-4k-instruct-q4.gguf", "parameters": "3.8B", "port": 8082, "license": "MIT", "min_bytes": 2_350_000_000},
    "qwen3-4b": {"name": "Qwen3 4B", "file": "Qwen3-4B-Q4_K_M.gguf", "parameters": "4B", "port": 8083, "license": "Apache-2.0", "min_bytes": 2_450_000_000},
}


class LocalModelRuntime:
    def __init__(self, model_id, spec):
        self.model_id, self.spec = model_id, spec
        self.model_path = MODEL_ROOT / spec["file"]
        self.url = f"http://127.0.0.1:{spec['port']}"
        self.process, self.loading, self.error = None, False, None
        self._lock = threading.Lock()

    @property
    def executable(self):
        matches = list(RUNTIME_ROOT.rglob("llama-server.exe"))
        return matches[0] if matches else None

    @property
    def installed(self):
        return bool(self.executable and self.model_path.exists()
                    and self.model_path.stat().st_size >= self.spec["min_bytes"])

    def is_ready(self):
        try:
            with urlopen(f"{self.url}/health", timeout=0.6) as response:
                return response.status == 200
        except (OSError, URLError):
            return False

    def status(self):
        ready = self.is_ready()
        return {
            "installed": self.installed, "ready": ready, "loading": self.loading and not ready,
            "error": self.error, "model_path": str(self.model_path),
            "model_bytes": self.model_path.stat().st_size if self.model_path.exists() else 0,
            "engine": "llama.cpp", "model": self.spec["name"], "parameters": self.spec["parameters"],
            "quantization": "Q4_K_M", "device": "CPU", "endpoint": self.url,
            "process_id": self.process.pid if self.process and self.process.poll() is None else None,
        }

    def ensure_started(self, background=True):
        if self.is_ready():
            return
        if background:
            if not self.loading:
                threading.Thread(target=self._start, daemon=True).start()
        else:
            self._start()

    def _start(self):
        with self._lock:
            if self.is_ready():
                return
            self.loading, self.error = True, None
            try:
                executable = self.executable
                if not executable or not self.installed:
                    raise RuntimeError(f"{self.spec['name']} runtime or model weights are not installed")
                command = [str(executable), "-m", str(self.model_path), "--host", "127.0.0.1",
                           "--port", str(self.spec["port"]), "-c", "4096", "-t",
                           str(max(2, (os.cpu_count() or 4) - 2)), "--parallel", "1"]
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.process = subprocess.Popen(command, cwd=str(executable.parent),
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                creationflags=flags)
                deadline = time.time() + 120
                while time.time() < deadline:
                    if self.is_ready():
                        return
                    if self.process.poll() is not None:
                        raise RuntimeError(f"llama.cpp exited with code {self.process.returncode}")
                    time.sleep(0.5)
                raise RuntimeError(f"{self.spec['name']} loading timed out")
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.loading = False

    def _complete(self, messages, max_tokens=320, temperature=0.15, timeout=180):
        if not self.is_ready():
            self.ensure_started(background=False)
        if not self.is_ready():
            raise RuntimeError(self.error or f"{self.spec['name']} is not ready")
        body = json.dumps({"model": self.spec["file"], "messages": messages,
                           "temperature": temperature, "top_p": 0.9,
                           "max_tokens": max_tokens, "stream": False}).encode("utf-8")
        request = Request(f"{self.url}/v1/chat/completions", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        answer = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if self.model_id == "qwen3-4b":
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        if not answer:
            raise RuntimeError("The local model returned an empty answer")
        return answer, payload.get("usage", {})

    def chat(self, question, context, max_tokens=240, comparison=False):
        system = (
            "You are a grounded research assistant. Answer using ONLY the numbered sources supplied. "
            "Cite every factual sentence with source numbers such as [1]. If the sources do not "
            "answer the question, say: The searched sources do not contain enough evidence to "
            "answer reliably. Use at most three short factual sentences. Do not invent names, "
            "dates, numbers, or citations. "
            "Do not cite an unrelated source. Do not copy source passages or add a Sources section."
        )
        if comparison:
            system += (" For a comparison, write exactly two concise bullet lines: one per subject. "
                       "Each line must state a directly sourced fact and cite its own source. "
                       "The contrast should be clear from the two lines. Do not add a broad "
                       "introductory or concluding claim. If one subject lacks evidence, say so "
                       "on that line. Do not mix facts across subjects.")
        user = f"SOURCES:\n{context}\n\nQUESTION:\n{question}\n\nAnswer in at most 100 words. Put a valid source citation after every factual sentence."
        if self.model_id == "qwen3-4b":
            user += " /no_think"
        answer, usage = self._complete([{"role": "system", "content": system},
                                        {"role": "user", "content": user}], max_tokens=max_tokens)
        answer = re.split(r"\n\s*(?:Sources?|References?)\s*:", answer, maxsplit=1,
                          flags=re.IGNORECASE)[0].strip()
        return answer, usage

    def chat_general(self, question, history=None, max_tokens=120):
        messages = [{"role": "system", "content": (
            "You are a concise and friendly local assistant. Respond naturally to greetings "
            "and casual conversation. Do not invent facts, claim that you searched, or add "
            "citations. Invite the user to ask a question.")}]
        for item in (history or [])[-6:]:
            role, content = item.get("role"), str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content[:1200]})
        messages.append({"role": "user", "content": question + (" /no_think" if self.model_id == "qwen3-4b" else "")})
        return self._complete(messages, max_tokens=max_tokens, temperature=0.35, timeout=120)

    def classify_intent(self, question):
        answer, _ = self._complete([{"role": "system", "content": (
            "Classify the user's message. Reply with exactly CHAT for greetings, small talk, "
            "feelings, thanks, or questions about the assistant/conversation. Reply with exactly "
            "SEARCH for factual questions, explanations, research, people, places, events, "
            "comparisons, or claims that should use external evidence.")},
            {"role": "user", "content": question}], max_tokens=4, temperature=0.0, timeout=60)
        return "chat" if "CHAT" in answer.upper() else "search"

    def rewrite_search_query(self, question):
        answer, _ = self._complete([{"role": "system", "content": (
            "Rewrite the user's question as a short Wikipedia search query. Correct obvious "
            "spelling mistakes in names. Return only search terms; no explanation.")},
            {"role": "user", "content": question}], max_tokens=24, temperature=0.0, timeout=60)
        return " ".join(answer.strip().strip('"\'` .?!').split())[:120]


MODEL_RUNTIMES = {model_id: LocalModelRuntime(model_id, spec)
                  for model_id, spec in MODEL_SPECS.items()}
MODEL_RUNTIME = MODEL_RUNTIMES["qwen2.5-1.5b"]
