"""Local passage/claim inference. Weights are downloaded only by the setup script."""
import os
import threading
from functools import lru_cache
from pathlib import Path

MODEL_ID = 'cross-encoder/nli-MiniLM2-L6-H768'
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models-local' / 'nli-minilm'


class NLIRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = self.tokenizer = None
        self.error = None

    def status(self):
        return {'model': MODEL_ID, 'installed': (MODEL_PATH / 'model.safetensors').exists(),
                'loaded': self.model is not None, 'error': self.error,
                'method': 'natural-language-inference', 'calibrated': False}

    def load(self):
        if self.model is not None:
            return True
        if not self.status()['installed'] or os.getenv('GROUNDED_NLI', '1') == '0':
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            torch.set_num_threads(min(4, os.cpu_count() or 2))
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH), local_files_only=True).eval()
            self.error = None
            return True
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            return False

    @lru_cache(maxsize=1024)
    def predict(self, passage, claim):
        with self.lock:
            if not self.load():
                return None
            import torch
            encoded = self.tokenizer(passage, claim, return_tensors='pt', truncation=True, max_length=384)
            with torch.inference_mode():
                scores = self.model(**encoded).logits.softmax(-1)[0].tolist()
            # Label order is documented by this specific model's publisher.
            return dict(zip(('contradiction', 'entailment', 'neutral'), map(float, scores)))


NLI = NLIRuntime()
