"""
Model loading utilities for hallucination detection project.
Supports 4-bit quantized loading of Qwen2.5-1.5B, Gemma-2B, and Phi-3-mini
with full hook access to internal activations.
"""

import torch
import gc
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from loguru import logger

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    raise ImportError(
        "transformers and bitsandbytes are required. "
        "Install with: pip install transformers bitsandbytes accelerate"
    )


# ── Model Registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "qwen2",
        "num_layers": 28,
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "num_kv_heads": 2,  # GQA
        "layer_accessor": "model.layers",
        "attn_accessor": "self_attn",
    },
    "gemma-2b": {
        "hf_id": "google/gemma-2b",
        "family": "gemma",
        "num_layers": 18,
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "num_kv_heads": 1,  # MQA
        "layer_accessor": "model.layers",
        "attn_accessor": "self_attn",
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "family": "phi3",
        "num_layers": 32,
        "hidden_size": 3072,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "layer_accessor": "model.layers",
        "attn_accessor": "self_attn",
    },
}


def get_quantization_config(
    quant_type: Literal["4bit", "8bit", "none"] = "4bit",
    compute_dtype: str = "float16",
) -> Optional[BitsAndBytesConfig]:
    """Create a BitsAndBytesConfig for quantized model loading.

    Args:
        quant_type: Quantization level - "4bit", "8bit", or "none".
        compute_dtype: Compute dtype for 4-bit ("float16" or "bfloat16").

    Returns:
        BitsAndBytesConfig or None if quant_type is "none".
    """
    if quant_type == "none":
        return None

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dt = dtype_map.get(compute_dtype, torch.float16)

    if quant_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dt,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_type == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")


def get_model_layers(model, model_name: str):
    """Get the list of transformer layers from a loaded model.

    Args:
        model: Loaded HuggingFace model.
        model_name: Short name from MODEL_REGISTRY.

    Returns:
        nn.ModuleList of transformer layers.
    """
    info = MODEL_REGISTRY[model_name]
    accessor = info["layer_accessor"]

    obj = model
    for attr in accessor.split("."):
        obj = getattr(obj, attr)
    return obj


def get_attention_module(layer, model_name: str):
    """Get the attention module from a transformer layer.

    Args:
        layer: A single transformer layer.
        model_name: Short name from MODEL_REGISTRY.

    Returns:
        The self-attention module.
    """
    info = MODEL_REGISTRY[model_name]
    return getattr(layer, info["attn_accessor"])


class ModelLoader:
    """Handles loading and managing quantized language models with hook access.

    This loader ensures:
    - Models are loaded with bitsandbytes 4-bit quantization (NF4)
    - Activations remain in float16/bfloat16 (unquantized) for hook extraction
    - Proper tokenizer configuration (padding, chat templates)
    - Memory-efficient loading with device_map="auto"

    Usage:
        loader = ModelLoader(model_name="qwen2.5-1.5b", quant_type="4bit")
        model, tokenizer = loader.load()
        layers = loader.get_layers()
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-1.5b",
        quant_type: Literal["4bit", "8bit", "none"] = "4bit",
        compute_dtype: str = "float16",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.model_info = MODEL_REGISTRY[model_name]
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    @property
    def hf_id(self) -> str:
        return self.model_info["hf_id"]

    @property
    def num_layers(self) -> int:
        return self.model_info["num_layers"]

    @property
    def hidden_size(self) -> int:
        return self.model_info["hidden_size"]

    def load(self) -> tuple:
        """Load the model and tokenizer.

        Returns:
            Tuple of (model, tokenizer).
        """
        if self._is_loaded:
            logger.info(f"Model {self.model_name} already loaded, returning cached.")
            return self.model, self.tokenizer

        logger.info(f"Loading {self.model_name} ({self.hf_id}) with {self.quant_type} quantization...")

        # ── Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── Model ──
        quant_config = get_quantization_config(self.quant_type, self.compute_dtype)

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir,
        }

        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            # Full precision — load to specified device
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            load_kwargs["torch_dtype"] = dtype_map.get(self.compute_dtype, torch.float32)
            if self.device != "cpu":
                load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, **load_kwargs
        )
        self.model.eval()

        # Log memory usage
        param_bytes = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        logger.info(
            f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M, "
            f"Memory: {param_bytes / 1024**3:.2f} GB"
        )

        self._is_loaded = True
        return self.model, self.tokenizer

    def get_layers(self):
        """Get transformer layers from the loaded model."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return get_model_layers(self.model, self.model_name)

    def get_strategic_layer_indices(self) -> list:
        """Get strategic layer indices for signal extraction.

        Returns layers at: [0, n//4, n//2, 3n//4, n-1]
        These cover early, early-mid, mid, late-mid, and final layers.
        """
        n = self.num_layers
        indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        return indices

    def unload(self):
        """Unload model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Model {self.model_name} unloaded and memory cleared.")

    def format_prompt(self, question: str, context: str) -> str:
        """Format a RAG-style prompt with question and retrieved context.

        Args:
            question: The user question.
            context: The retrieved document context.

        Returns:
            Formatted prompt string.
        """
        system_msg = (
            "You are a helpful assistant. Answer the question based ONLY on the "
            "provided context. If the context does not contain enough information "
            "to answer, say so."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Try chat template first, fall back to simple format
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: simple format
        return f"{system_msg}\n\n{user_msg}"

    def __repr__(self):
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"ModelLoader(model={self.model_name}, quant={self.quant_type}, "
            f"device={self.device}, status={status})"
        )
