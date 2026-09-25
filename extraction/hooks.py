"""
Signal Extraction Hooks for Internal Model State Recording.

Provides PyTorch forward hooks to capture attention weights, hidden states,
and logit entropy from transformer layers during inference. These signals
form the input features for the hallucination probe classifier.

Designed to work with bitsandbytes 4-bit quantized models — activations
remain in float16/bfloat16 regardless of weight quantization.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ExtractionResult:
    """Container for extracted signals from a single generation pass.

    Attributes:
        hidden_states: Dict mapping layer_idx -> tensor of shape [seq_len, hidden_dim].
        attention_weights: Dict mapping layer_idx -> tensor of shape [num_heads, seq_len, seq_len].
        attention_entropy: Dict mapping layer_idx -> tensor of shape [num_heads, seq_len].
        logit_entropy: Tensor of shape [num_generated_tokens] with per-step prediction entropy.
        token_probs: Tensor of shape [num_generated_tokens] with per-step top-1 probability.
        top_k_probs: Tensor of shape [num_generated_tokens, k] with per-step top-k probabilities.
        generated_token_ids: List of generated token IDs.
        metadata: Additional metadata dict.
    """
    hidden_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_weights: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_entropy: Dict[int, torch.Tensor] = field(default_factory=dict)
    logit_entropy: Optional[torch.Tensor] = None
    token_probs: Optional[torch.Tensor] = None
    top_k_probs: Optional[torch.Tensor] = None
    generated_token_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_feature_dict(self) -> Dict[str, torch.Tensor]:
        """Convert extraction result to a flat feature dictionary for probe training."""
        features = {}

        # Hidden state features (per layer)
        for layer_idx, hs in self.hidden_states.items():
            # hs shape: [seq_len, hidden_dim]
            features[f"hidden_state_l{layer_idx}_mean"] = hs.mean(dim=0)
            features[f"hidden_state_l{layer_idx}_last"] = hs[-1]
            features[f"hidden_state_l{layer_idx}_std"] = hs.std(dim=0)

        # Attention entropy features (per layer)
        for layer_idx, ae in self.attention_entropy.items():
            # ae shape: [num_heads, seq_len]
            features[f"attn_entropy_l{layer_idx}_mean"] = ae.mean()
            features[f"attn_entropy_l{layer_idx}_max"] = ae.max()
            features[f"attn_entropy_l{layer_idx}_std"] = ae.std()
            features[f"attn_entropy_l{layer_idx}_per_head_mean"] = ae.mean(dim=1)

        # Logit entropy features
        if self.logit_entropy is not None:
            features["logit_entropy_mean"] = self.logit_entropy.mean()
            features["logit_entropy_max"] = self.logit_entropy.max()
            features["logit_entropy_std"] = self.logit_entropy.std()
            features["logit_entropy_last"] = self.logit_entropy[-1]
            features["logit_entropy_trajectory"] = self.logit_entropy

        # Token probability features
        if self.token_probs is not None:
            features["token_prob_mean"] = self.token_probs.mean()
            features["token_prob_min"] = self.token_probs.min()
            features["token_prob_std"] = self.token_probs.std()

        return features


class HookManager:
    """Manages PyTorch forward hooks for extracting internal model signals.

    This class registers hooks on specified transformer layers to capture:
    1. Hidden states (residual stream outputs)
    2. Attention weights (post-softmax attention patterns)
    3. Per-layer output for entropy computation

    Hooks are registered via context manager or explicit attach/detach.

    Usage:
        hook_mgr = HookManager(model, model_name="qwen2.5-1.5b", layer_indices=[0, 7, 14, 21, 27])
        hook_mgr.attach()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        signals = hook_mgr.collect()
        hook_mgr.detach()
    """

    def __init__(
        self,
        model,
        model_name: str,
        layer_indices: Optional[List[int]] = None,
        capture_hidden_states: bool = True,
        capture_attention: bool = True,
        device: str = "cpu",
    ):
        from models.loader import get_model_layers, MODEL_REGISTRY

        self.model = model
        self.model_name = model_name
        self.model_info = MODEL_REGISTRY[model_name]
        self.device = device

        # Get layers
        self.layers = get_model_layers(model, model_name)
        self.num_layers = len(self.layers)

        # Determine which layers to hook
        if layer_indices is None:
            # Default: strategic layers
            n = self.num_layers
            self.layer_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        else:
            self.layer_indices = sorted(layer_indices)

        self.capture_hidden_states = capture_hidden_states
        self.capture_attention = capture_attention

        # Storage for captured activations
        self._hidden_states: Dict[int, List[torch.Tensor]] = {}
        self._attention_weights: Dict[int, List[torch.Tensor]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._is_attached = False

        logger.info(
            f"HookManager initialized for {model_name} | "
            f"Layers: {self.layer_indices} | "
            f"Capture: hidden_states={capture_hidden_states}, attention={capture_attention}"
        )

    def _make_hidden_state_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that captures hidden states."""
        def hook_fn(module, input, output):
            # Transformer layer output is typically (hidden_states, ...) tuple
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Detach, move to CPU to save GPU memory
            self._hidden_states.setdefault(layer_idx, []).append(
                hidden.detach().to(self.device).float()
            )
        return hook_fn

    def _make_attention_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that captures attention weights."""
        from models.loader import get_attention_module

        def hook_fn(module, input, output):
            # For attention modules, output is typically (attn_output, attn_weights, ...)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self._attention_weights.setdefault(layer_idx, []).append(
                        attn_weights.detach().to(self.device).float()
                    )
        return hook_fn

    def attach(self):
        """Register all hooks on the model."""
        if self._is_attached:
            logger.warning("Hooks already attached. Call detach() first.")
            return

        self.clear()

        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            if self.capture_hidden_states:
                hook = layer.register_forward_hook(
                    self._make_hidden_state_hook(layer_idx)
                )
                self._hooks.append(hook)

            if self.capture_attention:
                attn_module = getattr(layer, self.model_info["attn_accessor"])
                hook = attn_module.register_forward_hook(
                    self._make_attention_hook(layer_idx)
                )
                self._hooks.append(hook)

        self._is_attached = True
        logger.debug(f"Attached {len(self._hooks)} hooks")

    def detach(self):
        """Remove all hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._is_attached = False
        logger.debug("All hooks detached")

    def clear(self):
        """Clear captured activation storage."""
        self._hidden_states.clear()
        self._attention_weights.clear()

    def collect(self) -> ExtractionResult:
        """Collect all captured signals into an ExtractionResult.

        Returns:
            ExtractionResult with captured hidden states and attention data.
        """
        result = ExtractionResult()
        eps = 1e-9

        # Process hidden states
        for layer_idx, hs_list in self._hidden_states.items():
            if hs_list:
                # Concatenate across generation steps
                # Each entry: [batch=1, seq_len, hidden_dim] — take last step's full sequence
                result.hidden_states[layer_idx] = hs_list[-1][0]  # [seq_len, hidden_dim]

        # Process attention weights and compute attention entropy
        for layer_idx, aw_list in self._attention_weights.items():
            if aw_list:
                # Take the last generation step's attention
                # Shape: [batch=1, num_heads, seq_len, seq_len]
                attn = aw_list[-1][0]  # [num_heads, seq_len, seq_len]
                result.attention_weights[layer_idx] = attn

                # Compute entropy along key dimension
                # H = -sum(p * log(p)) for each (head, query_position)
                attn_entropy = -torch.sum(
                    attn * torch.log(attn + eps), dim=-1
                )  # [num_heads, seq_len]
                result.attention_entropy[layer_idx] = attn_entropy

        return result

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()
        return False


def compute_logit_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute prediction entropy from logits.

    Args:
        logits: Tensor of shape [batch, seq_len, vocab_size] or [seq_len, vocab_size].
        temperature: Temperature for softmax (1.0 = standard).

    Returns:
        Entropy tensor of shape [seq_len] in nats.
    """
    if logits.dim() == 3:
        logits = logits[0]  # Remove batch dim

    eps = 1e-9
    probs = F.softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    return entropy


def compute_top_k_probs(
    logits: torch.Tensor, k: int = 5
) -> tuple:
    """Compute top-k probabilities from logits.

    Args:
        logits: Tensor of shape [seq_len, vocab_size].
        k: Number of top probabilities to return.

    Returns:
        Tuple of (top_probs [seq_len, k], top_indices [seq_len, k]).
    """
    if logits.dim() == 3:
        logits = logits[0]

    probs = F.softmax(logits.float(), dim=-1)
    top_probs, top_indices = probs.topk(k, dim=-1)
    return top_probs, top_indices
