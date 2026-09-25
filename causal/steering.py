"""
Activation Steering for Hallucination Mitigation.

When the probe flags a likely hallucination, this module applies lightweight
activation edits — nudging the internal state toward the 'grounded' direction
found during causal analysis — as a real-time fix during generation.

Implements:
1. Steering direction computation from contrastive grounded/hallucinated pairs
2. Runtime steering hook that adds the direction vector during generation
3. Strength tuning (alpha parameter sweep)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
from loguru import logger

from models.loader import ModelLoader, get_model_layers


class ActivationSteerer:
    """Computes and applies activation steering vectors for hallucination mitigation.

    The steering direction is computed as the mean difference between
    grounded and hallucinated internal representations:
        v_steer = mean(h_grounded) - mean(h_hallucinated)

    During generation, when the probe flags high hallucination risk,
    this vector is added to the residual stream:
        h' = h + alpha * v_steer

    This nudges the model toward grounded behavior.

    Usage:
        steerer = ActivationSteerer(model_loader)
        steerer.compute_direction(grounded_texts, hallucinated_texts, layer=14)
        with steerer.apply_steering(alpha=2.0):
            output = model.generate(...)
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        target_layer: Optional[int] = None,
    ):
        self.model_loader = model_loader
        self.model, self.tokenizer = model_loader.load()
        self.layers = get_model_layers(self.model, model_loader.model_name)

        # Auto-select target layer (mid-to-late layers work best)
        if target_layer is None:
            n = len(self.layers)
            self.target_layer = 3 * n // 4  # Default: 75% depth
        else:
            self.target_layer = target_layer

        self.steering_vector = None
        self._direction_computed = False

    def _extract_layer_activation(
        self,
        text: str,
        layer_idx: int,
        token_position: str = "last",
    ) -> torch.Tensor:
        """Extract activation at a specific layer and position.

        Args:
            text: Input text.
            layer_idx: Which layer.
            token_position: "last" (last token), "mean" (average), or "all".

        Returns:
            Activation tensor of shape [hidden_dim] or [seq_len, hidden_dim].
        """
        captured = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured["act"] = output[0].detach()
            else:
                captured["act"] = output.detach()

        hook = self.layers[layer_idx].register_forward_hook(hook_fn)

        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            hook.remove()

        act = captured["act"][0]  # [seq_len, hidden_dim]

        if token_position == "last":
            return act[-1]  # [hidden_dim]
        elif token_position == "mean":
            return act.mean(dim=0)  # [hidden_dim]
        else:
            return act  # [seq_len, hidden_dim]

    def compute_direction(
        self,
        grounded_texts: List[str],
        hallucinated_texts: List[str],
        layer_idx: Optional[int] = None,
        method: str = "difference_in_means",
    ):
        """Compute the steering direction from contrastive examples.

        Args:
            grounded_texts: List of grounded generation texts.
            hallucinated_texts: List of hallucinated generation texts.
            layer_idx: Layer to compute direction at (None = use target_layer).
            method: "difference_in_means" or "pca".
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        logger.info(
            f"Computing steering direction at layer {layer_idx} "
            f"from {len(grounded_texts)} grounded + {len(hallucinated_texts)} hallucinated examples"
        )

        # Extract activations for both classes
        grounded_acts = []
        for text in grounded_texts:
            act = self._extract_layer_activation(text, layer_idx, "last")
            grounded_acts.append(act)

        hallucinated_acts = []
        for text in hallucinated_texts:
            act = self._extract_layer_activation(text, layer_idx, "last")
            hallucinated_acts.append(act)

        grounded_mean = torch.stack(grounded_acts).mean(dim=0)
        hallucinated_mean = torch.stack(hallucinated_acts).mean(dim=0)

        if method == "difference_in_means":
            # Direction: grounded - hallucinated
            # Adding this vector pushes toward grounded behavior
            direction = grounded_mean - hallucinated_mean
        elif method == "pca":
            # PCA on the concatenated activations
            all_acts = torch.stack(grounded_acts + hallucinated_acts)
            # Center
            centered = all_acts - all_acts.mean(dim=0)
            # SVD
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            direction = Vh[0]  # First principal component
            # Orient so it points from hallucinated to grounded
            if (grounded_mean - hallucinated_mean) @ direction < 0:
                direction = -direction
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize
        self.steering_vector = direction / direction.norm()
        self._direction_computed = True

        # Log cosine similarity between direction and class means
        cos_sim = F.cosine_similarity(
            self.steering_vector.unsqueeze(0),
            (grounded_mean - hallucinated_mean).unsqueeze(0),
        ).item()
        logger.info(
            f"Steering direction computed (norm={direction.norm():.4f}). "
            f"Cosine similarity with class difference: {cos_sim:.4f}"
        )

    @contextmanager
    def apply_steering(
        self,
        alpha: float = 2.0,
        layer_idx: Optional[int] = None,
        apply_to_all_positions: bool = True,
    ):
        """Context manager that applies steering during generation.

        Args:
            alpha: Steering strength multiplier. Larger = stronger push.
            layer_idx: Layer to steer at (None = target_layer).
            apply_to_all_positions: If True, steer at all positions; if False, only last.

        Usage:
            with steerer.apply_steering(alpha=2.0):
                output = model.generate(input_ids, max_new_tokens=256)
        """
        if not self._direction_computed:
            raise RuntimeError("Call compute_direction() first.")

        if layer_idx is None:
            layer_idx = self.target_layer

        vec = self.steering_vector.to(self.model.device, dtype=self.model.dtype)

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                if apply_to_all_positions:
                    modified = hidden + alpha * vec
                else:
                    modified = hidden.clone()
                    modified[:, -1, :] = modified[:, -1, :] + alpha * vec
                return (modified,) + output[1:]
            else:
                if apply_to_all_positions:
                    return output + alpha * vec
                else:
                    modified = output.clone()
                    modified[:, -1, :] = modified[:, -1, :] + alpha * vec
                    return modified

        hook = self.layers[layer_idx].register_forward_hook(steering_hook)
        try:
            yield
        finally:
            hook.remove()

    def generate_steered(
        self,
        prompt: str,
        alpha: float = 2.0,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> str:
        """Generate text with activation steering applied.

        Args:
            prompt: Input prompt.
            alpha: Steering strength.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with self.apply_steering(alpha=alpha):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **generate_kwargs,
            )

        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def sweep_alpha(
        self,
        prompt: str,
        alpha_values: List[float] = None,
        max_new_tokens: int = 256,
    ) -> Dict[float, str]:
        """Generate with multiple steering strengths for comparison.

        Args:
            prompt: Input prompt.
            alpha_values: List of alpha values to test.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Dict mapping alpha -> generated text.
        """
        if alpha_values is None:
            alpha_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

        results = {}
        for alpha in alpha_values:
            if alpha == 0.0:
                # No steering baseline
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            else:
                text = self.generate_steered(prompt, alpha=alpha, max_new_tokens=max_new_tokens)

            results[alpha] = text
            logger.info(f"Alpha={alpha:.1f}: {text[:100]}...")

        return results
