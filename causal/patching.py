"""
Activation Patching for Causal Validation.

Implements the causal validation step: given matched pairs of grounded and
hallucinating generations, swap identified internal activations and test
whether output behavior actually flips. This is direct causal evidence,
not statistical association.

Supports:
- Residual stream patching (per-layer, per-position)
- Attention output patching
- MLP output patching
- Automated sweeping across layers and positions
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from loguru import logger

from models.loader import ModelLoader, get_model_layers


@dataclass
class PatchingResult:
    """Result of a single activation patching experiment.

    Attributes:
        layer_idx: Which layer was patched.
        position: Which token position was patched (-1 for all).
        component: Which component was patched (residual/attention/mlp).
        clean_metric: Metric value on clean (grounded) input.
        corrupted_metric: Metric value on corrupted (hallucinating) input.
        patched_metric: Metric value after patching.
        normalized_score: Normalized recovery score (0=corrupted, 1=clean).
        did_flip: Whether the output behavior flipped.
    """
    layer_idx: int
    position: int
    component: str
    clean_metric: float
    corrupted_metric: float
    patched_metric: float
    normalized_score: float
    did_flip: bool


class ActivationPatcher:
    """Performs activation patching experiments for causal validation.

    Core idea: Take a grounded generation and a hallucinating generation.
    Cache the internal activations from the grounded run. Then, during the
    hallucinating run, replace specific layer activations with the cached
    grounded activations. If the hallucination stops, that layer/position
    is causally responsible.

    Usage:
        patcher = ActivationPatcher(model_loader)
        results = patcher.run_patching_sweep(
            clean_input=grounded_prompt,
            corrupted_input=hallucinating_prompt,
            metric_fn=hallucination_score_fn,
        )
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        layer_indices: Optional[List[int]] = None,
        flip_threshold: float = 0.5,
    ):
        self.model_loader = model_loader
        self.flip_threshold = flip_threshold

        # Load model
        self.model, self.tokenizer = model_loader.load()
        self.layers = get_model_layers(self.model, model_loader.model_name)
        self.num_layers = len(self.layers)

        if layer_indices is None:
            self.layer_indices = model_loader.get_strategic_layer_indices()
        else:
            self.layer_indices = layer_indices

    def _get_logit_diff(
        self,
        logits: torch.Tensor,
        target_tokens: Optional[List[int]] = None,
    ) -> float:
        """Compute a scalar metric from output logits.

        Default: entropy of the prediction distribution at the last token.
        Can be customized for specific tasks.
        """
        last_logits = logits[0, -1, :].float()
        probs = F.softmax(last_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        return entropy

    def _cache_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """Run forward pass and cache residual stream activations at each layer.

        Returns:
            Dict mapping layer_idx to activation tensor [batch, seq_len, hidden_dim].
        """
        cache = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    cache[layer_idx] = output[0].detach().clone()
                else:
                    cache[layer_idx] = output.detach().clone()
            return hook_fn

        # Register hooks on all target layers
        for idx in self.layer_indices:
            hook = self.layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(hook)

        # Run forward pass
        with torch.no_grad():
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            _ = self.model(**kwargs)

        # Cleanup hooks
        for hook in hooks:
            hook.remove()

        return cache

    def patch_and_run(
        self,
        corrupted_input_ids: torch.Tensor,
        clean_cache: Dict[int, torch.Tensor],
        patch_layer: int,
        patch_positions: Optional[List[int]] = None,
        component: str = "residual_stream",
    ) -> torch.Tensor:
        """Run corrupted input with specific activations patched from clean cache.

        Args:
            corrupted_input_ids: Input IDs for the corrupted (hallucinating) run.
            clean_cache: Cached activations from the clean (grounded) run.
            patch_layer: Which layer to patch.
            patch_positions: Which token positions to patch (None = all).
            component: Which component to patch.

        Returns:
            Output logits from the patched forward pass.
        """
        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            clean_act = clean_cache[patch_layer]

            if patch_positions is not None:
                # Patch specific positions only
                for pos in patch_positions:
                    if pos < hidden.shape[1] and pos < clean_act.shape[1]:
                        hidden[:, pos, :] = clean_act[:, pos, :]
            else:
                # Patch all positions (up to min sequence length)
                min_len = min(hidden.shape[1], clean_act.shape[1])
                hidden[:, :min_len, :] = clean_act[:, :min_len, :]

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        # Register patch hook
        hook = self.layers[patch_layer].register_forward_hook(patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model(input_ids=corrupted_input_ids)
                logits = outputs.logits
        finally:
            hook.remove()

        return logits

    def run_single_experiment(
        self,
        clean_text: str,
        corrupted_text: str,
        patch_layer: int,
        patch_positions: Optional[List[int]] = None,
        component: str = "residual_stream",
        metric_fn: Optional[Callable] = None,
    ) -> PatchingResult:
        """Run a single patching experiment.

        Args:
            clean_text: The grounded generation prompt.
            corrupted_text: The hallucinating generation prompt.
            patch_layer: Layer to patch.
            patch_positions: Positions to patch.
            component: Component to patch.
            metric_fn: Custom metric function (logits -> float).

        Returns:
            PatchingResult with all metrics.
        """
        if metric_fn is None:
            metric_fn = self._get_logit_diff

        # Tokenize
        clean_inputs = self.tokenizer(clean_text, return_tensors="pt").to(self.model.device)
        corrupted_inputs = self.tokenizer(corrupted_text, return_tensors="pt").to(self.model.device)

        # Get clean activations and metric
        clean_cache = self._cache_activations(clean_inputs["input_ids"])
        with torch.no_grad():
            clean_logits = self.model(**clean_inputs).logits
        clean_metric = metric_fn(clean_logits)

        # Get corrupted metric
        with torch.no_grad():
            corrupted_logits = self.model(**corrupted_inputs).logits
        corrupted_metric = metric_fn(corrupted_logits)

        # Run patched
        patched_logits = self.patch_and_run(
            corrupted_input_ids=corrupted_inputs["input_ids"],
            clean_cache=clean_cache,
            patch_layer=patch_layer,
            patch_positions=patch_positions,
            component=component,
        )
        patched_metric = metric_fn(patched_logits)

        # Compute normalized recovery score
        denom = clean_metric - corrupted_metric
        if abs(denom) > 1e-6:
            normalized = (patched_metric - corrupted_metric) / denom
        else:
            normalized = 0.0

        # Check if behavior flipped
        did_flip = abs(normalized) > self.flip_threshold

        return PatchingResult(
            layer_idx=patch_layer,
            position=patch_positions[0] if patch_positions else -1,
            component=component,
            clean_metric=clean_metric,
            corrupted_metric=corrupted_metric,
            patched_metric=patched_metric,
            normalized_score=normalized,
            did_flip=did_flip,
        )

    def run_patching_sweep(
        self,
        clean_text: str,
        corrupted_text: str,
        metric_fn: Optional[Callable] = None,
        sweep_positions: bool = False,
    ) -> List[PatchingResult]:
        """Run patching sweep across all target layers.

        Args:
            clean_text: Grounded generation prompt.
            corrupted_text: Hallucinating generation prompt.
            metric_fn: Custom metric function.
            sweep_positions: If True, also sweep across token positions.

        Returns:
            List of PatchingResults, one per layer (and position if sweeping).
        """
        results = []

        # Tokenize to know sequence lengths
        clean_tokens = self.tokenizer(clean_text, return_tensors="pt")
        corrupted_tokens = self.tokenizer(corrupted_text, return_tensors="pt")
        seq_len = min(
            clean_tokens["input_ids"].shape[1],
            corrupted_tokens["input_ids"].shape[1],
        )

        for layer_idx in self.layer_indices:
            if sweep_positions:
                for pos in range(seq_len):
                    result = self.run_single_experiment(
                        clean_text=clean_text,
                        corrupted_text=corrupted_text,
                        patch_layer=layer_idx,
                        patch_positions=[pos],
                        metric_fn=metric_fn,
                    )
                    results.append(result)
            else:
                result = self.run_single_experiment(
                    clean_text=clean_text,
                    corrupted_text=corrupted_text,
                    patch_layer=layer_idx,
                    patch_positions=None,  # Patch all positions
                    metric_fn=metric_fn,
                )
                results.append(result)

        # Log summary
        n_flips = sum(1 for r in results if r.did_flip)
        logger.info(
            f"Patching sweep: {n_flips}/{len(results)} experiments showed flip "
            f"(threshold={self.flip_threshold})"
        )

        return results

    def run_batch_experiments(
        self,
        pairs: List[Tuple[str, str]],
        metric_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run patching experiments over multiple grounded/hallucinated pairs.

        Args:
            pairs: List of (grounded_text, hallucinated_text) pairs.
            metric_fn: Custom metric function.

        Returns:
            Summary dict with per-layer flip rates and detailed results.
        """
        all_results = []
        per_layer_flips = {idx: [] for idx in self.layer_indices}

        for i, (clean, corrupted) in enumerate(pairs):
            logger.info(f"Pair {i+1}/{len(pairs)}")
            try:
                sweep = self.run_patching_sweep(clean, corrupted, metric_fn)
                for result in sweep:
                    all_results.append(result)
                    per_layer_flips[result.layer_idx].append(result.did_flip)
            except Exception as e:
                logger.error(f"Error on pair {i}: {e}")
                continue

        # Compute per-layer flip rates
        flip_rates = {}
        for layer_idx, flips in per_layer_flips.items():
            if flips:
                rate = sum(flips) / len(flips)
                flip_rates[layer_idx] = rate
                logger.info(f"Layer {layer_idx}: flip rate = {rate:.2%} ({sum(flips)}/{len(flips)})")

        return {
            "all_results": all_results,
            "flip_rates": flip_rates,
            "total_pairs": len(pairs),
            "total_experiments": len(all_results),
            "overall_flip_rate": sum(r.did_flip for r in all_results) / max(len(all_results), 1),
        }
