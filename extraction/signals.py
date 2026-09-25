"""
Signal Processing & Feature Engineering.

Converts raw extracted signals (hidden states, attention weights, entropy)
into compact feature vectors suitable for probe training.

Includes:
- PCA dimensionality reduction for hidden states
- Feature normalization
- Feature aggregation strategies (mean, max, std, last token)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from loguru import logger

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    PCA = None
    StandardScaler = None


class SignalProcessor:
    """Processes raw extracted signals into feature vectors for probe training.

    Handles:
    1. Hidden state dimensionality reduction via PCA
    2. Feature aggregation across tokens (mean, max, std, last)
    3. Attention pattern statistics
    4. Entropy trajectory features
    """

    def __init__(
        self,
        pca_dim: int = 64,
        aggregation: List[str] = None,
    ):
        self.pca_dim = pca_dim
        self.aggregation = aggregation or ["mean", "max", "std", "last"]
        self.pca_models: Dict[int, Any] = {}  # Per-layer PCA models
        self.scaler = StandardScaler() if StandardScaler else None
        self._is_fitted = False

    def fit(self, signal_files: List[str]):
        """Fit PCA models on training signal data.

        Args:
            signal_files: List of .pt signal file paths.
        """
        logger.info(f"Fitting SignalProcessor on {len(signal_files)} files...")

        # Collect hidden states per layer for PCA
        layer_hidden_states: Dict[int, List[np.ndarray]] = {}

        for fp in signal_files:
            try:
                data = torch.load(fp, map_location="cpu", weights_only=False)
                hidden_states = data.get("hidden_states", {})
                for layer_idx, hs in hidden_states.items():
                    if isinstance(hs, torch.Tensor):
                        hs = hs.numpy()
                    # Use last token representation
                    last_token = hs[-1:, :]  # [1, hidden_dim]
                    layer_hidden_states.setdefault(layer_idx, []).append(last_token)
            except Exception as e:
                logger.warning(f"Error loading {fp}: {e}")
                continue

        # Fit PCA per layer
        if PCA is not None:
            for layer_idx, hs_list in layer_hidden_states.items():
                if not hs_list:
                    continue
                X = np.vstack(hs_list)  # [n_samples, hidden_dim]
                actual_dim = min(self.pca_dim, X.shape[0], X.shape[1])
                pca = PCA(
                    n_components=actual_dim,
                    svd_solver="randomized",
                    random_state=42,
                )
                pca.fit(X)
                self.pca_models[layer_idx] = pca
                logger.info(
                    f"  Layer {layer_idx}: PCA {X.shape[1]} -> {actual_dim} "
                    f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})"
                )

        self._is_fitted = True
        logger.info("SignalProcessor fitted.")

    def process_single(self, signal_data: Dict[str, Any]) -> np.ndarray:
        """Process a single signal file into a feature vector.

        Args:
            signal_data: Loaded .pt signal dict.

        Returns:
            1D feature vector (numpy array).
        """
        features = []

        # 1. Entropy features (always present)
        entropy = signal_data.get("logit_entropy")
        if entropy is not None:
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.numpy()
            features.extend([
                float(np.mean(entropy)),
                float(np.max(entropy)),
                float(np.min(entropy)),
                float(np.std(entropy)),
                float(np.median(entropy)),
                float(entropy[-1]) if len(entropy) > 0 else 0.0,
                # Entropy trajectory slope (is it increasing?)
                float(np.polyfit(np.arange(len(entropy)), entropy, 1)[0])
                if len(entropy) > 1 else 0.0,
            ])
        else:
            features.extend([0.0] * 7)

        # 2. Token probability features
        token_probs = signal_data.get("token_probs")
        if token_probs is not None:
            if isinstance(token_probs, torch.Tensor):
                token_probs = token_probs.numpy()
            features.extend([
                float(np.mean(token_probs)),
                float(np.min(token_probs)),
                float(np.std(token_probs)),
                float(np.median(token_probs)),
            ])
        else:
            features.extend([0.0] * 4)

        # 3. Attention entropy features (per-layer)
        attn_entropy = signal_data.get("attention_entropy", {})
        for layer_idx in sorted(attn_entropy.keys()):
            ae = attn_entropy[layer_idx]
            if isinstance(ae, torch.Tensor):
                ae = ae.numpy()
            features.extend([
                float(np.mean(ae)),
                float(np.max(ae)),
                float(np.std(ae)),
            ])

        # 4. Hidden state features (PCA-reduced)
        hidden_states = signal_data.get("hidden_states", {})
        for layer_idx in sorted(hidden_states.keys()):
            hs = hidden_states[layer_idx]
            if isinstance(hs, torch.Tensor):
                hs = hs.numpy()

            if layer_idx in self.pca_models:
                # PCA-reduced last token
                last_token = hs[-1:, :]
                reduced = self.pca_models[layer_idx].transform(last_token).flatten()
                features.extend(reduced.tolist())
            else:
                # Fallback: basic statistics of last token hidden state
                last_token = hs[-1, :]
                features.extend([
                    float(np.mean(last_token)),
                    float(np.std(last_token)),
                    float(np.max(np.abs(last_token))),
                ])

        return np.array(features, dtype=np.float32)

    def process_batch(self, signal_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of signal files.

        Args:
            signal_files: List of .pt file paths.

        Returns:
            Tuple of (feature_matrix [n_samples, n_features], labels [n_samples]).
        """
        features = []
        labels = []

        for fp in signal_files:
            try:
                data = torch.load(fp, map_location="cpu", weights_only=False)
                feat = self.process_single(data)
                features.append(feat)
                labels.append(int(data.get("label", 0)))
            except Exception as e:
                logger.warning(f"Error processing {fp}: {e}")
                continue

        X = np.stack(features)
        y = np.array(labels)
        logger.info(f"Processed {len(features)} files -> feature matrix {X.shape}")
        return X, y

    def save(self, path: str):
        """Save fitted processor to disk."""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pca_models": self.pca_models,
            "pca_dim": self.pca_dim,
            "aggregation": self.aggregation,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SignalProcessor":
        """Load fitted processor from disk."""
        import joblib
        data = joblib.load(path)
        proc = cls(pca_dim=data["pca_dim"], aggregation=data["aggregation"])
        proc.pca_models = data["pca_models"]
        proc._is_fitted = True
        return proc
