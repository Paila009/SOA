"""
Probe Classifiers for Hallucination Detection.

Implements three levels of probe complexity:
1. Entropy-only baseline (logistic regression on entropy features only)
2. Full logistic regression probe (all signal features)
3. MLP probe (non-linear, for complex feature interactions)

All probes are deliberately small — strong performance reflects
signal informativeness, not classifier capacity.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    classification_report,
)


# ── Feature Extraction Utilities ─────────────────────────────────────────────

def extract_entropy_features(signal_data: Dict[str, Any]) -> np.ndarray:
    """Extract entropy-only features from saved signal data.

    Features: [mean_entropy, max_entropy, std_entropy, final_entropy,
               min_token_prob, mean_token_prob, std_token_prob]

    Args:
        signal_data: Dict loaded from a .pt signal file.

    Returns:
        1D numpy array of entropy features.
    """
    features = []

    # Logit entropy features
    entropy = signal_data.get("logit_entropy")
    if entropy is not None:
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.numpy()
        features.extend([
            float(np.mean(entropy)),
            float(np.max(entropy)),
            float(np.std(entropy)),
            float(entropy[-1]),  # Final token entropy
            float(np.median(entropy)),
        ])
    else:
        features.extend([0.0] * 5)

    # Token probability features
    token_probs = signal_data.get("token_probs")
    if token_probs is not None:
        if isinstance(token_probs, torch.Tensor):
            token_probs = token_probs.numpy()
        features.extend([
            float(np.min(token_probs)),
            float(np.mean(token_probs)),
            float(np.std(token_probs)),
        ])
    else:
        features.extend([0.0] * 3)

    return np.array(features, dtype=np.float32)


def extract_full_features(
    signal_data: Dict[str, Any],
    pca_dim: Optional[int] = 64,
    pca_model: Optional[Any] = None,
) -> np.ndarray:
    """Extract all features (entropy + attention + hidden states) from signal data.

    Args:
        signal_data: Dict loaded from a .pt signal file.
        pca_dim: Dimensionality for PCA on hidden states (None = no reduction).
        pca_model: Pre-fitted PCA model (None = skip hidden state features).

    Returns:
        1D numpy array of all features.
    """
    features = []

    # 1. Entropy features (same as baseline)
    features.extend(extract_entropy_features(signal_data))

    # 2. Attention entropy features (per-layer statistics)
    attn_entropy = signal_data.get("attention_entropy", {})
    for layer_idx in sorted(attn_entropy.keys()):
        ae = attn_entropy[layer_idx]
        if isinstance(ae, torch.Tensor):
            ae = ae.numpy()
        # Per-layer attention entropy: mean, max, std across heads
        features.extend([
            float(np.mean(ae)),
            float(np.max(ae)),
            float(np.std(ae)),
        ])

    # 3. Hidden state features
    hidden_states = signal_data.get("hidden_states", {})
    if hidden_states:
        for layer_idx in sorted(hidden_states.keys()):
            hs = hidden_states[layer_idx]
            if isinstance(hs, torch.Tensor):
                hs = hs.numpy()

            if pca_model is not None:
                # PCA-reduced last token
                last_hs = hs[-1:, :]  # [1, hidden_dim]
                reduced = pca_model.transform(last_hs).flatten()
                features.extend(reduced.tolist())
            else:
                # Direct statistics of last token hidden state
                last_hs = hs[-1, :]  # [hidden_dim]
                features.extend([
                    float(np.mean(last_hs)),
                    float(np.std(last_hs)),
                    float(np.max(last_hs)),
                    float(np.min(last_hs)),
                    float(np.linalg.norm(last_hs)),
                    float(np.median(last_hs)),
                    float(np.percentile(last_hs, 25)),
                    float(np.percentile(last_hs, 75)),
                ])

    # 4. Pre-computed feature dict (if available)
    feat_dict = signal_data.get("features", {})
    if isinstance(feat_dict, dict):
        # Add entropy trajectory and other pre-computed stats
        for key in sorted(feat_dict.keys()):
            val = feat_dict[key]
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, torch.Tensor):
                features.append(float(val.item()) if val.numel() == 1 else float(val.mean().item()))

    return np.array(features, dtype=np.float32)


# ── Entropy-Only Baseline ────────────────────────────────────────────────────

class EntropyBaseline:
    """Logistic regression baseline using only entropy/probability features.

    This is the 'dumb' baseline that the richer probes must beat.
    If they can't, the extra signals aren't adding value.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.model = LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 10),
            cv=5,
            penalty="l2",
            scoring="roc_auc",
            max_iter=max_iter,
            random_state=42,
            class_weight="balanced",
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the entropy baseline.

        Args:
            X: Feature matrix [n_samples, n_features].
            y: Binary labels [n_samples] (0=grounded, 1=hallucinated).
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info(
            f"EntropyBaseline fitted. Best C: {self.model.C_[0]:.4f}, "
            f"Train AUROC (CV): {self.model.scores_[1].mean(axis=0).max():.4f}"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get hallucination probability predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data.

        Returns:
            Dict with precision, recall, f1, auroc, auprc.
        """
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0
        )

        results = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auroc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs)),
        }

        logger.info(
            f"EntropyBaseline eval: AUROC={results['auroc']:.4f}, "
            f"F1={results['f1']:.4f}, P={results['precision']:.4f}, "
            f"R={results['recall']:.4f}"
        )
        return results

    def save(self, path: str):
        """Save model to disk."""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)
        logger.info(f"EntropyBaseline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EntropyBaseline":
        """Load model from disk."""
        import joblib
        data = joblib.load(path)
        probe = cls()
        probe.scaler = data["scaler"]
        probe.model = data["model"]
        probe._is_fitted = True
        return probe


# ── Logistic Regression Probe ────────────────────────────────────────────────

class LogisticProbe:
    """Full logistic regression probe using all signal features.

    Uses L2 regularization with cross-validated C selection.
    """

    def __init__(self, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.model = LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 15),
            cv=5,
            penalty="l2",
            scoring="roc_auc",
            max_iter=max_iter,
            random_state=42,
            class_weight="balanced",
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the logistic probe."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info(
            f"LogisticProbe fitted. Best C: {self.model.C_[0]:.4f}, "
            f"Features: {X.shape[1]}"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data."""
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0
        )
        results = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auroc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs)),
        }
        logger.info(
            f"LogisticProbe eval: AUROC={results['auroc']:.4f}, "
            f"F1={results['f1']:.4f}"
        )
        return results

    def save(self, path: str):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "LogisticProbe":
        import joblib
        data = joblib.load(path)
        probe = cls()
        probe.scaler = data["scaler"]
        probe.model = data["model"]
        probe._is_fitted = True
        return probe


# ── MLP Probe ────────────────────────────────────────────────────────────────

class MLPProbeNet(nn.Module):
    """Small MLP for hallucination detection probe.

    Architecture: Input -> [Linear -> LayerNorm -> GELU -> Dropout] x N -> Linear -> Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPProbe:
    """MLP-based probe for hallucination detection.

    Trains a small neural network on extracted signal features.
    Kept deliberately small (< 100K parameters) so strong performance
    reflects signal informativeness, not classifier capacity.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = early_stopping_patience
        self.device = device

        self.scaler = StandardScaler()
        self.model = None
        self._is_fitted = False

    def _build_model(self, input_dim: int):
        """Build the MLP model."""
        self.input_dim = input_dim
        self.model = MLPProbeNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train the MLP probe.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for early stopping).
            y_val: Validation labels.

        Returns:
            Training history dict with loss and metric trajectories.
        """
        # Make the small neural probe reproducible across runs.
        np.random.seed(42)
        torch.manual_seed(42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        # Build model
        self._build_model(X_train_scaled.shape[1])

        # Convert to tensors
        X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        # Class weights for imbalanced data
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(self.device)

        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_auroc": []}
        best_val_auroc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            indices = torch.randperm(len(X_tensor))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_train_loss)

            # Validation
            if X_val_scaled is not None:
                val_metrics = self._evaluate_internal(X_val_scaled, y_val)
                history["val_loss"].append(val_metrics["loss"])
                history["val_auroc"].append(val_metrics["auroc"])
                scheduler.step(val_metrics["loss"])

                # Early stopping
                if val_metrics["auroc"] > best_val_auroc:
                    best_val_auroc = val_metrics["auroc"]
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs}: "
                        f"train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"val_auroc={val_metrics['auroc']:.4f}"
                    )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_fitted = True
        logger.info(f"MLPProbe training complete. Best val AUROC: {best_val_auroc:.4f}")
        return history

    def _evaluate_internal(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Internal evaluation on already-scaled data."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            loss = nn.BCEWithLogitsLoss()(logits, y_tensor).item()
            probs = torch.sigmoid(logits).cpu().numpy()

        auroc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0
        return {"loss": loss, "auroc": auroc}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get hallucination probability predictions."""
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data."""
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0
        )
        results = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auroc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs)),
        }
        logger.info(
            f"MLPProbe eval: AUROC={results['auroc']:.4f}, F1={results['f1']:.4f}"
        )
        return results

    def save(self, path: str):
        """Save model, scaler, and config."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MLPProbe":
        data = torch.load(path, map_location=device, weights_only=False)
        config = data["config"]
        probe = cls(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            device=device,
        )
        probe._build_model(config["input_dim"])
        probe.model.load_state_dict(data["model_state"])
        probe.scaler = data["scaler"]
        probe._is_fitted = True
        return probe
