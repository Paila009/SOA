"""Versioned bridge between saved internal signals and the trained research probe."""

from pathlib import Path
import threading


class ProbeAdapter:
    VERSION = "qwen-signals-v2-aligned"
    REQUIRED_KEYS = {"logit_entropy", "token_probs", "hidden_states"}

    def __init__(self, artifact_dir, expected_features=331):
        self.artifact_dir = Path(artifact_dir)
        self.expected_features = expected_features
        self.model_path = self.artifact_dir / "logistic_probe.joblib"
        self.processor_path = self.artifact_dir / "signal_processor.joblib"
        self._artifact_check = None
        self._lock = threading.Lock()

    def predict(self, record):
        """Research score from actual signals, never a blocking decision."""
        import numpy as np
        import joblib
        if record.get('metadata',{}).get('model') != 'qwen2.5-1.5b':
            raise ValueError('Probe requires Qwen2.5-1.5B internal signals')
        with self._lock:
            bundle = joblib.load(self.model_path)
            processor = joblib.load(self.processor_path)
            hidden = record.get('hidden_states',{})
            if set(hidden) != set(processor['pca_models']):
                raise ValueError('Hidden-state layers differ from the fitted processor')
            entropy = np.asarray(record['logit_entropy'],dtype=float)
            probs = np.asarray(record['token_probs'],dtype=float)
            if entropy.size == 0 or probs.shape != entropy.shape:
                raise ValueError('Missing or inconsistent token signals')
            features = [entropy.mean(),entropy.max(),entropy.min(),entropy.std(),np.median(entropy),
                        entropy[-1],np.polyfit(np.arange(len(entropy)),entropy,1)[0] if len(entropy)>1 else 0,
                        probs.mean(),probs.min(),probs.std(),np.median(probs)]
            for layer,pca in sorted(processor['pca_models'].items()):
                vector = np.asarray(hidden[layer],dtype=np.float32).reshape(1,-1)
                if vector.shape[1] != pca.n_features_in_:
                    raise ValueError('Hidden-state width is incompatible')
                features.extend(pca.transform(vector)[0].tolist())
            array = np.asarray(features,dtype=np.float32).reshape(1,-1)
            if array.shape[1] != bundle['scaler'].n_features_in_ or not np.isfinite(array).all():
                raise ValueError('Invalid feature dimensions or nonfinite values')
            score = float(bundle['model'].predict_proba(bundle['scaler'].transform(array))[0,1])
            aligned = self.artifact_dir.name == 'probes-aligned'
            return {'raw_score':score,'features':array.shape[1],'adapter_version':self.VERSION,
                    'validated_probability':False,'used_for_guard':False,
                    'aligned_probe':aligned,
                    'warning':('Aligned held-out classifier score; not a calibrated factuality probability and not used for blocking.'
                               if aligned else 'Legacy labels were inherited from different answers. Research score is unvalidated.')}

    def check_artifacts(self):
        if self._artifact_check is not None:
            return self._artifact_check
        if not self.model_path.exists() or not self.processor_path.exists():
            self._artifact_check = {"compatible": False, "error": "artifact missing"}
            return self._artifact_check
        try:
            import joblib
            model_bundle = joblib.load(self.model_path)
            processor_bundle = joblib.load(self.processor_path)
            feature_count = int(getattr(model_bundle.get("scaler"), "n_features_in_", 0))
            layers = len(processor_bundle.get("pca_models", {}))
            self._artifact_check = {"compatible": feature_count == self.expected_features,
                                    "features": feature_count, "pca_layers": layers, "error": None}
        except Exception as exc:
            self._artifact_check = {"compatible": False, "error": f"{type(exc).__name__}: {exc}"}
        return self._artifact_check

    def validate_signal_record(self, record):
        missing = sorted(self.REQUIRED_KEYS - set(record))
        hidden_states = record.get("hidden_states", {})
        return {"compatible": not missing and bool(hidden_states), "missing": missing,
                "layers": len(hidden_states), "adapter_version": self.VERSION}

    def status(self):
        artifacts_ready = self.model_path.exists() and self.processor_path.exists()
        artifact_check = self.check_artifacts()
        return {"adapter_version": self.VERSION, "artifacts_ready": artifacts_ready,
                "processor_ready": self.processor_path.exists(), "probe_ready": self.model_path.exists(),
                "expected_features": self.expected_features, "live_activation_runtime": (self.artifact_dir.parents[1]/'models-local/qwen-research/model.safetensors').exists(),
                "live_probability_available": False,
                'research_runtime_installed': (self.artifact_dir.parents[1]/'models-local/qwen-research/config.json').exists(),
                'label_validity':('exact annotated-response alignment' if self.artifact_dir.name == 'probes-aligned'
                                  else 'legacy response-label mismatch; re-extraction required'),
                "artifact_compatibility": artifact_check,
                "reason": (("The experimental Qwen option uses an exact-response aligned held-out classifier. "
                            if self.artifact_dir.name == 'probes-aligned' else
                            "The experimental Qwen option currently uses legacy mismatched labels. ") +
                           "Its score is not a calibrated factuality probability and never controls the guard. "
                           "Standard llama.cpp models use the separate evidence verifier." if artifacts_ready else
                           "Trained processor or probe artifact is missing.")}
