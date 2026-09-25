"""
SelfCheckGPT Baseline Implementation.

Wraps the selfcheckgpt library to provide a consistent interface
for comparison against the internal probing approach.

SelfCheckGPT detects hallucination by sampling multiple outputs and
measuring consistency — if the model 'knows' the answer, repeated
samples will agree; if it's hallucinating, they'll diverge.
"""

import torch
import time
from typing import List, Dict, Any, Optional
from loguru import logger


class SelfCheckGPTBaseline:
    """SelfCheckGPT baseline for hallucination detection.

    Generates K additional samples and measures consistency using NLI.
    Higher inconsistency score -> more likely hallucination.

    This is the main comparison target: it requires K extra forward passes
    (expensive), while our probe requires 0 extra passes.
    """

    def __init__(
        self,
        model_loader,
        num_samples: int = 5,
        method: str = "nli",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        self.model_loader = model_loader
        self.num_samples = num_samples
        self.method = method
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None
        self.selfcheck = None

    def _setup(self):
        """Load model and SelfCheckGPT module."""
        if self.model is None:
            self.model, self.tokenizer = self.model_loader.load()

        if self.selfcheck is None:
            try:
                if self.method == "nli":
                    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
                    device = next(self.model.parameters()).device
                    self.selfcheck = SelfCheckNLI(device=device)
                elif self.method == "bertscore":
                    from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
                    device = next(self.model.parameters()).device
                    self.selfcheck = SelfCheckBERTScore(device=device)
                else:
                    logger.warning(f"Method '{self.method}' not supported, falling back to manual NLI")
                    self.selfcheck = None
            except ImportError:
                logger.warning("selfcheckgpt not installed. Using manual implementation.")
                self.selfcheck = None

    def _generate_samples(
        self,
        prompt: str,
        num_samples: int,
    ) -> List[str]:
        """Generate multiple stochastic samples for consistency checking."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        samples = []

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                )
            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            samples.append(text)

        return samples

    def _segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except (ImportError, OSError):
            # Fallback: simple sentence splitting
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def score_single(
        self,
        prompt: str,
        target_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score a single example for hallucination.

        Args:
            prompt: The RAG prompt.
            target_response: Pre-generated response to check (if None, generates one).

        Returns:
            Dict with hallucination score, per-sentence scores, timing, and cost info.
        """
        self._setup()

        start_time = time.time()

        # Generate target if not provided
        if target_response is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            target_response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Generate stochastic samples
        samples = self._generate_samples(prompt, self.num_samples)

        # Segment target into sentences
        sentences = self._segment_sentences(target_response)

        # Score
        sentence_scores = []
        if self.selfcheck is not None and sentences:
            try:
                scores = self.selfcheck.predict(
                    sentences=sentences,
                    sampled_passages=samples,
                )
                sentence_scores = list(scores)
            except Exception as e:
                logger.error(f"SelfCheckGPT scoring error: {e}")
                sentence_scores = [0.5] * len(sentences)
        else:
            sentence_scores = [0.5] * len(sentences)

        elapsed = time.time() - start_time

        # Aggregate
        passage_score = float(sum(sentence_scores) / max(len(sentence_scores), 1))

        return {
            "hallucination_score": passage_score,
            "sentence_scores": sentence_scores,
            "sentences": sentences,
            "target_response": target_response,
            "num_samples": self.num_samples,
            "forward_passes": 1 + self.num_samples,  # 1 target + K samples
            "wall_time_s": elapsed,
            "method": f"selfcheck_{self.method}",
        }


class SemanticEntropyBaseline:
    """Semantic Entropy baseline for hallucination detection.

    Samples K responses, clusters them by semantic equivalence (using NLI),
    and computes Shannon entropy over the cluster distribution.
    High entropy = diverse meanings = likely hallucination.

    Based on: Kuhn et al., "Semantic Uncertainty" (ICLR 2023)
    """

    def __init__(
        self,
        model_loader,
        num_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        entailment_threshold: float = 0.5,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
    ):
        self.model_loader = model_loader
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.entailment_threshold = entailment_threshold
        self.nli_model_name = nli_model_name

        self.model = None
        self.tokenizer = None
        self.nli_model = None
        self.nli_tokenizer = None

    def _setup(self):
        """Load models."""
        if self.model is None:
            self.model, self.tokenizer = self.model_loader.load()

        if self.nli_model is None:
            try:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
                self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
                device = next(self.model.parameters()).device
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                    self.nli_model_name
                ).to(device)
                self.nli_model.eval()
                logger.info(f"Loaded NLI model: {self.nli_model_name}")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                raise

    def _check_entailment(self, text_a: str, text_b: str) -> bool:
        """Check bidirectional entailment between two texts."""
        if text_a.strip().lower() == text_b.strip().lower():
            return True

        device = next(self.nli_model.parameters()).device

        pairs = [(text_a, text_b), (text_b, text_a)]
        inputs = self.nli_tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        # Label mapping varies by model; usually entailment is index 0 or 2
        # For cross-encoder models: 0=contradiction, 1=entailment, 2=neutral
        entail_idx = 1  # Adjust based on model
        entail_a_b = probs[0, entail_idx].item() > self.entailment_threshold
        entail_b_a = probs[1, entail_idx].item() > self.entailment_threshold

        return entail_a_b and entail_b_a

    def _cluster_samples(self, samples: List[str]) -> List[List[int]]:
        """Cluster samples by bidirectional entailment."""
        clusters = []
        for i, sample in enumerate(samples):
            assigned = False
            for cluster in clusters:
                rep = samples[cluster[0]]
                if self._check_entailment(sample, rep):
                    cluster.append(i)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
        return clusters

    def score_single(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """Compute semantic entropy for a single prompt.

        Returns:
            Dict with semantic_entropy, num_clusters, timing, cost info.
        """
        import numpy as np

        self._setup()
        start_time = time.time()

        # Generate samples
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        samples = []
        sample_log_probs = []

        for _ in range(self.num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            gen_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            samples.append(text)

            # Compute log probability of the sequence
            if hasattr(outputs, "scores") and outputs.scores:
                log_prob = 0.0
                for t, score in enumerate(outputs.scores):
                    if t < len(gen_ids):
                        token_probs = torch.log_softmax(score[0], dim=-1)
                        log_prob += token_probs[gen_ids[t]].item()
                sample_log_probs.append(log_prob)

        # Cluster samples
        clusters = self._cluster_samples(samples)

        # Compute semantic entropy
        K = len(samples)
        if sample_log_probs:
            # Probability-weighted
            probs = np.exp(np.array(sample_log_probs) - np.max(sample_log_probs))
            probs = probs / np.sum(probs)
            cluster_probs = [sum(probs[idx] for idx in cluster) for cluster in clusters]
        else:
            # Frequency-based
            cluster_probs = [len(cluster) / K for cluster in clusters]

        se = -sum(p * np.log(p + 1e-12) for p in cluster_probs if p > 0)

        elapsed = time.time() - start_time

        return {
            "semantic_entropy": float(se),
            "num_clusters": len(clusters),
            "num_samples": self.num_samples,
            "cluster_sizes": [len(c) for c in clusters],
            "samples": samples,
            "forward_passes": self.num_samples,
            "wall_time_s": elapsed,
            "method": "semantic_entropy",
        }
