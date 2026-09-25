"""
Main Signal Extraction Pipeline.

Runs the base language model over the RAG dataset, recording internal signals
(attention, hidden states, entropy) at each generation step. Signals are saved
to disk for subsequent probe training and analysis.
"""

import torch
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from loguru import logger

from models.loader import ModelLoader
from extraction.hooks import (
    HookManager,
    ExtractionResult,
    compute_logit_entropy,
    compute_top_k_probs,
)


class SignalExtractor:
    """Orchestrates signal extraction from a language model over a dataset.

    For each example in the dataset:
    1. Formats the RAG prompt (question + context)
    2. Runs the model with hooks to capture internal signals
    3. Records logit entropy and token probabilities during generation
    4. Saves all signals to disk

    Usage:
        extractor = SignalExtractor(
            model_loader=ModelLoader("qwen2.5-1.5b"),
            layer_indices=[0, 7, 14, 21, 27],
            output_dir="outputs/signals",
        )
        extractor.run(dataset)
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        layer_indices: Optional[List[int]] = None,
        output_dir: str = "outputs/signals",
        max_new_tokens: int = 256,
        capture_attention: bool = True,
        capture_hidden_states: bool = True,
    ):
        self.model_loader = model_loader
        self.layer_indices = layer_indices
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.capture_attention = capture_attention
        self.capture_hidden_states = capture_hidden_states

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be initialized on first run
        self.model = None
        self.tokenizer = None
        self.hook_manager = None

    def _setup(self):
        """Load model and initialize hooks."""
        if self.model is None:
            self.model, self.tokenizer = self.model_loader.load()

            if self.layer_indices is None:
                self.layer_indices = self.model_loader.get_strategic_layer_indices()

            self.hook_manager = HookManager(
                model=self.model,
                model_name=self.model_loader.model_name,
                layer_indices=self.layer_indices,
                capture_hidden_states=self.capture_hidden_states,
                capture_attention=self.capture_attention,
                device="cpu",  # Store signals on CPU to save GPU memory
            )

    def extract_single(
        self,
        question: str,
        context: str,
        example_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract signals for a single RAG example.

        Args:
            question: The question to answer.
            context: The retrieved document context.
            example_id: Optional identifier for this example.

        Returns:
            Dict containing extraction results and metadata.
        """
        self._setup()

        # Format prompt
        prompt = self.model_loader.format_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Attach hooks and run generation
        self.hook_manager.clear()
        self.hook_manager.attach()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy for reproducibility
                    temperature=1.0,
                    output_scores=True,
                    output_attentions=self.capture_attention,
                    return_dict_in_generate=True,
                )
        finally:
            self.hook_manager.detach()

        # Collect hook results
        result = self.hook_manager.collect()

        # Process generation outputs
        generated_ids = outputs.sequences[0, prompt_len:].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        result.generated_token_ids = generated_ids

        # Compute logit entropy from generation scores
        if hasattr(outputs, "scores") and outputs.scores:
            # scores is a tuple of [batch, vocab_size] tensors, one per generated token
            all_logits = torch.stack(outputs.scores, dim=0)  # [num_tokens, batch, vocab]
            all_logits = all_logits[:, 0, :]  # [num_tokens, vocab]

            result.logit_entropy = compute_logit_entropy(all_logits)
            top_probs, top_indices = compute_top_k_probs(all_logits, k=5)
            result.token_probs = top_probs[:, 0]  # Top-1 probability per step
            result.top_k_probs = top_probs

        # Metadata
        result.metadata = {
            "example_id": example_id,
            "question": question,
            "context_preview": context[:200],
            "generated_text": generated_text,
            "prompt_length": prompt_len,
            "generation_length": len(generated_ids),
            "model": self.model_loader.model_name,
        }

        return {
            "result": result,
            "generated_text": generated_text,
            "example_id": example_id,
        }

    def run(
        self,
        dataset: List[Dict[str, Any]],
        resume_from: int = 0,
        save_every: int = 10,
    ) -> List[str]:
        """Run extraction over the full dataset.

        Args:
            dataset: List of dicts with keys "question", "context", "label", and optional "id".
            resume_from: Index to resume extraction from.
            save_every: Save checkpoint every N examples.

        Returns:
            List of output file paths.
        """
        self._setup()
        output_files = []

        logger.info(
            f"Starting signal extraction: {len(dataset)} examples, "
            f"starting from {resume_from}"
        )

        for idx in tqdm(range(resume_from, len(dataset)), desc="Extracting signals"):
            example = dataset[idx]
            example_id = example.get("id", f"example_{idx:04d}")

            try:
                start_time = time.time()
                extraction = self.extract_single(
                    question=example["question"],
                    context=example["context"],
                    example_id=example_id,
                )
                elapsed = time.time() - start_time

                # Save signals
                save_path = self.output_dir / f"{example_id}.pt"
                save_data = {
                    "features": extraction["result"].to_feature_dict(),
                    "logit_entropy": extraction["result"].logit_entropy,
                    "token_probs": extraction["result"].token_probs,
                    "hidden_states": extraction["result"].hidden_states,
                    "attention_entropy": extraction["result"].attention_entropy,
                    "label": int(example.get("is_hallucinated", example.get("label", 0))),
                    "generated_text": extraction["generated_text"],
                    "metadata": extraction["result"].metadata,
                    "extraction_time_s": elapsed,
                }
                torch.save(save_data, save_path)
                output_files.append(str(save_path))

                if (idx + 1) % save_every == 0:
                    logger.info(
                        f"Extracted {idx + 1}/{len(dataset)} examples. "
                        f"Last took {elapsed:.1f}s"
                    )

            except Exception as e:
                logger.error(f"Error extracting example {example_id}: {e}")
                # Save error log but continue
                error_path = self.output_dir / f"{example_id}_error.json"
                with open(error_path, "w") as f:
                    json.dump({"example_id": example_id, "error": str(e)}, f)
                continue

        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest = {
            "model": self.model_loader.model_name,
            "num_examples": len(output_files),
            "layer_indices": self.layer_indices,
            "max_new_tokens": self.max_new_tokens,
            "files": output_files,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Extraction complete. {len(output_files)} files saved to {self.output_dir}")
        return output_files
