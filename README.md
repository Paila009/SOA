# Do Language Models Know When They Are Guessing?

## Causal Evidence from Internal Uncertainty Signatures in Retrieval-Augmented Generation

**Target:** Conference/Workshop Publication (BlackboxNLP, NeurIPS/ICLR/ACL workshops)  
**Hardware:** HP Victus, 16GB RAM, NVIDIA GPU — Consumer Laptop Scale  
**Research model:** Qwen2.5-1.5B-Instruct. **Local chat choices:** Qwen2.5-1.5B, Phi-3 Mini 3.8B, Qwen3 4B (Q4_K_M). Gemma transfer remains a planned research experiment.

## Current Verified Status

The repository now includes a runnable localhost research dashboard and corrected probe training. The completed Qwen experiment uses 749 saved signal artifacts and explicit dataset-ID splits: 522 train, 112 validation, and 115 test examples.

| Completed method | Features | Test AUROC | Test F1 |
|---|---:|---:|---:|
| Entropy baseline | 8 | 0.6999 | 0.6610 |
| Full logistic probe | 331 | 0.7154 | 0.6552 |
| MLP probe | 331 | 0.6688 | 0.6429 |

The full-feature pipeline fits PCA only on training hidden states, applies the same processor to validation and test records, saves per-example predictions, and reports 1,000-sample bootstrap confidence intervals. The current result supports a modest predictive improvement in AUROC for the logistic probe. It does not yet support causal, cross-model, or mitigation claims.

## Run the Dashboard

From PowerShell in this directory:

```powershell
.\run_dashboard.ps1
```

Then open [http://127.0.0.1:8766](http://127.0.0.1:8766).

The chat runs real local GGUF models through llama.cpp. Select **Qwen2.5 1.5B**, **Phi-3 Mini 3.8B**, or **Qwen3 4B** in the sidebar. The first message to a model loads it into memory and may take longer. For each factual question the dashboard:

1. searches Wikipedia for attributable source passages (comparison questions search each subject separately);
2. sends those passages to the selected local model;
3. checks each claim against a single passage and checks explicit source numbers, negation, and numbers;
4. calculates answer support, query relevance, hallucination risk, latency, and unsupported-sentence counts; and
5. blocks the generated answer when the selected risk threshold is exceeded.

The UI also supports cancelling an active request, adding optional private evidence, disabling search for controlled experiments, and switching to an extractive evidence-only fallback. Verified probe results are available from the research-results dialog. The live text-overlap guard is **not** the trained activation probe and cannot prove factual truth; it can miss paraphrases and subtle contradictions. The offline probe's AUROC/F1 must not be presented as live-chat accuracy.

The added weights come from the [official Microsoft Phi-3 Mini GGUF](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) (MIT) and [official Qwen3 4B GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) (Apache-2.0) repositories. They run locally; no paid model API is used. Searches still require internet access.

## Reproduce Probe Training

The project-local runtime is stored in `.runtime`. Run any corrected probe with:

```powershell
.\run_training.ps1 -Type entropy-only
.\run_training.ps1 -Type logistic
.\run_training.ps1 -Type mlp
```

Consolidate the completed results with:

```powershell
python scripts\run_benchmark.py
```

Result artifacts are written to `outputs/probes` and `outputs/results`.

---

## Project Overview

This project investigates whether hallucination in RAG systems can be predicted **during a single generation pass** by probing internal model signals, and goes beyond correlation with:

1. **Causal validation** via activation patching
2. **Cross-model generalization** testing (Qwen → Gemma)
3. **Active mitigation** via activation steering
4. **Benchmarking** against SelfCheckGPT and semantic entropy baselines

## Repository Structure

```
hallucination-detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── configs/                     # Configuration files
│   └── default.yaml             # Default experiment config
├── data/                        # Data handling
│   ├── __init__.py
│   ├── download.py              # Download RAGTruth dataset
│   ├── subset.py                # Create stratified subset
│   └── gold_subset.py           # Gold subset verification tools
├── models/                      # Model loading & management
│   ├── __init__.py
│   └── loader.py                # Load quantized models with hooks
├── extraction/                  # Signal extraction pipeline
│   ├── __init__.py
│   ├── hooks.py                 # PyTorch hooks for internals
│   ├── extractor.py             # Main extraction pipeline
│   └── signals.py               # Signal processing & features
├── probes/                      # Probe classifiers
│   ├── __init__.py
│   ├── entropy_baseline.py      # Entropy-only baseline
│   ├── logistic_probe.py        # Logistic regression probe
│   └── mlp_probe.py             # MLP probe
├── causal/                      # Causal validation
│   ├── __init__.py
│   ├── patching.py              # Activation patching
│   └── steering.py              # Activation steering
├── baselines/                   # Comparison baselines
│   ├── __init__.py
│   ├── selfcheckgpt.py          # SelfCheckGPT implementation
│   └── semantic_entropy.py      # Semantic entropy baseline
├── evaluation/                  # Evaluation & benchmarking
│   ├── __init__.py
│   ├── metrics.py               # Precision, recall, F1, AUROC
│   ├── benchmark.py             # Head-to-head comparison
│   └── cost_analysis.py         # Compute cost comparison
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_analysis.ipynb
│   ├── 03_probe_training.ipynb
│   ├── 04_causal_experiments.ipynb
│   └── 05_final_results.ipynb
├── scripts/                     # Runnable scripts
│   ├── run_extraction.py        # Run signal extraction
│   ├── train_probe.py           # Train probe classifier
│   ├── run_patching.py          # Run activation patching
│   ├── run_steering.py          # Run activation steering
│   └── run_benchmark.py         # Run full benchmark
└── paper/                       # Paper draft
    ├── main.tex
    └── figures/
```

## Quick Start

```bash
# Create environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset
python -m data.download

# Run signal extraction (GPU recommended)
python scripts/run_extraction.py --model qwen2.5-1.5b --subset-size 500

# Train entropy baseline
python scripts/train_probe.py --type entropy-only

# Train full probe
python scripts/train_probe.py --type mlp
```

## Team Roles

| Member | Role | Key Deliverable |
|--------|------|----------------|
| Member 1 | Data & Ground-Truth Lead | ~500-1000 subset + ~50-80 gold examples |
| Member 2 | Signal Extraction & Causal Analysis | Hooks pipeline + activation patching |
| Member 3 | Probe Modeling & Generalization | Probe classifiers + cross-model transfer |
| Member 4 | Mitigation & Benchmarking | Activation steering + baseline comparison |

## Timeline

- **Weeks 1-2:** Data prep + signal extraction pipeline
- **Week 3:** Entropy-only baseline (critical checkpoint)
- **Weeks 4-5:** Full probe training + causal patching
- **Week 6:** Cross-model transfer + steering implementation
- **Weeks 7-8:** Full benchmark
- **Weeks 9-10:** Paper writing

## Hardware Requirements

- **Minimum:** 16GB RAM, any NVIDIA GPU (4GB+ VRAM)
- **Recommended:** 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Fallback:** Free-tier Google Colab/Kaggle (T4 GPU) for extraction batches
