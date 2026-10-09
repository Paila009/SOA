# Do Language Models Know When They Are Guessing?

## Causal Evidence from Internal Uncertainty Signatures in Retrieval-Augmented Generation

**Target:** Conference/Workshop Publication (BlackboxNLP, NeurIPS/ICLR/ACL workshops)  
**Hardware:** HP Victus, 16GB RAM, NVIDIA GPU — Consumer Laptop Scale  
**Research model:** Qwen2.5-1.5B-Instruct. **Local chat choices:** Qwen2.5-1.5B, Phi-3 Mini 3.8B, Qwen3 4B (Q4_K_M). Gemma transfer remains a planned research experiment.

## Customer research workspace — unified local/API edition

A customer-facing application lives in `customer/`. Its single model selector supports installed local GGUF models and Groq, Gemini, or OpenRouter API models. It includes Firebase email/Google login, account-scoped conversations and text-document libraries, streaming answers with Stop, and a saved evidence review attached to each answer. The Models panel lets a signed-in user connect a personal provider key, load that provider's current chat models, select an exact model, and disconnect later. The free-plan confirmation is an acknowledgement; it cannot verify the provider account's billing plan. OpenRouter choices are restricted to zero-cost `:free` text models, with no automatic paid fallback.

Open the public workspace at [paila009.github.io/SOA](https://paila009.github.io/SOA/). GitHub Pages serves the same customer interface in browser mode: Firebase sign-in, personal API connections, streamed chat, source/claim reviews, documents, and history. Provider requests use the user's own key directly from the browser; keys remain in memory and are cleared on refresh or sign-out. Workspace data is kept per account in that browser's IndexedDB rather than synchronized across devices. Firebase must authorize `paila009.github.io` for public Google sign-in, and provider availability/CORS/free quotas still apply.

**Connection status:** Firebase Google login and all three installed local chat models were connected on localhost; a real Qwen smoke response succeeded. No real model-provider key has been supplied for this update, so API generation and provider billing status have not been verified. Local evidence reviews and API reviews are fallible screening methods; neither establishes calibrated hallucination accuracy.

## Run the customer workspace

Run `./run_customer.ps1` after the one-time installation in [the customer setup guide](docs/CUSTOMER_SETUP.md), then open [localhost:8770](http://localhost:8770/). The local workspace runs installed Qwen2.5 1.5B, Phi-3 Mini 3.8B, and Qwen3 4B through llama.cpp in the same viewer as API models. Personal provider keys stay in account-scoped server memory until disconnected or the server restarts. An owner can instead configure shared server keys in the ignored `customer/.env` file; a shared-key public service requires a separately hosted backend.

Normal backend login checks signed Firebase ID tokens against Google's public certificates and requires verified email; it needs the public Firebase configuration, without an Admin service-account file. Set `FIREBASE_CHECK_REVOKED=true` only when adding credential-backed disabled-account/revocation checks.

Every completed answer exposes **Why this verdict** and **View searched sources**, with a visible Answer insights panel showing guard action, evidence coverage, claim reasons, and reviewed passages. Selecting an older answer restores its own saved analysis. Download links in Models point to official local weights; follow the [local model setup guide](docs/LOCAL_MODELS.md) to install the runtime and place those files. GitHub Pages cannot execute GGUF weights in the browser. The older port-8766 dashboard remains available for specialist internal-signal experiments.

## Current Verified Status

The repository includes a runnable localhost research dashboard. The repaired experiment teacher-forces Qwen over all 749 exact annotated responses and uses explicit dataset-ID splits: 522 train, 112 validation, and 115 test examples. The original mismatched artifacts remain preserved as legacy results. See [the F1 audit](docs/F1_AUDIT.md).

| Completed method | Features | Test AUROC | Test F1 |
|---|---:|---:|---:|
| Entropy baseline | 8 | 0.7232 | 0.6496 |
| Full logistic probe | 331 | 0.7979 | 0.7167 |
| MLP probe | 331 | 0.8194 | 0.7611 |

The full-feature pipeline fits PCA only on training hidden states and saves per-example predictions and bootstrap intervals. Labels and evaluated response text are now aligned, but 172 prompts were left-truncated to the 1,024-token extraction window. This is a teacher-forced, same-model held-out classification experiment—not calibrated live-chat accuracy. No causal, cross-model, or mitigation claim is established.

## Run the Dashboard Locally

The dashboard is currently a local application and has not been deployed as a public website.

From PowerShell in this directory:

```powershell
.\run_dashboard.ps1
```

After the server starts, open `http://127.0.0.1:8766` in a browser on the same computer.

The chat runs real local GGUF models through llama.cpp. Select **Qwen2.5 1.5B**, **Phi-3 Mini 3.8B**, or **Qwen3 4B** in the sidebar. The first message to a model loads it into memory and may take longer. For each factual question the dashboard:

1. searches Wikipedia with retry, relevance ranking, deduplication, diagnostics, and a short-lived local cache (comparison questions search each subject separately);
2. sends those passages to the selected local model;
3. checks each claim against one attributable passage using lexical/paraphrase, entity, number/date, citation, and negation signals;
4. reports evidence coverage separately from weakest-claim risk, with supported, partial, contradicted, and unverified outcomes; and
5. blocks the generated answer when the selected risk threshold is exceeded.

The UI also supports cancelling an active request, importing local PDF/text evidence, disabling search for controlled experiments, switching to an extractive evidence-only fallback, comparing installed models under shared settings, and exporting a privacy-safe local experiment history. Guided adversarial demos are included in the research dialog. See [`ENHANCEMENT_TODO.md`](ENHANCEMENT_TODO.md) for the completed implementation checklist and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for privacy, hardware, licensing, and deployment limits.

Aligned probe results are available from the research-results dialog; the legacy artifacts remain on disk for auditability. The live evidence guard uses a local NLI model when installed, with an explicitly labeled lexical fallback. It cannot prove factual truth, and its scores are not calibrated probabilities. The experimental Qwen internal-probe model extracts token entropy, observed-token probabilities, and block activations, then applies the fitted 331-feature processor. Its raw classifier score never controls blocking. Standard llama.cpp models do not expose these internal features.

Install optional local models with `python scripts/setup_verifier.py` (NLI) and `python scripts/setup_verifier.py --research` (experimental Qwen). They require transformers, torch, and safetensors; weights remain ignored by Git. Run `python scripts/evaluate_live_verifier.py` for 20 authored diagnostic cases. These development cases are not an independent accuracy benchmark. See [the reliability checklist](ENHANCEMENT_TODO.md) for remaining validation work.

Chat uses background inference jobs with token/claim events. Stop requests close llama.cpp streaming connections or interrupt Transformers between forward passes. Loading and an in-progress forward pass may finish before cancellation takes effect. Comparisons retrieve once and share a hashed evidence snapshot and generation settings; wording overlap is not factual agreement.

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
python scripts\evaluate_detectors.py
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
