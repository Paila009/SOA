# Running and deployment

## Local-first default

Run `./run_dashboard.ps1` from PowerShell and open `http://127.0.0.1:8766`.
The host and port can be changed with `GROUNDED_HOST` and `GROUNDED_PORT`.
Keep `127.0.0.1` for private laptop use. Binding to `0.0.0.0` exposes the
unencrypted research server to the local network and should only be done behind
an authenticated reverse proxy.

## Models and hardware

GGUF weights remain local and are excluded from Git. The included Q4 models run
on CPU but 4B inference can be slow on laptops. Model licenses are listed in the
selector metadata; verify the upstream license before redistribution.

## Privacy and security

At the user's request, new experiment-ledger entries store the complete question,
answer, reviewed evidence passages, claim analysis, model/settings, scores, and
timings in `outputs/history/experiments.jsonl` so older analyses can be reopened.
This file remains local and Git-ignored; clear it separately if local retention is
not desired. Imported documents are processed in memory,
limited to 8 MB and 100 PDF pages, and are not sent to an external upload API.
This localhost server has no login, TLS, or multi-user isolation.

## Container note

Containerizing the static/API layer is possible, but local GGUF inference also
requires a compatible `llama-server` binary and mounted model files. Those large
platform-specific artifacts intentionally are not packaged in this repository.
For a public deployment, separate the UI/API, authenticated inference service,
retrieval provider, rate limiting, secret storage, TLS, and persistent database.

## Scientific limits

The live evidence verifier is an attributable screening system, not a factuality
oracle. The active research results use exact-response aligned teacher-forced
signals and a held-out test split; see `F1_AUDIT.md`. They remain uncalibrated,
same-model classification scores rather than live factuality probabilities. The
experimental Transformers Qwen option produces real 331-feature probe inputs,
but its score never blocks answers. Standard llama.cpp models use the separate
NLI evidence verifier.
