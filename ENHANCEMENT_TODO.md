# Grounded Research Studio — Enhancement Plan

This checklist separates completed functionality from research claims that still
require experiments. Items are implemented in dependency order and marked only
after tests pass.

## Customer API workspace — requested 2026-10-08

The customer entry point now serves installed local models and optional API models in one viewer; the older research dashboard is preserved only for specialist experiment controls.

### Usable public workspace and session repair — 2026-10-09

- [x] Verify Firebase token signatures, audience, issuer, subject, and timestamps against Google's signed public certificates without requiring an Admin credential for ordinary login.
- [x] Distinguish invalid/expired sessions from temporary certificate-service failures; preserve the workspace and expose a retry/sign-in recovery path.
- [x] Keep credential-backed revocation/disabled-user checks optional behind `FIREBASE_CHECK_REVOKED=true`; do not claim revocation checking when it is disabled.
- [x] Add authenticated per-account API connect/catalog/disconnect endpoints to localhost. Validate exact chat model IDs using read-only provider catalog endpoints; retain personal keys only in server memory.
- [x] Scope provider clients, quota cooldowns, and captured job credentials to the user's connection without mutating the owner's settings or storing keys in saved answers.
- [x] Return account-specific model configuration with the workspace; redact credential inputs and raw upstream errors.
- [x] Expose explicit source, claim-verdict, reasoning, and saved-answer analysis controls in the customer interface.
- [x] Publish the actual customer interface in GitHub Pages browser mode with Firebase login, personal API connections, evidence review, and per-account IndexedDB history. Verified the public HTML contains the composer/analysis panel/browser runtime and all six release assets return HTTP 200 on 2026-10-09.
- [x] Verify Firebase project configuration lists `paila009.github.io` after the owner authorized the domain. Interactive public Google sign-in still needs a user acceptance check; browser automation was unavailable in this environment.
- [ ] Verify a real user-supplied API key through streamed generation, evidence review, and Stop. No provider key has been supplied; mocked provider tests do not establish live access or free billing.

The public browser workspace uses each user's own in-memory key directly with the provider and clears it on refresh/sign-out. It cannot expose shared-owner server keys or execute downloaded GGUF weights. Official download links are supplied; local execution needs a runtime/local app. A shared-key hosted service still needs a Python backend. Free-plan confirmations are acknowledgements rather than billing guarantees.

Integrated verification: 104 Python tests and 25 JavaScript frontend/runtime tests passed, including actual Pages asset construction, signed-token rejection, account isolation, current catalog/key checks, key redaction, streaming, timeouts/Stop, harmless greetings in reviewed-only mode, and selecting historical answer reviews. Real installed Qwen2.5 answered “What is cricket?” with the supplied passage in 0.91 seconds and its review linked that exact passage. Live Wikipedia retrieval returned sources. Browser CORS preflight checks succeeded for all three API providers. Provider generation tests use mocked responses because no real personal key was supplied. Windows visual/browser automation failed to start, so no visual or real OAuth acceptance claim is made.

### Restored monitoring and downloaded-model visibility — 2026-10-09

- [x] Keep Grounded guard, source-search controls, review method, and selected runtime visible before the first answer, with honest waiting states instead of fabricated risk percentages.
- [x] Show all installed local models as direct workspace buttons and group downloaded/connected API models in the composer selector; fall back safely from an obsolete saved selection.
- [x] Keep localhost on its Python backend even if an old Pages deployment flag is present, request uncached configuration, and add Refresh models.
- [x] Show actual source passages during generation and preserve supported, conflicting, and unverified claims, their reasons, and exact passages in each saved answer review.
- [x] Keep source and claim previews accessible on narrow screens instead of hiding them; preserve the composer outside the scrolling conversation.
- [x] Keep manually selected historical analysis visible during polling and provide Back to live monitor.
- [x] Lock model/guard changes during initial submission as well as generation, and recover the composer after a reported failure.

Local verification: 104 Python tests and 34 JavaScript frontend/runtime tests pass. All three installed models appear in `/api/config`; health, HTML, and app assets respond successfully with the model toolbar and persistent monitoring controls. Browser automation still fails at startup (`helper_unknown_error`), so a visual acceptance check is not claimed. The owner subsequently requested publishing these canonical frontend changes through the existing GitHub Pages workflow; downloaded-model execution still requires localhost.

### Free-tier live chat checklist — 2026-10-09

- [x] Gate Groq, Gemini, and OpenRouter behind explicit free-only owner confirmations; require OpenRouter model IDs to end in `:free` and provide no automatic paid fallback.
- [x] Remove the stale automatic model default; require a current model ID from the owner's Free Plan console.
- [x] Handle provider quota failures with a shared cooldown, no automatic retries or paid fallbacks; preserve an answer with an unverified review if review quota runs out.
- [x] Add an offline, secret-safe owner readiness check and an ignored local configuration template. No keys were supplied or invented.
- [x] Keep Firebase login mandatory for live chat, as the owner explicitly requested. No anonymous/local-live bypass.
- [x] Restart the customer preview on port 8770; verify health and all four page/static routes return HTTP 200.
- [ ] Connect the owner's Groq Free Plan key/model, then verify real streamed generation, evidence review, and Stop. Mocked tests do not establish provider readiness or billing status.
- [x] Add the owner's Firebase public web configuration locally and restart with mandatory authentication (`preview=false`). Anonymous and malformed-token requests return 401.
- [x] Enable Firebase Google Authentication for project `grounded-583ef`, use the authorized `localhost` origin, and verify that the owner can sign in and reach the private workspace.
- [x] Keep signed-in navigation in the workspace, refresh a stale Firebase ID token once after a 401, and report a refresh problem without replacing the workspace with the login page.
- [x] Add a Models library and composer selector that run installed local models and configured API models in the same customer viewer.
- [x] Reuse cancellable llama.cpp runtimes in the customer pipeline, bound local context for 4K models, and attach a conservative local evidence review without a second API call.
- [x] Fix the older local-model NLI regression: passages rejected as unverified no longer retain a source link or unrelated excerpt. `test_invented_claim_cannot_borrow_words_from_other_claim` now passes with the optional NLI runtime enabled.

Earlier verification: customer backend 33/33 tests and frontend 9/9 tests passed; Python compilation and JavaScript syntax checks passed. The authenticated service responded on `localhost:8770`, published all three installed models in the main selector, and a real Qwen inference returned `LOCAL MODEL READY`. Completed answers exposed explicit verdict-reason and searched-source actions. No API-provider key/model was present. The earlier Pages deployment was a showcase; the current follow-up replaces it with the customer interface in browser mode, while Python/server-local execution remains separate.

### Customer-facing cleanup

- [x] Remove infrastructure/provider setup checklists and environment-file instructions from the customer interface; keep owner setup in the documentation.
- [x] Replace the connections popup with compact answer/source Preferences that actually persist.
- [x] Keep modal titles and close controls visible; scroll only the contents when needed on small screens.
- [x] Show an inline unavailable-chat state instead of opening setup when sending a question. Keep typed text and provide access to the labelled sample.
- [x] Retain authentication/billing safeguards and honest preview status; no mock answer is presented as live generation.
- [x] Add an end-user Models panel and prevent normal signed-in navigation from reopening the login screen.

Cleanup checks: all 54 Python regression tests and five new frontend behavior tests pass. Browser checks confirmed saved preferences after refresh, question preservation without a setup popup, a non-scrolling dialog at 1280×800 and 390×844, and an always-visible close control with internally scrolling content at 390×500.

- [x] Create a responsive customer research home, chat, account screen, and source-review drawer, with an always-visible composer.
- [x] Add server-side adapters for Groq, Gemini's OpenAI-compatible API, and OpenRouter. No cloud model-weight hosting is required.
- [x] Wire Firebase email/password, verification, password reset, and Google login; verify ID tokens and account ownership at the backend. Provider-backed login is pending real configuration below.
- [x] Save per-account conversations, per-answer sources/reviews, and text-document libraries in SQLite. Include Markdown export and explicit deletion controls.
- [x] Add streamed draft updates, manual Stop, topic-comparison and writing modes, and optional reviewed-claims-only display.
- [x] Add API claim review requiring actual available source IDs and matching quotations. Distinguish unverified from contradicted; do not label coverage as accuracy.
- [x] Add consent before sending questions/evidence to APIs; server-only keys; body limits; origin/host checks; per-user and global request caps; job ownership and concurrency limits.
- [x] Keep unconfigured preview local-only and block billable model calls. Label the sample as hand-authored.
- [x] Add account-isolation, persistence, streaming-parser, cancellation, review-validation, upload, and security regression tests.
- [x] Verify saved review/source access after refresh in the browser and confirm composer/review bounds at laptop and phone viewport sizes.
- [ ] Connect the owner's chosen API provider and run a real generation + review + cancellation smoke test. No credential was supplied in this turn.
- [ ] Test real sign-up, email verification, login, reset, logout, and two-account isolation after the token-verification repair. An Admin credential is required only for the optional revocation checks, not ordinary signed-token verification.
- [ ] Before accepting public customers: configure HTTPS hosting, persistent storage/backups, provider budget alerts, deployment-level rate limiting, retention/privacy policy, and abuse/load testing.
- [ ] Independently evaluate the new API reviewer; expand academic retrieval and PDF ingestion only with appropriate source access and safe parsing. Current retrieval is Wikipedia introductory passages + selected text-document excerpts.

See `docs/CUSTOMER_SETUP.md` for activation and deployment boundaries. GitHub Pages runs the browser customer workspace using personal provider keys; shared-owner API keys, SQLite histories, and server-local llama.cpp execution require the separately hosted Python backend.

Verification on 2026-10-08: 54 available regression tests passed (18 customer tests + 36 existing tests), Python compilation and both frontend JavaScript syntax checks passed. Customer `/api/health` returned `ok` on port 8770. Real API/Firebase acceptance tests remain unchecked above. One upstream Starlette/httpx test-client deprecation warning was emitted; it did not affect the tests.

## Reliability follow-up — requested 2026-09-26

Earlier checkmarks describe the initial implementation, not full research validation.
The seven engineering follow-ups below are implemented. Scientific validation
and trustworthy retraining remain separately unchecked; implementation does not
establish accuracy.

- [x] 1. Run a real local NLI model with explicit entailment/contradiction/neutral outputs and honest fallback status.
- [x] 2. Treat absent names/dates as unverified; require evidence of a conflict before reporting contradiction. Query-word overlap no longer inflates hallucination risk.
- [x] 3. Stream from inference, review completed claims during generation, and propagate cancellation to inference. Verified real local cancellation; loading/forward passes may finish before cancellation takes effect.
- [x] 4. Resolve simple factual pronoun follow-ups using conversation history and retain that context during answering. Arbitrary ambiguous references are not solved.
- [x] 5. Freeze one evidence snapshot and generation settings for all models in a comparison. Real Qwen/Phi smoke test returned the same snapshot ID.
- [x] 6. Add labeled live-verifier diagnostics, false-positive/missed-claim reporting, and explain the saved F1 results in docs/F1_AUDIT.md. The 20 authored cases are development checks, not independent validation.
- [x] 7. Connect a compatible activation extractor to the fitted probe and validate model/feature dimensions. Real Qwen generation produced 331 features and a raw probe score; it is explicitly unvalidated and never controls the guard.
- [x] Reject response-label mismatches in the training entry point; preserve legacy artifacts.

### Scientific work still required

- [x] Extract Qwen signals by teacher-forcing all 749 exact annotated responses. The original labels now describe the evaluated response text; 172 prompts required left truncation to the 1,024-token extraction window.
- [x] Retrain entropy, logistic, and MLP classifiers on the aligned 522/112/115 train/validation/test split and evaluate once on the reserved test set. Test AUROC/F1: entropy 0.723/0.650, logistic 0.798/0.717, MLP 0.819/0.761.
- [ ] Calibrate the aligned scores on an independent calibration set before interpreting them as probabilities. The current held-out ranking/classification metrics are not live-chat accuracy.
- [ ] Collect an independent, sufficiently large evaluation set for live NLI and tune thresholds using validation data only.
- [ ] Measure cross-model transfer, feature-distribution alignment, and causal steering separately. Passing dimension checks does not validate probe-score accuracy.

## Answer analysis history — requested 2026-09-26

- [x] Add a View analysis action to every generated assistant answer.
- [x] Restore that answer's sources, claim review, risk, coverage, model and timing in the analysis panel.
- [x] Persist the most recent 30 complete analyses in browser-local storage across refreshes.
- [x] Store complete new-run analyses in the Git-ignored local experiment ledger and add View/JSON actions.
- [x] Clearly disable View for older summary-only ledger entries that cannot reconstruct their claims.

### Reproducible checks

- `GROUNDED_NLI=0 python -m unittest discover -s tests` for isolated regression tests.
- `python scripts/evaluate_live_verifier.py` for real local NLI diagnostics (20/20 development cases passed).
- `python scripts/smoke_dashboard.py` for real Qwen/Phi inference, frozen evidence, and cancellation against localhost.
- `node --check dashboard/app.js` for frontend syntax. Browser check: a private-evidence question returned the real Qwen answer through the new job API.


## Phase 0 — Baseline and safety

- [x] Preserve the working local-model chat, retrieval guard, responsive UI, and existing tests.
- [x] Keep model weights, downloaded datasets, secrets, and generated outputs out of Git.
- [x] Add regression coverage for every new scoring and retrieval behavior.

## Phase 1 — Reliable retrieval

- [x] Add an in-memory retrieval cache with age and cache-hit metadata.
- [x] Retry transient Wikipedia failures with bounded backoff.
- [x] Expand/rewrite weak queries without depending on an LLM.
- [x] Rank passages by query relevance and remove duplicates.
- [x] Expose retrieval health, timing, query attempts, and failure categories to the UI.

## Phase 2 — Hybrid claim verification

- [x] Split answers into auditable factual claims while preserving citations.
- [x] Add lexical, entity, number, date, and negation consistency signals.
- [x] Add local MiniLM NLI verification with an explicitly labeled lexical fallback.
- [x] Represent each claim as supported, partial, contradicted, or unverified.
- [x] Require a single attributable passage for every supported claim.

## Phase 3 — Clear and calibrated scoring

- [x] Separate evidence coverage from weakest-claim risk.
- [x] Show supported/partial/contradicted/unverified counts.
- [x] Explain every guard decision in plain language.
- [x] Add validation-only threshold-selection tooling. Legacy labels are invalid for scientific calibration; validated recalibration remains pending above.
- [x] Never present offline probe AUROC/F1 as live-chat accuracy.

## Phase 4 — Model comparison

- [x] Add side-by-side comparison for installed local models.
- [x] Use the same question, retrieved context, temperature, and token budget for each model.
- [x] Compare latency, evidence coverage, risky claims, and answer agreement.
- [x] Allow cancellation and partial results when one model fails.

## Phase 5 — Experiment history and export

- [x] Persist privacy-safe experiment records locally as JSONL.
- [x] Add a history panel with filters for model and status (timestamps remain visible for date review).
- [x] Export individual runs and filtered experiment tables as JSON/CSV.
- [x] Record model ID, quantization, source URLs, timings, and score version without prompt text.

## Phase 6 — Research-probe integration

- [x] Add a versioned adapter for internal signal records.
- [x] Load and validate the trained logistic-probe and train-fitted processor artifacts.
- [x] Validate feature compatibility before inference.
- [x] Display probe availability separately from evidence-verification results.
- [x] Mark live probe output unavailable unless a compatible activation runtime is connected.

## Phase 7 — Evaluation and calibration

- [x] Add a reproducible evaluation runner for the local labeled split.
- [x] Report precision, recall, F1, AUROC, calibration error, and confidence intervals.
- [x] Compare available entropy/logistic/MLP artifacts and mark lexical/hybrid/SelfCheckGPT unevaluated until aligned labels exist.
- [x] Add adversarial cases for paraphrases, entity swaps, numbers, dates, and negation.
- [x] Produce machine-readable and dashboard-ready result summaries.

## Phase 8 — Research UX and document grounding

- [x] Add document upload for local PDF/text evidence with size/type limits.
- [x] Show exact supporting and contradicting passages per claim.
- [x] Keep detailed filtered-claim evidence inside the dedicated analysis panel.
- [x] Add guided supported, contradicted, and fabricated demo questions.
- [x] Add accessible charts for detector comparisons and numeric model-comparison cards.

## Phase 9 — Production readiness

- [x] Add host/port configuration through environment variables without exposing secrets to the browser.
- [x] Add structured logs, request IDs, bounded retrieval timeouts, and graceful keyboard shutdown.
- [x] Add backend integration tests and responsive CSS regression coverage.
- [x] Add deployment documentation while keeping localhost as the default.
- [x] Document hardware, privacy, model-license, and scientific-validity limitations.
