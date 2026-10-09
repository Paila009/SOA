# Grounded customer workspace — unified local/API edition

## What this build does

The customer app uses one responsive interface for installed local GGUF models and optional model APIs. Local inference runs through llama.cpp on the computer hosting the workspace; API inference runs at the selected provider. A public website still needs ordinary web/backend hosting, and a visitor cannot use models on their own laptop without a separately installed local companion/runtime.

| Application | Start command | Local address | Purpose |
| --- | --- | --- | --- |
| Customer workspace | `./run_customer.ps1` | `http://localhost:8770` | API-based research, accounts, saved reviews |
| Existing research dashboard | `./run_dashboard.ps1` | `http://127.0.0.1:8766` | Optional specialist internal-signal experiments |

Nothing is publicly deployed, and no GitHub push is included in this change.

### Included

- Research, Compare (topics), and Write modes; conversational context; server-streamed drafts delivered through owner-scoped job polling; Stop control.
- Qwen2.5 1.5B, Phi-3 Mini 3.8B, and Qwen3 4B in the same customer chat when their local GGUF files and llama.cpp runtime are installed.
- Optional explicitly configured Groq Free Plan, Gemini Free Tier, and OpenRouter `:free` models. All API requests originate on the server; the browser never receives a provider key.
- Firebase email/password registration, verification, password reset, and Google sign-in wiring. The server verifies ID tokens (including revocation) and requires a verified email.
- Per-account SQLite conversations, answer-level evidence snapshots/reviews, Markdown exports, text-document library, and explicit deletion controls.
- Claim review with supported, contradicted, unverified, and nonfactual labels. Supported/contradicted verdicts require a quoted passage that actually exists in the returned evidence.
- Source and claim drawers that open for the selected saved answer, not just the latest reply.
- Compact mobile/laptop layouts, native accessible dialogs, keyboard actions, reduced-motion support, and escaped text rendering.
- A Models library that clearly separates configured Groq live-chat models, installed local research models, and providers disabled by the free-only policy.

### Not yet connected or established

The Firebase web configuration, Google sign-in, and local models are connected on `localhost`; a real local Qwen smoke response succeeded. No API-provider key/model has been configured, so provider responses, provider billing status, and deployment are not verified. Do not present the hand-authored sample, local text matching, or offline tests as evidence of detector accuracy.

### Customer interface versus owner setup

Infrastructure setup belongs in this owner guide, not in the customer workspace. The Preferences dialog only controls answer display and source search, with selections saved in the current browser. The normal interface does not show Firebase/provider connection checklists, environment-file instructions, or API-key requests. If live chat is unavailable, an inline message preserves the typed question and offers a labelled sample conversation; it does not open a setup popup. This presentation change does not bypass authentication or enable unconfigured model calls.

## 1. Install once

From the repository directory, using Python 3.12 or newer:

```powershell
python -m venv .venv-customer
.\.venv-customer\Scripts\python.exe -m pip install -r customer/requirements.txt
# Only if customer/.env does not already exist (never overwrite saved keys):
if (-not (Test-Path customer/.env)) { Copy-Item customer/.env.example customer/.env }
.\run_customer.ps1
```

This turn installed the environment on the current computer already. Without credentials, the app runs in local preview mode. Preview access is restricted to loopback, uses an HTTP-only signed cookie, and **cannot make model API calls even if a provider key is present**. Preview ownership resets if the server restarts or the cookie expires; real Firebase histories persist by user ID across restarts.

The app loads `customer/.env`. Existing environment variables take precedence. Do not commit that file, service-account JSON, SQLite databases, or logs; ignore rules are included.

## 2. Local chat and optional free-tier APIs

Local chat is enabled by default with `ENABLE_LOCAL_MODELS=true` and requires no model API key. On this laptop, all three listed local models are installed and appear in the main customer model selector.

Optional API models require an owner-created key stored only in `customer/.env`, never in chat or frontend code. The project supports Groq Free Plan, Gemini Free Tier, and OpenRouter model IDs explicitly ending in `:free`. Provider policies and model availability can change; the confirmation flags are owner acknowledgements, not billing verification.

Check the account's current [free-plan model limits](https://console.groq.com/docs/rate-limits) and choose an available chat model from your console. Model access changes, so there is no silently selected default; the old Llama default has been removed. Comma-separated model IDs are supported. A model appearing in the selector means it is configured, not that access has been tested.

```dotenv
GROQ_API_KEY=your-secret-here
GROQ_MODELS=your-current-free-plan-chat-model-id
# Set true ONLY after you verify the organization is on the Free Plan:
GROQ_FREE_PLAN_CONFIRMED=true
```

**Important:** the confirmation flag is an owner acknowledgement, not a billing-verification API. The app cannot determine a Groq account's plan from its key. The same model/key on a paid organization can incur charges. Keep the organization free; changing a flag here does not change your provider plan. [Groq billing documentation](https://console.groq.com/docs/billing-faqs) explains the difference. Ordinary website hosting and login services have separate limits/costs.

Without the matching confirmation, key, and model ID, an API model is hidden and rejected at both the HTTP route and provider adapter. OpenRouter additionally rejects model IDs that do not end in `:free`. There is no automatic paid fallback, retry, or provider switch.

For Gemini, create a Gemini-specific restricted key in [Google AI Studio](https://aistudio.google.com/apikey); the Firebase browser key is not a Gemini model key. Then configure an exact currently available model ID:

```dotenv
GEMINI_API_KEY=your-private-gemini-key
GEMINI_MODELS=your-current-free-tier-model-id
GEMINI_FREE_TIER_CONFIRMED=true
```

OpenRouter is accepted only for explicit free model IDs:

```dotenv
OPENROUTER_API_KEY=your-private-openrouter-key
OPENROUTER_MODELS=provider/model:free
OPENROUTER_FREE_ONLY_CONFIRMED=true
```

Run the offline, secret-safe readiness check:

```powershell
.\.venv-customer\Scripts\python.exe -m customer.check_setup
```

This lists missing configuration without printing secrets or making API calls. It does **not** validate provider access, billing, Firebase Admin credentials, or real login. Connect Firebase in step 3 and restart the server before testing an actual question. Preview still cannot make API calls; authentication has not been bypassed.

Optional: set `REVIEW_MODEL=groq:model-id` to use another allowed configured Groq model for evidence review. Otherwise the selected generation model also reviews the draft. Using the same model can create correlated errors.

Each factual research answer can consume **two API requests from the shared free quota**: generation and review. Generation is capped by `MAX_OUTPUT_TOKENS` (default 1,400); a structured review can use up to 3,500 output tokens. All customers share the provider organization's limits, so a free tier is not an unlimited public service. Long context can exhaust token limits even when request counts look small.

On HTTP 429, generation stops without retry/fallback and a shared in-process cooldown respects integer `Retry-After` seconds (bounded 1–86,400; otherwise 60 seconds). During cooldown, new requests are rejected before creating a conversation or spending an app request slot. Already-running requests may still finish. A review-only quota failure preserves the draft and explicitly marks its claims unverified. The cooldown resets on restart; provider limits do not. Stop closes the outgoing response where possible but does not restore already consumed quota. No successful live provider call has been tested without your credentials.

Official adapter references:

- [Groq text chat and streaming](https://console.groq.com/docs/text-chat)
- [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- [OpenRouter API reference](https://openrouter.ai/docs/api-reference/overview)

## 3. Connect Firebase Authentication

In your Firebase project:

1. Add a Web app and obtain its public configuration.
2. In Authentication, enable Email/Password and Google (if desired).
3. Add `127.0.0.1`, `localhost`, and eventually your actual website domain to authorized domains as appropriate. Configure your sender/templates and support email.
4. Put the web configuration fields into `customer/.env`:

```dotenv
FIREBASE_API_KEY=public-web-api-key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your-project
FIREBASE_APP_ID=your-web-app-id
```

These are web configuration identifiers, not the model API secret or Firebase Admin private key. You may share this web config to finish integration. **Do not share a service-account private key in chat.**

5. Give the backend Application Default Credentials for the intended Firebase project, or store the Admin service-account JSON outside this repository and set its absolute path in `GOOGLE_APPLICATION_CREDENTIALS`. Protect that file and restrict access. Do not upload it to the website or GitHub.
6. Restart `run_customer.ps1`. The app now requires Firebase login instead of local preview.
7. Create an account, verify its email, and sign in again. Google accounts with a verified email can sign in directly. Use two accounts to verify that conversations/documents cannot cross account boundaries.

The frontend uses pinned Firebase browser modules. The backend validates Firebase ID tokens with `check_revoked=True`; arbitrary client-supplied user IDs are never trusted. User data is stored in SQLite, **not Firestore**. Firestore rules are not a substitute for the backend's ownership checks.

Official references: [Firebase web setup](https://firebase.google.com/docs/web/alt-setup), [server ID-token verification](https://firebase.google.com/docs/auth/admin/verify-id-tokens).

## 4. Evidence, accuracy, and research limits

The pipeline retrieves Wikipedia introductory passages (up to four, or two per topic in Compare mode), combines them with selected excerpts from up to three uploaded UTF-8 `.txt`/`.md` documents, and supplies that snapshot to the API model. Search is skipped for simple greetings and Write mode. Model answering uses recent conversation; search query resolution is deliberately basic, so vague follow-ups may retrieve irrelevant sources.

Documents are capped at 40,000 characters each; uploaded text is stored privately on the app server. Only selected, question-ranked excerpt windows are sent to the model. It is not full-paper ingestion or a complete academic literature review. PDFs should be exported to relevant text first. Scanned PDFs, citations behind paywalls, scholarly ranking, and arbitrary URL fetching are not implemented.

After API generation, a second API request reviews up to 24 sentence-sized claims against those passages. Local generation instead uses a conservative local passage-matching review and makes no second model API call. The backend rejects fabricated source IDs and quotations in API reviews. Both methods improve traceability but do not prove logical entailment, source reliability, or factual truth.

The default "Full answer" preference shows a draft while it is being generated and attaches a review afterward. Every completed answer has a visible evidence strip with separate **Why this verdict** and **View searched sources** actions; the drawer preserves the grounded check, exact searched passages, and review method. The optional "Reviewed answer" preference holds the draft until the review finishes, then filters to supported/nonfactual claims. It is **post-generation filtering**, not internal causal steering or token-level hallucination prevention. Manual Stop is available during the job. Browser API calls have bounded response times and a generation that exceeds the six-minute safety window is stopped instead of leaving the composer permanently busy.

Coverage is the fraction of reviewed factual claims marked supported. It is **not** calibrated accuracy, confidence, or a hallucination probability. Missing evidence is unverified, not false. Retrieval/reviewer failure is surfaced, and no substitute evidence is invented. Source snapshots and original filtered drafts remain available in saved reviews.

The separate research probes (aligned Qwen entropy/logistic/MLP results) are unchanged and not used by this customer API pipeline. No new F1, live accuracy, causal validity, cross-model transfer, or steering result is claimed.

## 5. Customer deployment checklist

This is a runnable, connection-ready **private-beta foundation**, not an audited public production service. Before accepting real customer data:

- Configure `APP_ENV=production`, an HTTPS `PUBLIC_ORIGIN`, all Firebase web fields, and server-side credentials. Production fails closed without required configuration. Bind the backend behind a trusted HTTPS reverse proxy; do not expose the unconfigured preview.
- Use one application worker on a persistent disk with SQLite backups and restore testing. Jobs/concurrency limits are in-process; multiple workers/replicas require a shared job queue and centralized rate limits. A server restart interrupts active jobs, though already saved messages remain.
- Set provider-side spending limits/alerts. App limits count research jobs, not currency. Default caps are 30 requests per user and 200 total per rolling 24h, plus six per user per minute and four simultaneous jobs globally. Token counts/charges are not metered by the app.
- Add edge-level abuse protection for signup and authentication, central monitoring without prompt/secret leakage, dependency/security review, and realistic concurrent-user load tests.
- Set a privacy notice, retention/deletion policy, provider data-processing choices, appropriate terms, and an incident/support process. Documents and excerpts are not encrypted by this app at rest; use encrypted storage and access controls. Do not use it for regulated/confidential material without the required protections.
- Decide whether to migrate account data to managed database storage. Firebase currently provides authentication only.
- Test actual provider behavior (streaming, model IDs, quota failures, stop, review JSON quality) and actual Firebase sign-up/login/logout/reset/revocation. Add independent human-labelled evaluation before making reliability claims.

Deleting a library document does not remove excerpts already embedded in past answer reviews; deleting those conversations or all workspace data does. Workspace deletion does not delete the Firebase identity. Non-content usage counters remain for up to 24h to prevent abuse-limit bypass. Preview data and uploads remain on your machine until removed; API mode sends selected content to the chosen provider after user acknowledgment.

## 6. Verification

```powershell
.\.venv-customer\Scripts\python.exe -m pytest tests -q
.\.venv-customer\Scripts\python.exe -m compileall -q customer
node --check customer/web/app.js
node --check dashboard/app.js
node --test tests/customer_ui.test.cjs
```

Customer regression tests cover unauthenticated/unverified denial, user isolation for chats/documents/jobs/export, preview billing denial, request limits, content/host/origin limits, missing/fabricated quotations, streamed parsing, persistent history, and cancellation. Model/Firebase tests inject deterministic test adapters and use disposable databases; they never call a paid API or claim live-auth validation.

Browser checks include a saved sample answer's review and source tabs after refresh, mobile navigation, and viewport bounds for the composer and review drawer. Live endpoint `/api/health` identifies this service as `grounded-customer` to distinguish it from the original dashboard.

### Latest verification — 2026-10-09

Unified-model and navigation checks: 33 customer backend tests and nine frontend behavior tests pass. The complete Python regression suite passes 69/69 tests with the optional NLI runtime enabled. The unrelated-evidence regression is fixed: an unverified claim no longer keeps a source link or passage merely because of incidental overlap. The checks cover free-only API gates, streaming, Stop, shared quota cooldown, unverified review on quota exhaustion, stale-token retry, signed-in navigation, same-view local selection, local execution without API consent, and the visible verdict/source actions attached to completed answers. Compilation and JavaScript syntax checks pass; the authenticated localhost service publishes all three local models, and a real Qwen local response succeeded. No API-provider key/model is connected, so API-generation checks remain mocked/offline.

## GitHub Pages

The `pages/` directory and `.github/workflows/pages.yml` publish a static interactive showcase at `https://paila009.github.io/SOA/`. It intentionally does not imitate a working hosted chat: GitHub Pages cannot run FastAPI, Firebase Admin token verification, SQLite, retrieval, API proxying, or local llama.cpp models. Deploy the Python service on an HTTPS backend host before presenting login and chat as a public customer service.

Live browser automation was unavailable because the Windows UI helper failed to start, so the latest verification used endpoint checks, a real local-Qwen end-to-end job, JavaScript behavior tests, and the complete Python suite. Earlier laptop/phone layout checks remain recorded above.
