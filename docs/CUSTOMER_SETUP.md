# Grounded customer workspace — unified local/API edition

## What this build does

The customer app uses one responsive interface for installed local GGUF models and model APIs. On localhost, local inference runs through llama.cpp on the computer hosting the workspace and API calls run through Python. At [paila009.github.io/SOA](https://paila009.github.io/SOA/), the same interface runs in browser mode: Firebase login, personal API connections, streamed answers, source/claim review, documents, and saved history. Browser mode sends requests directly to the user's selected provider with their own temporary key; it does not need to host model weights.

Public browser history and documents are stored per Firebase account in that browser's IndexedDB. They do not synchronize across devices. Personal keys stay in memory and are cleared on refresh or sign-out. Localhost instead stores histories in account-scoped SQLite and personal provider connections in server memory until disconnect or restart. Shared owner keys require a separately hosted backend. Downloaded local GGUF files need a local runtime/app; they cannot execute on GitHub Pages.

| Application | Start command | Local address | Purpose |
| --- | --- | --- | --- |
| Customer workspace | `./run_customer.ps1` | `http://localhost:8770` | Local/API research, accounts, saved reviews |
| Existing research dashboard | `./run_dashboard.ps1` | `http://127.0.0.1:8766` | Optional specialist internal-signal experiments |

GitHub Pages serves the browser edition. The Python backend and installed local models remain on the workspace host unless deployed separately.

### Included

- Research, Compare (topics), and Write modes; conversational context; server-streamed drafts delivered through owner-scoped job polling; Stop control.
- Qwen2.5 1.5B, Phi-3 Mini 3.8B, and Qwen3 4B in the same customer chat when their local GGUF files and llama.cpp runtime are installed.
- Personal Groq, Gemini, and OpenRouter API connections through the Models panel, with a provider model list and exact-model selection. On localhost, the backend keeps personal keys in account-scoped memory. Pages mode uses personal browser-memory keys directly. Shared owner keys remain server-only.
- Firebase email/password registration, verification, password reset, and Google sign-in wiring. Localhost validates ID-token signatures/claims against Google's public certificates and requires verified email. Credential-backed revocation checks are optional.
- Per-account SQLite conversations, answer-level evidence snapshots/reviews, Markdown exports, text-document library, and explicit deletion controls.
- Claim review with supported, contradicted, unverified, and nonfactual labels. Supported/contradicted verdicts require a quoted passage that actually exists in the returned evidence.
- Visible verdict/source actions and an evidence panel that open for the selected saved answer, not just the latest reply.
- Compact mobile/laptop layouts, native accessible dialogs, keyboard actions, reduced-motion support, and escaped text rendering.
- A Models library with personal API connection controls, connected models, local installation status, and official model download links.

### Not yet connected or established

The Firebase web configuration, Google sign-in, and local models were connected on `localhost`; a real local Qwen smoke response succeeded. This update repairs backend token verification and adds real personal-provider connections. No actual provider key was supplied, so successful live API generation, provider quotas, and billing status remain unverified. Public Google sign-in also requires `paila009.github.io` to be an authorized Firebase domain. Do not present the hand-authored sample, local text matching, or offline tests as evidence of detector accuracy.

### Customer interface versus owner setup

Infrastructure setup belongs in this owner guide. Preferences controls answer display and source search. Models offers the user-facing personal API connection form and official local-model downloads; environment-file and Firebase Admin instructions stay out of normal customer flows. If live chat is unavailable, an inline message preserves the typed question and offers a labelled sample conversation. Missing connections and authentication failures are shown explicitly rather than leaving the composer busy indefinitely.

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

For a personal API connection, sign in, open **Models**, choose Groq, Google Gemini, or OpenRouter, enter your own key in that connection form, load its current chat-model list, choose an exact model, and confirm your free-plan/model selection. Use Disconnect to remove the connection. Never put a provider key into a chat message. Localhost validates the key through read-only provider endpoints before adding the model; the key is kept only for your account in the running server's memory. Pages mode validates and uses the key directly in browser memory, clearing it on refresh/sign-out.

On the Python backend, the same personal key is used for generation and API evidence review. Pages mode makes a generation request, then applies a conservative passage-matching review in the browser without a second model call. A configured model may still fail because of provider quota, account permissions, a discontinued ID, or browser CORS restrictions. No provider key is bundled with the site. Free-plan confirmation is an acknowledgement, not a billing-verification API: Groq/Gemini keys on paid accounts may incur charges. OpenRouter choices are restricted to text models explicitly ending in `:free` with zero prompt/completion/request pricing in the returned catalog. Provider terms and free limits can change.

For shared owner API access through the Python backend, put an owner-created key only in the ignored `customer/.env`. The following configuration is not used by the public Pages browser edition:

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

This lists server configuration without printing secrets or making API calls. It does **not** validate provider access, billing, Firebase Admin credentials, or real login. Its Admin credential-path entry is optional when `FIREBASE_CHECK_REVOKED=false`. Personal browser/server-memory connections do not appear as saved `.env` keys in this check. Connect Firebase in step 3 and restart the server before testing an actual question. Preview still cannot make API calls.

Optional: set `REVIEW_MODEL=groq:model-id` to use another allowed configured Groq model for evidence review. Otherwise the selected generation model also reviews the draft. Using the same model can create correlated errors.

Each factual API research answer on the Python backend can consume **two API requests**: generation and review. Generation is capped by `MAX_OUTPUT_TOKENS` (default 1,400); a structured review can use up to 3,500 output tokens. Pages mode uses one model-generation request and a browser passage matcher for review. Personal connections consume that user's provider quota; shared-owner connections consume a shared provider organization quota. A free tier is not an unlimited public service. Long context can exhaust token limits even when request counts look small.

On HTTP 429, Python generation stops without retry/fallback and a connection-scoped cooldown respects integer `Retry-After` seconds (bounded 1–86,400; otherwise 60 seconds). Shared owner connections retain a shared cooldown. During cooldown, new requests are rejected before creating a conversation or spending an app request slot. Already-running requests may still finish. A review-only quota failure preserves the draft and explicitly marks its claims unverified. The cooldown resets on restart; provider limits do not. Stop closes the outgoing response where possible but does not restore already consumed quota. No successful live provider call has been tested without your credentials.

Official adapter references:

- [Groq text chat and streaming](https://console.groq.com/docs/text-chat)
- [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- [OpenRouter API reference](https://openrouter.ai/docs/api-reference/overview)

## 3. Connect Firebase Authentication

In your Firebase project:

1. Add a Web app and obtain its public configuration.
2. In Authentication, enable Email/Password and Google (if desired).
3. Add `127.0.0.1`, `localhost`, and `paila009.github.io` (for this Pages workspace) to Authentication's authorized domains. Add any future production domain separately. Configure your sender/templates and support email.
4. Put the web configuration fields into `customer/.env`:

```dotenv
FIREBASE_API_KEY=public-web-api-key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your-project
FIREBASE_APP_ID=your-web-app-id
```

These are web configuration identifiers, not the model API secret or Firebase Admin private key. You may share this web config to finish integration. **Do not share a service-account private key in chat.**

5. Leave `FIREBASE_CHECK_REVOKED=false` for ordinary signed-token verification. The backend uses Google's public certificates and strictly checks RS256 signatures, signing key, project audience, issuer, subject, expiry, issued-at, and auth-time. No Admin file is needed for these checks. For immediate disabled-user/revoked-token checks, set `FIREBASE_CHECK_REVOKED=true` and provide Application Default Credentials or an Admin service-account JSON outside the repository via `GOOGLE_APPLICATION_CREDENTIALS`. Never upload that private file to the website or GitHub.
6. Restart `run_customer.ps1`. The app now requires Firebase login instead of local preview.
7. Create an account, verify its email, and sign in again. Google accounts with a verified email can sign in directly. Use two accounts to verify that conversations/documents cannot cross account boundaries.

The frontend uses pinned Firebase browser modules. Arbitrary client-supplied user IDs are never trusted by the backend. Normal verification accepts only valid signed Firebase ID tokens with a verified email. Google-certificate network failures return a retryable service error; invalid/expired tokens return an authentication error. An Admin credential problem cannot cause ordinary login failure when optional revocation checking is disabled. Default verification does not immediately detect disabled accounts or revoked, still-unexpired tokens; enable the optional credential-backed checks when that behavior is needed.

Localhost user data is stored in SQLite, **not Firestore**. Pages uses IndexedDB scoped to the signed-in account and current browser. Neither edition uses Firestore for histories; Firestore rules do not replace backend ownership checks. Pages uses Firebase's authentication session for its local interface and does not pretend to verify tokens on an absent Python server.

Official references: [Firebase web setup](https://firebase.google.com/docs/web/alt-setup), [server ID-token verification](https://firebase.google.com/docs/auth/admin/verify-id-tokens).

## 4. Evidence, accuracy, and research limits

The pipeline retrieves Wikipedia introductory passages (up to four, or two per topic in Compare mode), combines them with selected excerpts from up to three uploaded UTF-8 `.txt`/`.md` documents, and supplies that snapshot to the API model. Search is skipped for simple greetings and Write mode. Model answering uses recent conversation; search query resolution is deliberately basic, so vague follow-ups may retrieve irrelevant sources.

Documents are capped at 40,000 characters each; uploaded text is stored privately on the app server. Only selected, question-ranked excerpt windows are sent to the model. It is not full-paper ingestion or a complete academic literature review. PDFs should be exported to relevant text first. Scanned PDFs, citations behind paywalls, scholarly ranking, and arbitrary URL fetching are not implemented.

After API generation on the Python backend, a second API request reviews up to 24 sentence-sized claims against those passages. Local GGUF generation uses a conservative local passage matcher instead. Pages mode also reviews in the browser with conservative exact/near-verbatim passage matching and makes no second review API call. It can miss paraphrases and does not establish semantic contradiction. The backend rejects fabricated source IDs and quotations in API reviews. All methods improve traceability but do not prove logical entailment, source reliability, or factual truth. The selected answer's Details tab identifies the method actually used.

The default "Full answer" preference shows a draft while it is being generated and attaches a review afterward. Every completed answer has a visible evidence strip with separate **Why this verdict** and **View searched sources** actions; the evidence panel preserves that answer's grounded check, exact searched passages, and review method. Selecting an older saved answer restores its own snapshot. The optional "Reviewed answer" preference holds the draft until the review finishes, then filters to supported/nonfactual claims. It is **post-generation filtering**, not internal causal steering or token-level hallucination prevention. Manual Stop is available during the job. Requests have bounded response times; the composer recovers after completion, cancellation, or a reported failure.

Coverage is the fraction of reviewed factual claims marked supported. It is **not** calibrated accuracy, confidence, or a hallucination probability. Missing evidence is unverified, not false. Retrieval/reviewer failure is surfaced, and no substitute evidence is invented. Source snapshots and original filtered drafts remain available in saved reviews.

The separate research probes (aligned Qwen entropy/logistic/MLP results) are unchanged and not used by this customer API pipeline. No new F1, live accuracy, causal validity, cross-model transfer, or steering result is claimed.

## 5. Customer deployment checklist

This is a runnable, connection-ready **private-beta foundation**, not an audited public production service. Before accepting real customer data:

- Configure `APP_ENV=production`, an HTTPS `PUBLIC_ORIGIN`, and all Firebase web fields. Shared API keys stay server-side; Firebase Admin credentials are required only if optional revocation checks are enabled. Production fails closed without required configuration. Bind the backend behind a trusted HTTPS reverse proxy; do not expose the unconfigured preview.
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

### Verification — 2026-10-09

Earlier verification covered 69 Python tests and nine frontend behavior tests, a real local-Qwen end-to-end job, compilation, and JavaScript syntax. The new per-account provider changes pass 44 focused provider/customer tests, including catalog verification, authentication gates, key redaction, unchanged owner settings, user isolation, captured job credentials, and streaming through the real adapter with mocked HTTP responses. Full integrated tests and the new Pages deployment are verified separately before publication. No actual provider key has been supplied, so no live API-generation or free-billing result is claimed.

## GitHub Pages

The workflow `.github/workflows/pages.yml` runs `scripts/build_pages.py` and deploys `dist/pages` to [paila009.github.io/SOA](https://paila009.github.io/SOA/). The builder copies the canonical `customer/web` interface with its browser runtime; it no longer deploys a separate marketing showcase. Its allowlist includes frontend assets and the four public Firebase web identifiers from `pages/firebase-config.json`. Provider keys, service-account files, SQLite data, research outputs, and model weights are excluded.

To build locally:

```powershell
python scripts/build_pages.py
```

The public interface requires Firebase login before live chat. In Firebase Console, authorize `paila009.github.io` for Google sign-in. A user then opens Models, connects their own free-plan API key, selects a returned chat model, and sends a question. The chosen provider receives the question and selected evidence directly from the browser. Keys are held only for the current page lifetime, while documents/conversations/reviews persist in browser storage scoped to the Firebase user. Clearing site data removes that browser's history; it is not a cross-device cloud workspace.

The Models panel also offers official local-model file downloads and local setup guidance. Follow [LOCAL_MODELS.md](LOCAL_MODELS.md) for the runtime folder, exact weight filenames, and startup steps. Downloading weights alone does not activate browser chat: run the downloaded model through the runtime and this Python app. Pages cannot start llama.cpp, reach a visitor's installed models automatically, or use the owner laptop's models as a public server.

For customers to use shared owner API keys, synchronized server histories, and server-side verification/review, deploy the Python service on an HTTPS backend host. The public browser edition is a private-beta implementation with provider/browser limits and unvalidated evidence screening; no production-readiness or calibrated reliability claim is made.

Release verification on 2026-10-09: the public URL serves the actual customer workspace, with composer, analysis panel, deployment configuration, and all frontend assets returning HTTP 200. Firebase project configuration confirms `paila009.github.io` is authorized. The full suite passed 104 Python and 25 JavaScript tests; real local Qwen generation and live Wikipedia retrieval succeeded. Windows browser automation could not initialize (`helper_unknown_error`), so interactive public OAuth and visual layout were not validated in this run. Live provider generation still needs a real user-supplied key; mock streaming tests and successful CORS preflights are not proof of account/model access.

### Local monitoring restoration — 2026-10-09

On localhost, the workspace now exposes downloaded-model buttons above the conversation and an always-visible Answer insights monitor. Grounded guard chooses full-draft-plus-review or reviewed-only filtering; Sources controls retrieval. Pending metrics stay unmeasured until review finishes. The monitor shows reported retrieval passages during generation, then each answer's saved guard action, supported/conflicting/unverified claims, reasons, and source snapshot. Older selected analysis stays selected while another answer is running; Back to live monitor returns to that job's progress.

Source and claim previews are no longer hidden by laptop/phone breakpoints. The composer stays outside the scrolling content. Configuration requests bypass cache, and localhost uses the Python runtime even if an obsolete Pages flag was present. A downloaded GGUF file is still executable only through the installed local runtime, not GitHub Pages.

The full Python suite passes 104 tests and the JavaScript frontend/runtime suite passes 34 tests. These cover model fallback and direct toolbar clicks, initial-submission locks, guard persistence, waiting/live source states, distinct conflicting versus missing evidence, and historical selection during polling. No measured accuracy, calibrated hallucination probability, fake-source detector, semantic contradiction guarantee, or causal steering is added. The owner subsequently requested publishing this canonical interface through the existing GitHub Pages workflow; provider generation remains untested until free-plan access is confirmed. Windows browser automation again could not start, so visual and live OAuth acceptance remain unverified.

### Pages release and Groq connection verification — 2026-10-09

Commit `7f14ae2` deployed successfully through Pages workflow run `37889321655`; the public workspace and app module return HTTP 200 with the restored toolbar, guard controls, source monitor, and historical/live analysis selection. The provided Groq secret was excluded from the commit and static artifact.

Following the owner's free-account confirmation, localhost now offers catalog-verified `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, and `qwen/qwen3.8-27b` alongside the three installed local models. The key lives only in ignored `customer/.env`; restart loads it server-side. Health remains non-preview with Firebase authentication required, and anonymous workspace requests still return 401. A real GPT-OSS 20B streamed answer and API evidence review succeeded against one supplied cricket passage in 1.77 seconds. That single connection diagnostic is not a measured reliability or latency result; the other two models and live provider cancellation were not generation-tested.

The public Pages site still requires each visitor to connect their own key in Models. It does not receive the owner's localhost key or local weights. Free-only confirmation is the owner's acknowledgement; catalog/model success cannot establish the provider billing plan. Visual/OAuth acceptance and real public browser generation remain unverified.

### Finding and selecting models on Pages — 2026-10-09

When no API connection exists, the composer now shows **Choose model**, not a disabled empty selector. The workspace toolbar also offers **Connect Groq**, **Connect Google Gemini**, and **Connect OpenRouter** shortcuts that focus the corresponding connection form. Enter your own key there, select **Find available models**, choose a returned model, acknowledge your free-plan configuration, and connect. Ready API models then appear in both the toolbar and the composer selector. Provider changes clear the previous credential and invalidate outstanding catalog requests.

**Local downloads** opens the official weight files and local-app setup instructions. Pages cannot discover or execute weights installed on a visitor's computer; localhost still offers installed models in this same interface. Refreshing the public page clears its temporary keys, so reconnect to resume live API chat; stored chats and answer reviews remain available. This focused picker change retains the existing grounded guard, searched-source monitor, and saved-answer analysis. It adds no detector-accuracy claim and does not publish any owner API key.
