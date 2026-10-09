# Download and run local models

The Models panel offers GGUF file downloads. A model file needs an inference
runtime; GitHub Pages cannot start programs on a visitor's laptop. Local use
opens the same Grounded interface at localhost.

## Windows setup

1. [Download the app](https://github.com/Paila009/SOA/archive/refs/heads/main.zip),
   extract it, and follow [CUSTOMER_SETUP.md](CUSTOMER_SETUP.md) to install the
   Python environment and configure Firebase. Login remains required.
2. Download a **Windows x64 (CPU)** archive from the official
   [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases). Extract
   the entire archive, including its DLLs, under `runtime/llama-cpp/` in the app
   directory. Grounded searches it recursively for `llama-server.exe`.
3. Create `models-local/` in the app directory and place downloaded weights
   there with these exact filenames:

| Model | File | Official download |
| --- | --- | --- |
| Qwen2.5 1.5B | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | [GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true) |
| Phi-3 Mini | `Phi-3-mini-4k-instruct-q4.gguf` | [GGUF](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true) |
| Qwen3 4B | `Qwen3-4B-Q4_K_M.gguf` | [GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=true) |

4. Keep `ENABLE_LOCAL_MODELS=true` in your ignored `customer/.env`. Start
   `./run_customer.ps1`, open [localhost:8770](http://localhost:8770/), and sign
   in. Models lists fully downloaded files and installed runtimes alongside
   any connected API models. Firebase must authorize `localhost`.

Start with Qwen2.5 1.5B on a laptop. Larger models need more free RAM and are
slower on CPU. The first request may load weights; Stop cancels generation.
Google login and public source retrieval still require internet, but local
inference needs no provider API key. Local answers and evidence checks can err.

Your existing installation stays available on localhost. Public visitors cannot
use the project owner's laptop or its models. Weights, runtime binaries, personal
histories and provider secrets are excluded from Git.
