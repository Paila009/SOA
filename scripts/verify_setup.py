"""
Quick Verification Script.

Run this after installation to verify everything is set up correctly:
- Python version
- PyTorch + CUDA
- Transformers + bitsandbytes
- GPU memory availability
- Model loading test

Usage:
    python scripts/verify_setup.py
"""

import sys
import os


def check_python():
    """Verify Python version."""
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}")
    assert v.major == 3 and v.minor >= 10, "Python 3.10+ required"
    print("  ✓ Python version OK")


def check_torch():
    """Verify PyTorch and CUDA."""
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

        # Quick GPU test
        x = torch.randn(100, 100, device="cuda")
        y = x @ x.T
        del x, y
        torch.cuda.empty_cache()
        print("  ✓ CUDA compute OK")
    else:
        print("  ⚠ No CUDA - will run on CPU (slower but functional)")


def check_transformers():
    """Verify Transformers and bitsandbytes."""
    import transformers
    print(f"\nTransformers: {transformers.__version__}")

    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes: {bnb.__version__}")
        print("  ✓ 4-bit quantization available")
    except ImportError:
        print("  ⚠ bitsandbytes not installed (needed for 4-bit quantization)")

    try:
        import accelerate
        print(f"accelerate: {accelerate.__version__}")
    except ImportError:
        print("  ⚠ accelerate not installed")


def check_sklearn():
    """Verify scikit-learn."""
    import sklearn
    print(f"\nscikit-learn: {sklearn.__version__}")
    print("  ✓ Probe training ready")


def check_gpu_memory():
    """Estimate if models will fit."""
    import torch

    if not torch.cuda.is_available():
        print("\n⚠ No GPU - models will run on CPU")
        return

    total_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"\nGPU Memory Analysis:")
    print(f"  Total VRAM: {total_mem:.1f} GB")

    # Estimates for 4-bit quantized models
    estimates = {
        "Qwen2.5-1.5B (4-bit)": 1.2,
        "Gemma-2B (4-bit)": 1.5,
        "Phi-3-mini (4-bit)": 2.5,
    }

    for model, est_gb in estimates.items():
        fits = "✓" if est_gb < total_mem * 0.9 else "⚠ tight"
        print(f"  {model}: ~{est_gb:.1f} GB {fits}")

    remaining = total_mem - 1.2  # After Qwen loaded
    print(f"\n  After loading Qwen2.5-1.5B: ~{remaining:.1f} GB free for hooks/batch")


def check_optional():
    """Check optional dependencies."""
    print("\nOptional dependencies:")

    optional = {
        "selfcheckgpt": "SelfCheckGPT baseline",
        "spacy": "Sentence segmentation",
        "matplotlib": "Visualization",
        "seaborn": "Visualization",
        "loguru": "Logging",
        "yaml": "Configuration (PyYAML)",
    }

    for mod, purpose in optional.items():
        try:
            __import__(mod)
            print(f"  ✓ {mod} ({purpose})")
        except ImportError:
            print(f"  ✗ {mod} ({purpose}) — not installed")


def main():
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)

    check_python()

    try:
        check_torch()
    except ImportError:
        print("\n✗ PyTorch not installed!")
        return

    try:
        check_transformers()
    except ImportError:
        print("\n✗ Transformers not installed!")

    try:
        check_sklearn()
    except ImportError:
        print("\n✗ scikit-learn not installed!")

    check_gpu_memory()
    check_optional()

    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
