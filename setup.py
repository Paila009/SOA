from setuptools import setup, find_packages

setup(
    name="hallucination-detection",
    version="0.1.0",
    description="Detecting and mitigating hallucination in RAG via internal uncertainty signatures",
    author="Research Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.42.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "baselines": [
            "selfcheckgpt>=0.1.0",
            "sentence-transformers>=2.3.0",
            "spacy>=3.7.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
)
