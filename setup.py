from setuptools import setup, find_packages

setup(
    name="deephear",
    version="0.1.0",
    author="Yug Dave",
    description="DeepHear: Multimodal contrastive + PHQ-8 classification library",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12",
        "transformers>=4.20",
        "datasets",
        "numpy",
        "pandas",
        "tqdm",
        "torchaudio",
        "librosa",
        "pydub",
    ],
    python_requires=">=3.7",
)