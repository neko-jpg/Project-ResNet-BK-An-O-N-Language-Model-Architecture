"""
Setup script for Mamba-Killer ResNet-BK

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#') and not line.startswith('--')
        ]
else:
    requirements = [
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
        'tensorboard>=2.13.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'tokenizers>=0.13.0',
        'accelerate>=0.20.0',
        'einops>=0.6.0',
    ]

setup(
    name="mamba-killer-resnet-bk",
    version="0.9.0",
    author="Teppei Arai",
    author_email="arat252539@gmail.com",
    description="Mamba-Killer: Ultra-Scale ResNet-BK with Birman-Schwinger Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "hub": [
            "huggingface_hub>=0.16.0",
        ],
        "all": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "wandb>=0.15.0",
            "huggingface_hub>=0.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mamba-killer-train=train:main",
            "mamba-killer-eval=scripts.evaluate:main",
            "mamba-killer-prepare-data=scripts.prepare_datasets:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],
    },
)
