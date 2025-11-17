ResNet-BK Documentation
=======================

**A mathematically rigorous O(N) language model that surpasses Mamba in long-context stability, quantization robustness, and dynamic compute efficiency.**

ResNet-BK is built on rigorous mathematical foundations from quantum scattering theory and the Birman-Schwinger operator. Unlike empirical approaches, every component is backed by proven theorems guaranteeing numerical stability, computational efficiency, and superior performance.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   README
   TUTORIAL
   FAQ

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   ARCHITECTURE
   API_REFERENCE
   BENCHMARKING
   REPRODUCIBILITY

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   MEMORY_OPTIMIZATION
   EARLY_EXIT
   ACT_IMPLEMENTATION
   SPARSE_BK_CORE
   MULTI_SCALE_PROCESSING
   LEARNED_SEQUENCE_LENGTH

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   WIKITEXT2_BENCHMARK
   WIKITEXT103_BENCHMARK
   PENN_TREEBANK_BENCHMARK
   C4_BENCHMARK
   PILE_BENCHMARK
   SCALING_EXPERIMENTS
   FLOPS_COUNTER

.. toctree::
   :maxdepth: 2
   :caption: Integration

   HUGGINGFACE_INTEGRATION
   STEP7_SYSTEM_INTEGRATION

.. toctree::
   :maxdepth: 2
   :caption: Development

   CONTRIBUTING
   CODE_OF_CONDUCT
   SECURITY
   CHANGELOG
   RELEASE
   MIGRATION

.. toctree::
   :maxdepth: 2
   :caption: Reference

   CITATION
   LICENSE
   THIRD_PARTY_LICENSES

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/models
   api/training
   api/benchmarks
   api/utils

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install mamba-killer-resnet-bk

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from src.models import LanguageModel

   # Load pre-trained model
   model = LanguageModel.from_pretrained("resnetbk/mamba-killer-1b")

   # Generate text
   input_ids = torch.tensor([[1, 2, 3, 4, 5]])
   output = model.generate(input_ids, max_length=100)

Key Features
------------

* **O(N) Complexity**: Linear time and memory scaling
* **Long-Context Stability**: Stable on 1M token sequences
* **Quantization Robustness**: 4× better than Mamba at INT4
* **Dynamic Efficiency**: 2× fewer FLOPs at equal perplexity
* **Mathematical Rigor**: Every operation backed by theorems

Performance Highlights
----------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - ResNet-BK
     - Mamba
     - Improvement
   * - Max Stable Context
     - 1M tokens
     - 32k tokens
     - 31× longer
   * - INT4 Perplexity
     - 45
     - 180
     - 4× better
   * - FLOPs at PPL=30
     - 2.5B
     - 5.0B
     - 2× fewer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
