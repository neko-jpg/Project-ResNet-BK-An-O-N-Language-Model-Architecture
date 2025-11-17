# Changelog

All notable changes to the ResNet-BK (Mamba-Killer) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community infrastructure setup
- GitHub issue templates
- Citation information (BibTeX, DOI, arXiv)
- Continuous integration with GitHub Actions
- Release process documentation
- Migration guides

## [0.9.0] - 2025-01-15

### Added
- Hugging Face integration with transformers-compatible model class
- PyTorch Hub integration for easy model loading
- ONNX and TensorRT export capabilities
- Theoretical verification suite with mathematical property tests
- Epsilon-parametrized model family for compression
- Koopman operator compression
- Clark measure computation and preservation

### Changed
- Improved numerical stability with LAP-based kernels
- Enhanced memory optimization with semiseparable structure
- Updated documentation with comprehensive examples

### Fixed
- Numerical stability issues in long-context training
- Memory leaks in gradient checkpointing
- Quantization errors in INT4 mode

## [0.8.0] - 2024-12-20

### Added
- Adaptive Computation Time (ACT) with scattering-based halting
- Learned sparsity for G_ii computation
- Multi-scale processing for efficiency
- Dynamic efficiency benchmarks
- FLOPs counter for accurate measurement

### Changed
- Improved scattering-based routing performance
- Enhanced ACT halting criteria
- Optimized multi-scale downsampling

### Fixed
- ACT early exit bugs
- Sparsity mask computation errors

## [0.7.0] - 2024-12-01

### Added
- Post-training quantization (PTQ) for INT8/INT4
- Quantization-aware training (QAT)
- Mixed-precision quantization strategies
- Quantization robustness benchmarks
- Quantization graph generation

### Changed
- Improved quantization error handling
- Enhanced bit-width sweep functionality

### Fixed
- Quantization precision issues
- INT4 group-wise quantization bugs

## [0.6.0] - 2024-11-15

### Added
- Long-context training infrastructure (up to 1M tokens)
- Streaming evaluation for ultra-long sequences
- Mamba baseline for fair comparison
- Long-context stability graph generation
- Fair FLOPs and memory measurement

### Changed
- Improved gradient stability monitoring
- Enhanced loss spike detection
- Optimized memory usage for long sequences

### Fixed
- Divergence issues at N=128k
- Memory overflow in streaming evaluation

## [0.5.0] - 2024-11-01

### Added
- Semiseparable matrix structure implementation
- Memory optimization strategies (ZeRO, CPU offloading)
- Gradient checkpointing with structure awareness
- Hierarchical semiseparable structure
- Mixed-precision with structure-aware precision

### Changed
- Reduced memory usage by 70% vs dense attention
- Improved O(N) complexity implementation
- Enhanced batch sizing strategies

### Fixed
- Memory leaks in semiseparable factorization
- CPU offloading synchronization issues

## [0.4.0] - 2024-10-15

### Added
- Scattering-based router (parameter-free MoE routing)
- Spectral shift function computation
- Clark measure for adaptive expert allocation
- Resonance detection and handling
- Scattering phase visualization

### Changed
- Replaced MLP gating with physics-based routing
- Improved routing speed (10× faster)
- Enhanced interpretability with phase analysis

### Fixed
- Phase computation boundary issues
- Resonance detection false positives

## [0.3.0] - 2024-10-01

### Added
- Prime-Bump potential initialization
- GUE eigenvalue statistics verification
- Epsilon scheduling and annealing
- Prime sieve for potential generation
- Canonical coefficient computation

### Changed
- Improved initialization convergence (30% faster)
- Enhanced eigenvalue spacing analysis
- Optimized prime bump placement

### Fixed
- Finite overlap condition violations
- GUE statistics verification bugs

## [0.2.0] - 2024-09-15

### Added
- Birman-Schwinger kernel implementation
- Mourre estimate verification
- Limiting Absorption Principle (LAP)
- Schatten norm monitoring
- Spectral clipping for stability
- Precision management (complex128/complex64)

### Changed
- Improved numerical stability guarantees
- Enhanced trace-class property enforcement
- Optimized resolvent kernel computation

### Fixed
- Numerical instability in resolvent computation
- Schatten norm overflow issues
- Precision upgrade threshold bugs

## [0.1.0] - 2024-09-01

### Added
- Initial ResNet-BK architecture
- BK-Core with O(N) theta/phi recursions
- Sparse MoE integration
- Basic training infrastructure
- WikiText-2 benchmark
- Documentation and examples

### Changed
- N/A (initial release)

### Fixed
- N/A (initial release)

---

## Version History Summary

- **0.9.x**: Community integration and deployment
- **0.8.x**: Dynamic compute efficiency (ACT, sparsity, multi-scale)
- **0.7.x**: Quantization robustness
- **0.6.x**: Long-context stability
- **0.5.x**: Memory optimization and semiseparable structure
- **0.4.x**: Scattering-based routing
- **0.3.x**: Prime-Bump initialization
- **0.2.x**: Birman-Schwinger core
- **0.1.x**: Initial architecture

## Migration Guides

See [MIGRATION.md](MIGRATION.md) for detailed migration guides between major versions.

## Links

- [GitHub Repository](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture)
- [Documentation](https://resnet-bk.readthedocs.io)
- [Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
- [Hugging Face Models](https://huggingface.co/resnet-bk)
