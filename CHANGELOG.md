# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation (CONTRIBUTING.md, QUICK_START.md)
- Docker support with docker-compose.yml
- Phase 2 stability improvements
- Complete test suite for Phase 2 components

### Changed
- Updated README.md with detailed installation and usage instructions
- Improved requirements.txt with all necessary dependencies
- Enhanced .gitignore for better project hygiene

### Fixed
- Phase 2 Lyapunov stability violations (100% resolution)
- CUDA memory access errors in memory resonance
- Overdamping issues (base_decay: 0.01 → 0.001)

## [0.9.0] - 2025-11-20

### Added
- **Phase 2: Breath of Life** - Dynamic memory mechanisms
  - Non-Hermitian Potential for adaptive forgetting
  - Dissipative Hebbian learning with fast weights
  - SNR-based memory filtering
  - Memory resonance via Zeta regularization
- Phase 2 integration tests
- Phase 2 benchmarking suite
- Comprehensive docstring verification
- CI/CD pipeline for Phase 2

### Changed
- Improved Phase 2 stability (92.5% warning reduction)
- Enhanced Lyapunov monitoring for energy tracking
- Optimized memory resonance with torch.bmm

### Performance
- Phase 2 test execution: 19.36 seconds
- Lyapunov violations: 630 → 0 (100% fix)
- Total warnings: 107 → 8 (92.5% reduction)

## [0.8.0] - 2025-11-19

### Added
- **BK-Core Triton Kernel** - Custom CUDA kernel implementation
  - 185× speedup over PyTorch baseline
  - Complex number support
  - Numerical correctness verification (MSE < 10⁻¹⁰)
  - Perfect numerical stability (0% NaN rate)
- Comprehensive Triton kernel test suite
- Benchmark scripts for performance validation

### Performance
- PyTorch (vmap): 554.18 ms
- Triton Kernel: 2.99 ms
- Speedup: 185.10× (target: 3.0×)

## [0.7.0] - 2025-11-18

### Added
- **Phase 1: Efficiency Engine** - Memory-efficient architecture
  - HTT Embedding with 99.6% compression
  - AR-SSM layers for O(N) complexity
  - BK-Core with Semiseparable structure
  - Gradient checkpointing support
- Phase 1 integration tests
- Memory validation scripts
- Throughput benchmarking

### Performance
- Memory reduction: 4.8% (Baseline → Phase 1)
- Throughput improvement: 3.3%
- Perplexity degradation: -0.46% (improvement)

## [0.6.0] - 2025-11-17

### Added
- **BK-Core Implementation** - Birman-Schwinger operator
  - Semiseparable matrix structure (H = T + UV^T)
  - O(N log N) memory complexity
  - 610× parameter reduction vs. standard attention
- Mathematical proofs and theoretical foundations
- Trace-class verification
- Schatten norm bounds

### Changed
- Refactored core architecture for modularity
- Improved documentation structure

## [0.5.0] - 2025-11-16

### Added
- **HTT Embedding** - Holographic Tensor Train
  - 99.6% parameter compression
  - Tensor Train decomposition with rank 4
  - Efficient embedding layer
- HTT compression tests and demos

## [0.4.0] - 2025-11-15

### Added
- **AR-SSM Layers** - Adaptive Rank State Space Models
  - O(N) sequence processing
  - Adaptive rank mechanism
  - Efficient state management
- AR-SSM unit tests

## [0.3.0] - 2025-11-14

### Added
- Initial project structure
- Basic model architecture
- Core mathematical foundations
- Preliminary experiments

### Performance
- Initial stability tests on RTX 3080
- Comparison with Mamba baseline

## [0.2.0] - 2025-11-13

### Added
- Mathematical framework documentation
- Birman-Schwinger operator theory
- Spectral analysis foundations

## [0.1.0] - 2025-11-12

### Added
- Project initialization
- Basic README
- License (MIT)
- Initial repository structure

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes
- **Performance**: Performance improvements

---

[Unreleased]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/releases/tag/v0.1.0
