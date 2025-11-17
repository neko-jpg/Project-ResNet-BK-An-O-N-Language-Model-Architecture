# Release Process

This document describes the release process for ResNet-BK.

## Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New functionality in a backwards-compatible manner
- **PATCH** version (0.0.X): Backwards-compatible bug fixes

### Version Number Guidelines

- **1.0.0**: First stable release with complete API
- **0.X.0**: Pre-release versions during development
- **X.Y.Z-alpha**: Alpha releases for early testing
- **X.Y.Z-beta**: Beta releases for wider testing
- **X.Y.Z-rc.N**: Release candidates before final release

## Release Checklist

### Pre-Release (1-2 weeks before)

- [ ] Create release branch: `git checkout -b release/vX.Y.Z`
- [ ] Update version numbers in:
  - [ ] `setup.py`
  - [ ] `src/__init__.py`
  - [ ] `docs/conf.py`
  - [ ] `README.md`
- [ ] Update CHANGELOG.md with all changes since last release
- [ ] Run full test suite: `pytest tests/ -v --cov`
- [ ] Run benchmark suite: `python scripts/mamba_vs_bk_benchmark.py --all`
- [ ] Update documentation: `cd docs && make html`
- [ ] Review and update migration guide if needed
- [ ] Create release notes draft
- [ ] Notify community of upcoming release

### Testing Phase (1 week)

- [ ] Deploy to test PyPI: `python setup.py sdist bdist_wheel && twine upload --repository testpypi dist/*`
- [ ] Test installation from test PyPI: `pip install --index-url https://test.pypi.org/simple/ resnet-bk`
- [ ] Run integration tests on fresh install
- [ ] Test on multiple platforms:
  - [ ] Linux (Ubuntu 20.04, 22.04)
  - [ ] macOS (Intel, Apple Silicon)
  - [ ] Windows 10/11
- [ ] Test with multiple Python versions:
  - [ ] Python 3.8
  - [ ] Python 3.9
  - [ ] Python 3.10
  - [ ] Python 3.11
- [ ] Test with multiple PyTorch versions:
  - [ ] PyTorch 2.0
  - [ ] PyTorch 2.1
  - [ ] PyTorch 2.2
- [ ] Test with multiple CUDA versions:
  - [ ] CUDA 11.8
  - [ ] CUDA 12.1
- [ ] Collect feedback from beta testers

### Release Day

- [ ] Merge release branch to main: `git checkout main && git merge release/vX.Y.Z`
- [ ] Create and push tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`
- [ ] Build distribution: `python setup.py sdist bdist_wheel`
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Create GitHub release with release notes
- [ ] Upload pre-trained models to Hugging Face Hub
- [ ] Update documentation website
- [ ] Announce release:
  - [ ] GitHub Discussions
  - [ ] Discord
  - [ ] Twitter/X
  - [ ] Reddit (r/MachineLearning)
  - [ ] Mailing list

### Post-Release (1-2 days after)

- [ ] Monitor for critical issues
- [ ] Respond to user feedback
- [ ] Update project website
- [ ] Create blog post about release
- [ ] Update citation information if paper published
- [ ] Archive old documentation versions
- [ ] Plan next release cycle

## Release Types

### Major Release (X.0.0)

**When to release:**
- Breaking API changes
- Major architectural changes
- Significant new features

**Additional requirements:**
- Migration guide must be comprehensive
- Deprecation warnings in previous version
- Extended testing period (2-3 weeks)
- Community announcement 1 month in advance

### Minor Release (0.X.0)

**When to release:**
- New features (backwards-compatible)
- Performance improvements
- New benchmarks or models

**Additional requirements:**
- Update examples and tutorials
- Benchmark comparison with previous version
- Standard testing period (1 week)

### Patch Release (0.0.X)

**When to release:**
- Bug fixes
- Documentation updates
- Security patches

**Additional requirements:**
- Minimal testing (critical paths only)
- Can be released quickly (1-2 days)
- Focus on regression testing

## Hotfix Process

For critical bugs in production:

1. Create hotfix branch from main: `git checkout -b hotfix/vX.Y.Z+1`
2. Fix the bug and add regression test
3. Update CHANGELOG.md
4. Increment patch version
5. Fast-track testing (critical paths only)
6. Merge to main and tag
7. Release immediately
8. Backport to supported versions if needed

## Version Support

- **Current major version**: Full support (features + bug fixes)
- **Previous major version**: Security fixes only (6 months)
- **Older versions**: No support

## Release Automation

We use GitHub Actions for automated releases:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```

## Release Notes Template

```markdown
# ResNet-BK vX.Y.Z

Release date: YYYY-MM-DD

## Highlights

- Major feature 1
- Major feature 2
- Major improvement 3

## What's New

### Features
- Feature 1 (#123)
- Feature 2 (#124)

### Improvements
- Improvement 1 (#125)
- Improvement 2 (#126)

### Bug Fixes
- Fix 1 (#127)
- Fix 2 (#128)

### Documentation
- Doc update 1 (#129)
- Doc update 2 (#130)

## Breaking Changes

- Breaking change 1 (see migration guide)
- Breaking change 2 (see migration guide)

## Deprecations

- Deprecated feature 1 (will be removed in vX+1.0.0)
- Deprecated feature 2 (will be removed in vX+1.0.0)

## Installation

```bash
pip install resnet-bk==X.Y.Z
```

## Upgrade

```bash
pip install --upgrade resnet-bk
```

See [MIGRATION.md](MIGRATION.md) for migration guide.

## Benchmarks

| Metric | vX.Y.Z | vX.Y.Z-1 | Change |
|--------|--------|----------|--------|
| PPL (WikiText-2) | 30.5 | 31.2 | -2.2% |
| Speed (tokens/s) | 1250 | 1200 | +4.2% |
| Memory (GB) | 8.5 | 9.2 | -7.6% |

## Contributors

Thanks to all contributors who made this release possible:

- @contributor1
- @contributor2
- @contributor3

## Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete list of changes.

## Links

- [Documentation](https://resnet-bk.readthedocs.io)
- [GitHub](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture)
- [PyPI](https://pypi.org/project/resnet-bk/)
- [Hugging Face](https://huggingface.co/resnet-bk)
```

## Emergency Rollback

If a release has critical issues:

1. Yank the release from PyPI: `twine yank resnet-bk vX.Y.Z`
2. Delete the GitHub release
3. Delete the tag: `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`
4. Announce the rollback
5. Fix the issue
6. Release new patch version

## Contact

For release-related questions:
- Email: arat252539@gmail.com
- GitHub: @neko-jpg
- Discord: #releases channel
