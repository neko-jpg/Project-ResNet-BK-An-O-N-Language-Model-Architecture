# ResNet-BK Paper

LaTeX source for: **"Mamba-Killer: A Mathematically Rigorous O(N) Language Model via Birman-Schwinger Operator Theory"**

## 📁 Files

- `main.tex` - Main paper (8 pages)
- `supplementary.tex` - Supplementary material (unlimited)
- `references.bib` - Bibliography (50+ citations)
- `theorem_templates.tex` - Theorem formatting
- `Makefile` - Build automation
- `WRITING_GUIDE.md` - Complete writing guide
- `COMPLETION_STATUS.md` - Progress tracker
- `PAPER_TODO.md` - Detailed checklist

## 🚀 Quick Start

### Compile the Paper

```bash
make all          # Compile main + supplementary
make main         # Compile main paper only
make supp         # Compile supplementary only
make view         # Open compiled PDF
```

### Generate Tables from Results

```bash
make tables       # Generate tables from experimental results
```

### Check for Issues

```bash
make check        # Check for TODOs, empty citations, etc.
```

### Create arXiv Submission

```bash
make arxiv        # Create submission package
```

## 📊 Current Status

**Overall Progress**: 75% Complete

✅ **Complete**:
- Paper structure and organization
- Mathematical foundations
- Method description
- References and citations
- Supplementary material
- Build infrastructure

⏳ **In Progress**:
- Experimental results (need to run)
- Figures (need to generate)
- Tables (need real data)

🔴 **To Do**:
- Final proofreading
- Conference formatting
- arXiv submission

See [COMPLETION_STATUS.md](COMPLETION_STATUS.md) for details.

## 🧪 Running Experiments

Before submission, run experiments:

```bash
# From project root
bash scripts/benchmarks/run_all_paper_experiments.sh
```

Takes 24-48 hours on 4× T4 GPUs.

## 📈 Generating Figures

After experiments:

```bash
python scripts/benchmarks/generate_stability_graph.py
python scripts/benchmarks/generate_quantization_graph.py
python scripts/benchmarks/generate_efficiency_graph.py
```

## 📝 Paper Structure

### Main Paper (8 pages)
1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experiments
6. Conclusion

### Supplementary (unlimited)
1. Extended Proofs
2. Additional Experiments
3. Implementation Details
4. Reproducibility

## 🎯 Target Conferences

- **NeurIPS 2025** (May deadline)
- **ICML 2025** (January deadline)
- **ICLR 2026** (September deadline)

## 📚 Resources

- [WRITING_GUIDE.md](WRITING_GUIDE.md) - Complete guide
- [COMPLETION_STATUS.md](COMPLETION_STATUS.md) - Progress
- [PAPER_TODO.md](PAPER_TODO.md) - Checklist

## ✅ Pre-Submission Checklist

- [ ] All experiments completed
- [ ] All figures generated
- [ ] All tables filled
- [ ] References complete
- [ ] Proofread 3+ times
- [ ] Format check
- [ ] Page limit check

---

**Status**: Ready for experiments
**Next**: Run experiments
**ETA**: 3-4 weeks
