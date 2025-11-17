# Paper Generation Quick Reference

## Overview

The paper generation system auto-generates a complete LaTeX research paper for Mamba-Killer ResNet-BK, including:
- Main paper (NeurIPS/ICML/ICLR style)
- Supplementary material with extended proofs
- Theorem/proof templates
- Bibliography
- Build automation

## Quick Start

```bash
# Generate paper with default settings (NeurIPS style)
python scripts/generate_paper.py

# Generate for specific conference
python scripts/generate_paper.py --style icml
python scripts/generate_paper.py --style iclr

# Specify output directory
python scripts/generate_paper.py --output my_paper
```

## Generated Files

```
paper/
├── main.tex                    # Main paper (8 pages)
├── supplementary.tex           # Supplementary material
├── theorem_templates.tex       # Reusable theorem templates
├── references.bib              # Bibliography
├── Makefile                    # Build automation
└── README.md                   # Instructions
```

## Building the Paper

```bash
cd paper

# Build main paper
make main

# Build supplementary material
make supp

# Build everything
make all

# View PDF
make view

# Clean build files
make clean
```

## Main Paper Structure

1. **Abstract**: 150-200 words summarizing key contributions
2. **Introduction**: Motivation, problem statement, contributions
3. **Related Work**: Comparison to Mamba, SSMs, MoE, quantization
4. **Method**: Mathematical formulation with theorems
   - Birman-Schwinger operator
   - Prime-Bump initialization
   - Scattering-based routing
   - Semiseparable structure
   - Adaptive computation
5. **Experiments**: Comprehensive benchmarks
   - Long-context stability
   - Quantization robustness
   - Dynamic efficiency
   - Ablation studies
6. **Conclusion**: Summary and future work

## Supplementary Material

1. **Extended Proofs**: Complete proofs of all theorems
2. **Additional Experiments**: Multi-dataset, downstream tasks, scaling
3. **Implementation Details**: Architecture, precision, optimization
4. **Hyperparameters**: Complete settings and training details

## Theorem Templates

Pre-formatted LaTeX templates for:
- Main theorems (trace-class stability, convergence, long-context)
- Propositions (scattering phase, semiseparable complexity)
- Lemmas (resolvent bounds, GUE spacing)
- Corollaries (quantization robustness)
- Definitions (Clark measure)
- Remarks (comparisons, insights)

## Customization

### Update Author Information

Edit `main.tex`:
```latex
\author{%
  Teppei Arai \\
  Independent Researcher \\
  \texttt{arat252539@gmail.com}
}
```

### Add Experimental Results

The generator automatically loads results from `results/*.json`:
- `stability_graph_test.json`: Long-context results
- `test_quantization.json`: Quantization results
- `efficiency_graph.json`: Efficiency results

### Add Figures

Place figures in `paper/figures/`:
- `longcontext_stability.pdf`
- `quantization_robustness.pdf`
- `dynamic_efficiency.pdf`
- `training_curves.pdf`
- `scaling_curves.pdf`

### Add Citations

Edit `references.bib` to add new citations:
```bibtex
@article{yourpaper2024,
  title={Your Paper Title},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## Conference-Specific Requirements

### NeurIPS
- Page limit: 8 pages (+ unlimited references)
- Style: `neurips_2024.sty`
- Double-blind review
- Supplementary material allowed

### ICML
- Page limit: 8 pages (+ unlimited references)
- Style: `icml2024.sty`
- Double-blind review
- Supplementary material allowed

### ICLR
- Page limit: 8 pages (+ unlimited references)
- Style: `iclr2024_conference.sty`
- Double-blind review
- Supplementary material allowed

## Submission Checklist

- [ ] Update author names and affiliations
- [ ] Add all experimental results
- [ ] Generate all figures (300 DPI, vector graphics)
- [ ] Verify all citations are complete
- [ ] Check page limits
- [ ] Anonymize for double-blind review
- [ ] Include supplementary material
- [ ] Verify all theorems have proofs
- [ ] Check mathematical notation consistency
- [ ] Run spell checker
- [ ] Verify all references are cited
- [ ] Test compilation on clean system

## Requirements

### LaTeX Distribution
- TeX Live (recommended)
- MiKTeX
- MacTeX (macOS)

### Conference Style Files
Download from conference websites:
- NeurIPS: https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
- ICML: https://icml.cc/Conferences/2024/StyleFiles
- ICLR: https://iclr.cc/Conferences/2024/StyleFiles

### Required Packages
All standard packages are included:
- amsmath, amssymb, amsthm (mathematics)
- graphicx (figures)
- booktabs (tables)
- algorithm, algorithmic (algorithms)
- hyperref (links)
- cleveref (cross-references)

## Troubleshooting

### Missing Style File
```
! LaTeX Error: File `neurips_2024.sty' not found.
```
**Solution**: Download style file from conference website and place in `paper/` directory.

### Missing Bibliography
```
LaTeX Warning: Citation `gu2023mamba' undefined.
```
**Solution**: Run `bibtex main` then `pdflatex main` twice.

### Figure Not Found
```
! LaTeX Error: File `figures/longcontext.pdf' not found.
```
**Solution**: Generate figures using visualization scripts or create placeholder.

### Page Limit Exceeded
```
Warning: Paper exceeds 8 pages
```
**Solution**: Move content to supplementary material or reduce verbosity.

## Advanced Usage

### Custom Templates

Create custom theorem template:
```latex
\begin{theorem}[Your Theorem Name]
\label{thm:yourtheorem}
Your theorem statement here.
\end{theorem}

\begin{proof}
Your proof here.
\end{proof}
```

### Auto-Generate from Results

The generator reads JSON files from `results/` and auto-populates tables:
```python
# In your benchmark script
results = {
    "resnetbk_ppl": 28.3,
    "mamba_ppl": 29.1,
    "improvement": 1.03
}
with open("results/my_experiment.json", "w") as f:
    json.dump(results, f)
```

### Multiple Versions

Generate papers for different conferences:
```bash
python scripts/generate_paper.py --output paper_neurips --style neurips
python scripts/generate_paper.py --output paper_icml --style icml
python scripts/generate_paper.py --output paper_iclr --style iclr
```

## Integration with Benchmarks

The paper generator integrates with benchmark results:

```bash
# Run benchmarks
python scripts/mamba_vs_bk_benchmark.py --all

# Generate graphs
python scripts/generate_stability_graph.py
python scripts/generate_quantization_graph.py
python scripts/generate_efficiency_graph.py

# Generate paper with results
python scripts/generate_paper.py
```

## Related Files

- `scripts/generate_paper.py`: Main generator script
- `scripts/generate_stability_graph.py`: Long-context graphs
- `scripts/generate_quantization_graph.py`: Quantization graphs
- `scripts/generate_efficiency_graph.py`: Efficiency graphs
- `scripts/mamba_vs_bk_benchmark.py`: Benchmark pipeline

## Requirements Satisfied

This implementation satisfies:
- **Requirement 15.1**: LaTeX paper template with NeurIPS/ICML style
- **Requirement 15.2**: Auto-generate method section from implementation
- **Requirement 15.3**: Auto-generate experiment section from results
- **Requirement 15.4**: Auto-generate related work section
- **Requirement 15.7**: Auto-generate related work comparisons
- **Requirement 15.8**: Highlight key differences and advantages
- **Requirement 15.9**: Generate supplementary material
- **Requirement 15.10**: Include extended proofs and experiments
- **Requirement 15.11**: Provide theorem/proof templates
- **Requirement 15.12**: Include assumptions and proof sketches

## Example Workflow

```bash
# 1. Run all benchmarks
python scripts/mamba_vs_bk_benchmark.py --all

# 2. Generate visualization graphs
python scripts/generate_stability_graph.py
python scripts/generate_quantization_graph.py
python scripts/generate_efficiency_graph.py

# 3. Generate paper
python scripts/generate_paper.py --style neurips

# 4. Build paper
cd paper
make all

# 5. View result
make view

# 6. Customize and iterate
# Edit main.tex, add figures, update results
make all
```

## Tips for Success

1. **Start Early**: Generate paper skeleton early in project
2. **Iterate Often**: Regenerate as results improve
3. **Version Control**: Commit generated files to track changes
4. **Peer Review**: Have colleagues review before submission
5. **Proofread**: Check for typos, grammar, consistency
6. **Test Build**: Verify compilation on clean system
7. **Backup**: Keep multiple versions and backups

## Support

For issues or questions:
1. Check this quick reference
2. Review generated README.md in paper directory
3. Consult LaTeX documentation
4. Check conference submission guidelines
5. Review example papers from previous years
