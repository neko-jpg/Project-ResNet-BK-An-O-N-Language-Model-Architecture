# ResNet-BK Paper Writing Guide

## 📝 Current Status

### ✅ Completed
- [x] Paper structure (main.tex)
- [x] Supplementary material (supplementary.tex)
- [x] References (references.bib)
- [x] Makefile for compilation
- [x] Table generation script
- [x] Mathematical foundations
- [x] Method description
- [x] Experimental setup

### ⏳ In Progress
- [ ] Actual experimental results
- [ ] Generated figures
- [ ] Statistical significance tests
- [ ] Ablation studies

### 📋 To Do
- [ ] Proofread and polish
- [ ] Add author information
- [ ] Format for conference submission
- [ ] Create arXiv version

## 🚀 Quick Start

### 1. Compile the Paper

```bash
cd paper
make all
```

This will generate:
- `main.pdf` - Main paper
- `supplementary.pdf` - Supplementary material

### 2. View the Paper

```bash
make view
```

### 3. Check for Issues

```bash
make check
```

## 📊 Adding Experimental Results

### Step 1: Run Experiments

```bash
# From project root
bash scripts/benchmarks/run_all_paper_experiments.sh
```

This will take 24-48 hours on 4× T4 GPUs.

### Step 2: Generate Tables

```bash
cd paper
make tables
```

This creates `generated_tables.tex` with all tables filled in.

### Step 3: Update main.tex

Replace placeholder tables in `main.tex` with:

```latex
\input{generated_tables.tex}
```

### Step 4: Add Figures

Place generated figures in `paper/figures/`:
- `figure1_stability.pdf`
- `figure2_quantization.pdf`
- `figure3_efficiency.pdf`

Then update figure references in `main.tex`.

## 📐 Paper Structure

### Main Paper (8 pages)

1. **Abstract** (150-200 words)
   - Problem statement
   - Key contributions
   - Main results
   - Reproducibility

2. **Introduction** (1.5 pages)
   - Motivation
   - Limitations of existing work
   - Our contributions
   - Paper organization

3. **Related Work** (1 page)
   - Efficient language models
   - Mixture-of-experts
   - Quantization
   - Mathematical foundations

4. **Method** (2.5 pages)
   - Birman-Schwinger formulation
   - Prime-Bump initialization
   - Scattering-based routing
   - Semiseparable structure
   - Adaptive computation

5. **Experiments** (2 pages)
   - Experimental setup
   - Long-context stability
   - Quantization robustness
   - Dynamic efficiency
   - Ablation studies
   - Statistical significance

6. **Conclusion** (0.5 pages)
   - Summary of contributions
   - Limitations
   - Future work
   - Broader impact

7. **References** (0.5 pages)

### Supplementary Material (unlimited pages)

1. **Extended Proofs**
   - All theorem proofs
   - Mathematical derivations

2. **Additional Experiments**
   - Multi-dataset results
   - Downstream tasks
   - Scaling analysis
   - Memory profiling

3. **Implementation Details**
   - Architecture specifications
   - Hyperparameters
   - Optimization techniques
   - Stability monitoring

4. **Reproducibility**
   - Complete setup instructions
   - Hardware requirements
   - Software versions
   - Random seeds

## ✍️ Writing Tips

### Mathematical Content

1. **Define all notation**
   - Use consistent notation throughout
   - Define symbols on first use
   - Include notation table in appendix

2. **Provide intuition**
   - Don't just state formulas
   - Explain what they mean
   - Use examples

3. **Reference theorems**
   - Cite original papers
   - Explain relevance to our work

### Experimental Content

1. **Be specific**
   - Exact hyperparameters
   - Hardware specifications
   - Software versions
   - Random seeds

2. **Show statistical significance**
   - Mean ± std over multiple runs
   - p-values with Bonferroni correction
   - Confidence intervals

3. **Fair comparison**
   - Identical settings for all baselines
   - Same data, tokenization, vocabulary
   - Normalize by compute, not wall-clock time

### Writing Style

1. **Be concise**
   - Every word counts (8-page limit)
   - Remove redundancy
   - Use active voice

2. **Be precise**
   - Avoid vague terms ("better", "faster")
   - Use specific numbers ("2× fewer FLOPs")
   - Quantify everything

3. **Be honest**
   - Acknowledge limitations
   - Report negative results
   - Don't oversell

## 🎨 Figures and Tables

### Figure Guidelines

1. **High quality**
   - Vector graphics (PDF, not PNG)
   - 300 DPI minimum
   - Readable font sizes (≥8pt)

2. **Clear labels**
   - Axis labels with units
   - Legend with clear descriptions
   - Caption explaining what to see

3. **Consistent style**
   - Same color scheme
   - Same font family
   - Same line styles

### Table Guidelines

1. **Formatting**
   - Use `booktabs` package
   - Horizontal lines only
   - Align numbers on decimal point

2. **Content**
   - Bold best results
   - Include error bars (±std)
   - Show statistical significance

3. **Caption**
   - Describe what table shows
   - Explain key findings
   - Reference in text

## 📤 Submission Checklist

### Before Submission

- [ ] All experiments completed
- [ ] All figures generated
- [ ] All tables filled in
- [ ] References complete and formatted
- [ ] Supplementary material complete
- [ ] Proofread for typos
- [ ] Check math notation consistency
- [ ] Verify all citations
- [ ] Run spell checker
- [ ] Check page limit (8 pages)
- [ ] Anonymize for review (if required)

### Conference-Specific

#### NeurIPS
- [ ] Use `neurips_2024.sty`
- [ ] 8 pages + unlimited references
- [ ] Supplementary material allowed
- [ ] Anonymize submissions
- [ ] Include reproducibility checklist

#### ICML
- [ ] Use `icml2024.sty`
- [ ] 8 pages + unlimited references
- [ ] Supplementary material allowed
- [ ] Anonymize submissions

#### ICLR
- [ ] Use OpenReview format
- [ ] 8 pages + unlimited references
- [ ] Supplementary material allowed
- [ ] Public reviews

### arXiv Submission

```bash
make arxiv
```

This creates `resnet_bk_arxiv.tar.gz` with:
- LaTeX source files
- Compiled bibliography
- Figures
- All necessary files

Upload to arXiv:
1. Go to https://arxiv.org/submit
2. Upload `resnet_bk_arxiv.tar.gz`
3. Select category: cs.LG (Machine Learning)
4. Add cross-lists: cs.CL, math.SP
5. Submit

## 🔧 Troubleshooting

### Compilation Errors

**Problem**: `! LaTeX Error: File 'neurips_2024.sty' not found`

**Solution**: Download from https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles

**Problem**: `! Undefined control sequence \cite{...}`

**Solution**: Run `bibtex main` then `pdflatex main` twice

**Problem**: Figures not showing

**Solution**: Check figure paths, ensure PDFs exist in `figures/` directory

### Content Issues

**Problem**: Over page limit

**Solution**: 
- Move details to supplementary
- Reduce figure sizes
- Tighten writing
- Use `\vspace{-2mm}` sparingly

**Problem**: Missing references

**Solution**: Add to `references.bib`, run `make main` again

## 📚 Resources

### LaTeX
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [TikZ Examples](https://texample.net/tikz/)

### Writing
- [How to Write a Great Research Paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
- [Mathematical Writing](http://jmlr.csail.mit.edu/reviewing-papers/knuth_mathematical_writing.pdf)

### Conferences
- [NeurIPS](https://neurips.cc/)
- [ICML](https://icml.cc/)
- [ICLR](https://iclr.cc/)

## 💡 Tips for Success

1. **Start early**: Don't wait until deadline
2. **Get feedback**: Share drafts with colleagues
3. **Iterate**: Expect multiple revisions
4. **Be thorough**: Complete experiments, not just promising ones
5. **Be honest**: Acknowledge limitations
6. **Be clear**: Explain complex ideas simply
7. **Be reproducible**: Provide all details

## 📞 Getting Help

If you encounter issues:

1. Check this guide
2. Read conference guidelines
3. Look at accepted papers from previous years
4. Ask on LaTeX Stack Exchange
5. Contact conference organizers

## 🎯 Final Checklist

Before final submission:

- [ ] Compile without errors
- [ ] All figures visible
- [ ] All tables complete
- [ ] All references cited
- [ ] Supplementary material complete
- [ ] Reproducibility statement included
- [ ] Code/data links working
- [ ] Proofread 3+ times
- [ ] Colleague review
- [ ] Format check
- [ ] Page limit check
- [ ] Anonymization check (if required)
- [ ] Submit!

Good luck! 🚀
