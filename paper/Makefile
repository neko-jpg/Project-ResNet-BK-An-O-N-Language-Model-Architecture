# Makefile for Mamba-Killer ResNet-BK Paper

.PHONY: all main supp clean view arxiv check

all: main supp

main: main.pdf

main.pdf: main.tex references.bib
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	@echo "Main paper compiled successfully: main.pdf"

supp: supplementary.pdf

supplementary.pdf: supplementary.tex references.bib
	pdflatex supplementary.tex
	bibtex supplementary
	pdflatex supplementary.tex
	pdflatex supplementary.tex
	@echo "Supplementary material compiled successfully: supplementary.pdf"

# Generate tables from experimental results
tables: generated_tables.tex

generated_tables.tex:
	python ../scripts/benchmarks/generate_paper_tables.py \
		--results_dir ../results/paper_experiments \
		--output generated_tables.tex

# Check for common LaTeX issues
check:
	@echo "Checking for common issues..."
	@grep -n "TODO\|FIXME\|XXX" *.tex || echo "No TODOs found"
	@grep -n "\\cite{}" *.tex || echo "No empty citations"
	@grep -n "Figure~\\ref{fig:[^}]*}" *.tex | grep -v "\\label{fig:" || echo "All figures referenced"
	@echo "Check complete"

# Create arXiv submission package
arxiv: main.pdf supplementary.pdf
	@echo "Creating arXiv submission package..."
	mkdir -p arxiv_submission
	cp main.tex supplementary.tex references.bib arxiv_submission/
	cp main.bbl supplementary.bbl arxiv_submission/
	cp -r figures arxiv_submission/ 2>/dev/null || true
	cd arxiv_submission && tar czf ../resnet_bk_arxiv.tar.gz *
	@echo "arXiv package created: resnet_bk_arxiv.tar.gz"

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
	rm -f *.fdb_latexmk *.fls *.synctex.gz
	@echo "Cleaned auxiliary files"

# Clean everything including PDFs
distclean: clean
	rm -f *.pdf
	rm -rf arxiv_submission resnet_bk_arxiv.tar.gz
	@echo "Cleaned all generated files"

# View the compiled paper
view: main.pdf
	@if command -v open > /dev/null; then \
		open main.pdf; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open main.pdf; \
	elif command -v evince > /dev/null; then \
		evince main.pdf; \
	else \
		echo "No PDF viewer found. Please open main.pdf manually."; \
	fi

# Continuous compilation (requires latexmk)
watch:
	latexmk -pdf -pvc main.tex

# Word count (approximate)
wordcount:
	@echo "Approximate word count:"
	@texcount -brief main.tex

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Compile main paper and supplementary material"
	@echo "  main       - Compile main paper only"
	@echo "  supp       - Compile supplementary material only"
	@echo "  tables     - Generate tables from experimental results"
	@echo "  check      - Check for common LaTeX issues"
	@echo "  arxiv      - Create arXiv submission package"
	@echo "  view       - Open compiled PDF"
	@echo "  watch      - Continuous compilation (requires latexmk)"
	@echo "  wordcount  - Count words (requires texcount)"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove all generated files including PDFs"
	@echo "  help       - Show this help message"

