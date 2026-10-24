# report/Makefile
# ================
# Build the arXiv paper PDF.
#
# Usage:
#   make tables    -- generate LaTeX tables from benchmark CSV
#   make           -- compile PDF (requires pdflatex + bibtex)
#   make clean     -- remove build artifacts
#   make all       -- tables + PDF

PYTHON   = python
PDFLATEX = pdflatex
BIBTEX   = bibtex
MAIN     = main

.PHONY: all tables pdf clean

all: tables pdf

tables:
	cd .. && $(PYTHON) report/generate_tables.py
	@echo "Tables generated."

pdf:
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	@echo "PDF built: $(MAIN).pdf"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk
	@echo "Cleaned build artifacts."
