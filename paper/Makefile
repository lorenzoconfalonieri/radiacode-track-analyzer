# Build the paper. Needs a TeX distribution with pdflatex (TeX Live, MacTeX or
# the ~100 MB BasicTeX); none is installed on the machine this was written on.
#
#   brew install --cask basictex     # then, in a new shell:
#   sudo tlmgr install microtype booktabs
#
# Figures are read from ../docs, so nothing is duplicated into this folder.

TEX  = pdflatex
FLAGS = -interaction=nonstopmode -halt-on-error

paper.pdf: paper.tex $(wildcard ../docs/*.png)
	$(TEX) $(FLAGS) paper.tex
	$(TEX) $(FLAGS) paper.tex   # second pass resolves \ref and \cite

.PHONY: clean
clean:
	rm -f paper.aux paper.log paper.out paper.pdf
