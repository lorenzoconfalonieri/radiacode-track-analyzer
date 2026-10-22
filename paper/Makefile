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

# arXiv compiles one flat folder, so the figures are copied in beside the
# source and \graphicspath pointed at it. The staged copy is built once here
# to prove it compiles on its own before anything is uploaded.
ARXIV = arxiv
FIGS  = $(shell sed -n 's/.*includegraphics[^{]*{\([^}]*\)}.*/\1/p' paper.tex)

arxiv.tar.gz: paper.tex $(addprefix ../docs/,$(FIGS))
	rm -rf $(ARXIV) && mkdir $(ARXIV)
	sed 's|^\\graphicspath{{../docs/}}|\\graphicspath{{./}}|' paper.tex > $(ARXIV)/paper.tex
	cp $(addprefix ../docs/,$(FIGS)) $(ARXIV)/
	cd $(ARXIV) && $(TEX) $(FLAGS) paper.tex > /dev/null && $(TEX) $(FLAGS) paper.tex > /dev/null
	cd $(ARXIV) && tar czf ../arxiv.tar.gz paper.tex $(FIGS)
	@echo "arxiv.tar.gz: $$(tar tzf arxiv.tar.gz | tr '\n' ' ')"

.PHONY: clean arxiv
arxiv: arxiv.tar.gz
clean:
	rm -rf paper.aux paper.log paper.out paper.pdf $(ARXIV) arxiv.tar.gz
