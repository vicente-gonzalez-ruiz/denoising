default:	all

.PHONY: clean all mrproper

all:	main.pdf

BIBINPUTS += ".:~/bibs"

main.pdf:	*.tex
		pdflatex main
		(export BIBINPUTS=${BIBINPUTS}:${BIBS}; bibtex main)
		pdflatex main
		makeglossaries main
		pdflatex main

clean:
		rm -f *.aux *.bbl *.blg *.log *.out *.pdf
