default:	all

.PHONY: clean all mrproper

all:	main.pdf

BIBINPUTS += ".:~/bibs"

main.pdf:	main.tex
		(export TEXINPUTS=${TEXINPUTS}; pdflatex main)
		(export TEXINPUTS=${TEXINPUTS}; pdflatex main)
		(export BIBINPUTS=${BIBINPUTS}; bibtex main)
		(export TEXINPUTS=${TEXINPUTS}; pdflatex main)

clean:
		rm -f *.aux *.bbl *.blg *.log *.out *.pdf
