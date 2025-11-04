NOTEBOOKS := $(wildcard *.ipynb)
OUTPUTS := $(NOTEBOOKS:.ipynb=.pdf)

all: $(OUTPUTS)

%.pdf: %.ipynb
	#jupyter nbconvert --execute $<
	jupyter nbconvert --ExecutePreprocessor.enabled=False --to notebook --execute $< --output /tmp/$<
info:
	@echo "Notebooks:" $(NOTEBOOKS)
	@echo "Outputs:" $(OUTPUTS)

# Clean output markers
clean:
	rm -f *.pdf
