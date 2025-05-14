# Find all Jupyter notebooks
NOTEBOOKS := $(wildcard *.ipynb)

# Define expected output images (modify this pattern if needed)
OUTPUTS := $(NOTEBOOKS:.ipynb=.pdf)

# Default target: only run missing notebooks
all: $(OUTPUTS)

# Rule to execute the notebook only if the output PNG is missing
%.pdf: %.ipynb
	#jupyter nbconvert --execute $<
	jupyter nbconvert --ExecutePreprocessor.enabled=False --to notebook --execute $< --output /tmp/$<
info:
	@echo "Notebooks:" $(NOTEBOOKS)
	@echo "Outputs:" $(OUTPUTS)

# Clean output markers
clean:
	rm -f *.pdf
