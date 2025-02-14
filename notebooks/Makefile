# Find all Jupyter notebooks
NOTEBOOKS := $(wildcard *.ipynb)

# Define expected output images (modify this pattern if needed)
OUTPUTS := $(NOTEBOOKS:.ipynb=.png)

# Default target: only run missing notebooks
all: $(OUTPUTS)

# Rule to execute the notebook only if the output PNG is missing
%.png: %.ipynb
	jupyter nbconvert --execute --inplace $< && touch $@

# Clean output markers
clean:
	rm -f *.png
