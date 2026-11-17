# Shortcuts for the common steps. Everything else: python scripts/reproduce.py groups
PYTHON ?= python

.PHONY: help setup data test analysis groups clean

help:
	@echo "setup     install the package and the pinned dependencies into the active environment"
	@echo "data      download and verify the corpus"
	@echo "test      run the test suite (offline, CPU, about a minute)"
	@echo "analysis  regenerate the paper's tables and numbers from results/"
	@echo "groups    list the experiment groups (python scripts/reproduce.py)"

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(PYTHON) -m prosody_llms.data.download

test:
	$(PYTHON) -m pytest

analysis:
	$(PYTHON) -m prosody_llms.analysis

groups:
	$(PYTHON) scripts/reproduce.py groups

clean:
	rm -rf .pytest_cache build *.egg-info src/*.egg-info
