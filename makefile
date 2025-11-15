# Makefile for CRISP-DM Insurance MLOps project

# Variables
PYTHON      := python3
VENV_DIR    := venv
PIP         := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python3
CSV_PATH    := dataAssurance.csv
MODEL_PATH  := insurance_model.joblib

# Declare "phony" targets (they are not actual files)
.PHONY: setup install test evaluate cluster run-all clean

# 1) Create a virtual environment
setup:
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# 2) Install dependencies (if venv already exists)
install:
	$(PIP) install -r requirements.txt

# 3) Run tests with pytest
test:
	$(PYTHON_VENV) -m pytest

# 4) Run model evaluation (training + metrics)
evaluate:
	$(PYTHON_VENV) main.py --action evaluate --csv $(CSV_PATH)

# 5) Run clustering analysis
cluster:
	$(PYTHON_VENV) main.py --action cluster --csv $(CSV_PATH) --clusters 3

# 6) Run tests + evaluation (useful locally & in CI)
run-all: test evaluate

# 7) Clean temporary / generated files
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
	rm -f $(MODEL_PATH)

