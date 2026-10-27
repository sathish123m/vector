PYTHON ?= .venv/bin/python
UV ?= uv
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup data test test-fast lint e0 e1 clean

setup:
	$(UV) sync --extra dev

data:
	bash scripts/download_data.sh

test:
	$(PYTHON) -m pytest -q

test-fast:
	$(PYTHON) -m pytest -q -m "not slow and not dataset"

lint:
	$(PYTHON) -m compileall -q src tests scripts

e0:
	$(PYTHON) scripts/run_phase1.py --config configs/experiments/e0_siftsmall.yaml

e1:
	$(PYTHON) scripts/run_phase1.py --config configs/experiments/e1_sift100k.yaml

clean:
	rm -rf .pytest_cache .hypothesis
