PYTHON ?= .venv/bin/python
UV ?= uv
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup data test test-fast lint e0 e1 e2 e2-10k e2-100k e3 clean

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

e2:
	$(PYTHON) scripts/run_churn.py --config configs/experiments/e2_churn_sift10k.yaml

e2-10k:
	$(PYTHON) scripts/run_churn.py --config configs/experiments/e2_churn_sift10k.yaml

e2-100k:
	$(PYTHON) scripts/run_churn.py --config configs/experiments/e2_churn_sift100k.yaml

e3:
	$(PYTHON) scripts/run_phase1.py --config configs/experiments/e3_sift1m.yaml

clean:
	rm -rf .pytest_cache .hypothesis
