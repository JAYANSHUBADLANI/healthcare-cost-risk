PYTHON ?= python3
SAMPLE ?= 2

.PHONY: help data test phase1 phase2 phase3 phase4 all clean

help:
	@echo "make data    download and unpack the CMS DE-SynPUF subsample (SAMPLE=2 by default)"
	@echo "make test    run the test suite"
	@echo "make phase1  build the member-year panel and audit what the data supports"
	@echo "make phase2  fit the two part model and evaluate it prospectively"
	@echo "make phase3  high cost identification, risk tiering and the business case"
	@echo "make phase4  driver explainability and figures"
	@echo "make all     run every phase in order"
	@echo "make clean   remove cached interim data and generated reports"

data:
	bash scripts/download_data.sh $(SAMPLE)

test:
	$(PYTHON) -m pytest -q

phase1:
	$(PYTHON) run.py 1

phase2:
	$(PYTHON) run.py 2

phase3:
	$(PYTHON) run.py 3

phase4:
	$(PYTHON) run.py 4

all:
	$(PYTHON) run.py all

clean:
	rm -rf data/interim reports/tables reports/figures reports/*.json
	find . -name __pycache__ -type d -exec rm -rf {} +
