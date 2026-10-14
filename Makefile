export PYTHONPATH := src

.PHONY: install data pipeline dashboard test lint docker

install:    ## install dependencies
	pip install -r requirements-dev.txt
data:       ## 3,000 customers, 860K card transactions, 90K account transactions (~1 min)
	python -m fincrime.data
pipeline:   ## fraud model, AML scenarios, alert triage -> outputs/report.md and alert queues
	python -m fincrime.pipeline
dashboard:  ## http://localhost:8501
	streamlit run dashboard.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker build -t uae-bank-fraud-aml . && docker run -p 8501:8501 uae-bank-fraud-aml
