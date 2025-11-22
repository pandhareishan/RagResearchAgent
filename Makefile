.PHONY: setup ingest run test fmt

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

ingest:
	python scripts/ingest.py --data-dir data/sample_docs --index-dir data/index

run:
	uvicorn src.app.main:app --reload

test:
	pytest -q

fmt:
	python -m pip install ruff && ruff check --fix . || true
