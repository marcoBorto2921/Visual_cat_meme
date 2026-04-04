.PHONY: index run lint

index:
	python scripts/build_index.py

run:
	python main.py

lint:
	ruff check src/ scripts/ main.py
