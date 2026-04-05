.PHONY: collect train run lint

collect:
	python scripts/collect_samples.py

train:
	python scripts/train_classifier.py

run:
	python main.py

lint:
	ruff check src/ scripts/ main.py
