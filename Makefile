.PHONY: run install lint

install:
	pip install -r requirements.txt

run:
	python main.py

lint:
	ruff check src/ main.py
