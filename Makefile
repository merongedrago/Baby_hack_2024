install:
	pip install --upgrade pip && pip install -r requirements.txt

test: 
	python -m pytest -cov=main test_MG_main.py

format:
	black *.py

lint:
	pylint --disable=R,C --ignore-patterns=test_.*?py *.py

all: install format lint test
