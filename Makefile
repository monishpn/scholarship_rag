PYTHON ?= python3
.PHONY: install test evaluate run
install:
	$(PYTHON) -m pip install -r requirements.txt
test:
	$(PYTHON) -m unittest discover -s tests -v
evaluate:
	$(PYTHON) evaluate.py --method all
run:
	$(PYTHON) app.py
