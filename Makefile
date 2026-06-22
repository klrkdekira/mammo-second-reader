PY ?= uv run python3

webapp:
	$(PY) -m src.web.app
