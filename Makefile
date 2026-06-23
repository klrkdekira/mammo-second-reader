PY ?= uv run python3

sync:
	uv sync

webapp:
	$(PY) -m src.web.app

splits:
	$(PY) -m src.data.make_splits

cache:
	$(PY) -m src.data.dicom_to_png

pipeline: data train results
