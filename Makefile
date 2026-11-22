.PHONY: install pull-models run eval test

# The project's Python. On this machine (Windows) the venv lives in .venv/Scripts;
# on Linux/macOS use: make PY=.venv/bin/python <target>
PY ?= .venv/Scripts/python.exe

install:
	uv venv .venv --python 3.12
	uv pip install -e . --python $(PY)

pull-models:
	@echo "Downloading the tiny models defined in config/models.yaml..."
	ollama pull qwen2.5:0.5b
	ollama pull qwen2.5:1.5b
	ollama pull qwen2.5:3b

run:
	$(PY) -m core.swarm --domain code

eval:
	$(PY) -m eval.run

test:
	$(PY) -m pytest tests/ -v
