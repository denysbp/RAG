USER := $(shell command whoami)
HOME := /home/$(USER)
PY = $(VENV)/bin/python3
SGOINFRE := $(HOME)/sgoinfre
VENV := $(SGOINFRE)/.venv
WGET := $(shell command -v wget 2> /dev/null)
CURL := $(shell command -v curl 2> /dev/null)

all: run

local:
	@test -f $(HOME)/.local/bin/env || mkdir -p $(HOME)/.local/bin && touch  $(HOME)/.local/bin/env

build_uv: local
ifdef WGET
		@wget -qO- https://astral.sh/uv/install.sh | sh
		@. $(HOME)/.local/bin/env
		@echo "We used wget to install the uv"
else ifdef CURL
			@curl -LsSf https://astral.sh/uv/install.sh | sh
			@. $(HOME)/.local/bin/env
			@echo "We used curl to install the uv"

endif

install: build_uv
	uv sync

run:
	@if [ ! -d ".venv" ]; then \
		uv sync; \
		uv run -m src; \
	else \
		uv run -m src; \
	fi
web:
	uv run fastapi dev src/web.py

clean:
	rm -rf src/__pycache__
	rm -rf .pytest_cache
	rm -rf tests/__pycache__

lint:
	uv run -m flake8 .
	uv run -m mypy  --config-file pyproject.toml .
