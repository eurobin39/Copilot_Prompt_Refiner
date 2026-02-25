#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv python: ${PYTHON_BIN}" >&2
  echo "Run: python -m venv .venv && .venv/bin/pip install -e '.[mcp]'" >&2
  exit 1
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src"
exec "${PYTHON_BIN}" -m copilot_prompt_refiner.mcp_server
