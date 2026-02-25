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

HOST="${PROMPT_REFINER_MCP_HOST:-0.0.0.0}"
PORT="${PROMPT_REFINER_MCP_PORT:-8080}"
TRANSPORT="${PROMPT_REFINER_MCP_TRANSPORT:-streamable-http}"
STREAMABLE_PATH="${PROMPT_REFINER_MCP_STREAMABLE_HTTP_PATH:-/mcp}"
STATELESS_HTTP="${PROMPT_REFINER_MCP_STATELESS_HTTP:-true}"
LOG_LEVEL="${PROMPT_REFINER_MCP_LOG_LEVEL:-INFO}"

exec "${PYTHON_BIN}" -m copilot_prompt_refiner.mcp_server \
  --transport "${TRANSPORT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --streamable-http-path "${STREAMABLE_PATH}" \
  --stateless-http "${STATELESS_HTTP}" \
  --log-level "${LOG_LEVEL}"
