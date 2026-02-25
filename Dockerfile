FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PROMPT_REFINER_MCP_TRANSPORT=streamable-http \
    PROMPT_REFINER_MCP_HOST=0.0.0.0 \
    PROMPT_REFINER_MCP_PORT=8080 \
    PROMPT_REFINER_MCP_STREAMABLE_HTTP_PATH=/mcp

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install -e ".[mcp]"

EXPOSE 8080

CMD ["./scripts/run_mcp_server_http.sh"]
