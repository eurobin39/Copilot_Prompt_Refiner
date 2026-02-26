# Copilot Prompt Refiner

A plugin-style framework that evaluates and improves external agent system prompts using a Microsoft Agent Framework Judge/Refine agent pattern.

## Goals
- Collect VS Code Copilot chat logs, user input, and file context
- Score prompt quality with multi-model ensemble evaluation
- Generate improvement feedback with Judge Agent
- Generate prompt revisions with Refine Agent
- Expose MCP tools for external agent integration
- Support iterative loops: `evaluate -> judge(ensemble/aggregation/tie-breaker) -> refine(small patch)`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set up `.env`:

```bash
cp .env.example .env
```

Required:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

Recommended:
- `PROMPT_REFINER_MAX_ITERS`
- `PROMPT_REFINER_JUDGE_MODELS`
- `PROMPT_REFINER_REFINE_MAX_GROWTH` (optional; no growth cap when empty)

To use the MCP server:

```bash
pip install -e .[mcp]
```

## CLI Usage

Example payload file:

```bash
cat > payload.json <<'JSON'
{
  "payload_input": {
    "workspace": "remote-repo",
    "user_input": "Summarize the outage root cause.",
    "definition_py_content": "SYSTEM_PROMPT = \"You are an agent ...\"",
    "logs": [
      {"role": "user", "content": "Summarize the outage root cause."},
      {"role": "assistant", "content": "Analyzing"}
    ],
    "ground_truth_content": {"ground_truth": "Missing DB migration"},
    "context_files": ["definition.py", "logs/agent.json"]
  }
}
JSON
```

Inspect resolved case input:

```bash
prompt-refiner discover --payload-file payload.json
```

Run evaluation/refinement:

```bash
prompt-refiner evaluate --payload-file payload.json
prompt-refiner refine --payload-file payload.json
prompt-refiner run --payload-file payload.json
```

## MCP Server

Run with stdio transport:

```bash
prompt-refiner-mcp
```

Or run with helper script (useful from external repos):

```bash
./scripts/run_mcp_server.sh
```

Run as remote HTTP (Streamable HTTP):

```bash
PROMPT_REFINER_MCP_TRANSPORT=streamable-http \
PROMPT_REFINER_MCP_HOST=0.0.0.0 \
PROMPT_REFINER_MCP_PORT=8080 \
PROMPT_REFINER_MCP_STREAMABLE_HTTP_PATH=/mcp \
PROMPT_REFINER_MCP_STATELESS_HTTP=true \
./scripts/run_mcp_server_http.sh
```

In Copilot Agent mode, `command` (subprocess + stdio) is usually the most stable.
For HTTP transport, `PROMPT_REFINER_MCP_STATELESS_HTTP=true` is recommended.

Available MCP tools:
- `discover_case_input`
- `evaluate_prompt`
- `refine_prompt`
- `run_refinement_pipeline`

If Copilot reads files such as `definition.py`, `logs/*`, and `ground_truth*` and forwards them as `payload_input`, the server can evaluate/refine without local file path access.
`payload_input.user_input` is required by default (recommended: pass current Copilot chat input). Set `require_user_input=false` only when log-based inference is intended.
`run_refinement_pipeline` supports overriding iterations via `payload_input.max_iters`.

`run_refinement_pipeline` example:

```json
{
  "payload_input": {
    "workspace": "remote-repo",
    "user_input": "Summarize the deployment failure cause.",
    "definition_py_content": "SYSTEM_PROMPT = \"You are an agent ...\"",
    "logs": "[{\"role\":\"user\",\"content\":\"Summarize the deployment failure cause.\"},{\"role\":\"assistant\",\"content\":\"...\"}]",
    "ground_truth_content": "{\"ground_truth\":\"Missing migration caused the failure\"}"
  }
}
```

`context`-based payloads from external Copilot clients are also supported:

```json
{
  "payload_input": {
    "user_input": "Improve Resume Assistant prompts",
    "context": {
      "project": "Resume Assistant",
      "current_system_prompts": {
        "resume_info_collector": "You are ...",
        "resume_job_analyzer": "You are ...",
        "resume_writer": "You are ...",
        "resume_reviewer": "You are ..."
      }
    }
  }
}
```

Compatibility mappings:
- `context.current_system_prompts` -> auto-converted to internal `prompt_sources`
- `context.user_input` -> fallback source for `payload_input.user_input`
- `context.ground_truth`, `context.ground_truth_content`, `context.logs`, `context.log_sources` -> same-meaning fallback fields

For multi-agent repos with prompts split across files:
- Pass prompt file contents through `prompt_sources` (or `files`)
- The server extracts candidate system prompts per file, scores them, and picks the best one

```json
{
  "payload_input": {
    "prompt_sources": [
      {"path": "agents/a/definition.py", "content": "SYSTEM_PROMPT = \"You are Agent A ...\""},
      {"path": "agents/b/reviewer.ts", "content": "export const systemPrompt = `You are Agent B ...`"}
    ],
    "user_input": "Analyze this workflow.",
    "log_sources": [
      {"path": "logs/old.json", "modified_at": "2026-02-01T10:00:00Z", "content": "[{\"role\":\"user\",\"content\":\"old\"}]"},
      {"path": "logs/new.json", "modified_at": "2026-02-01T10:10:00Z", "content": "[{\"role\":\"user\",\"content\":\"new\"}]"}
    ]
  }
}
```

### VS Code `mcp.json` example

```json
{
  "servers": {
    "copilotPromptRefiner": {
      "command": "prompt-refiner-mcp",
      "args": []
    }
  }
}
```

### VS Code remote HTTP MCP example

```json
{
  "servers": {
    "copilotPromptRefinerRemote": {
      "type": "http",
      "url": "https://YOUR_DOMAIN/mcp",
      "headers": {
        "Authorization": "Bearer ${env:MCP_API_TOKEN}"
      }
    }
  }
}
```

Notes:
- Use a client-reachable host in the URL, not `0.0.0.0`.
- Stateless HTTP mode is recommended.

### Using this MCP from an external repo

To call this server from another agent repo:

1. Prepare this repo for server execution
: Create `.venv`, run `pip install -e ".[mcp]"`, and set `.env`

2. Register absolute path in external repo `mcp.json`
: Replace `command` in `samples/mcp.external.example.json` with your local path

3. Invoke tools from external Copilot
: Call `discover_case_input`, then run `run_refinement_pipeline`

4. Have Copilot pass source files through payload
: Forward `prompt_sources`, `log_sources`, `ground_truth_content`, and `user_input` in `payload_input`

Notes:
- `run_refinement_pipeline` improves prompts but does not automatically re-run your external agent.
- Re-run the external agent to produce fresh logs, then evaluate again.
- Sample files:
- `samples/mcp.external.example.json` (local stdio/command)
- `samples/mcp.remote.http.example.json` (remote HTTP URL)
- `samples/payload.resume_assistant.context.json` (`context.current_system_prompts` payload example)

## Docker Deployment (Remote Sharing)

Build image:

```bash
docker build -t copilot-prompt-refiner-mcp:latest .
```

Run container (Streamable HTTP):

```bash
docker run --rm -p 8080:8080 \
  -e AZURE_OPENAI_ENDPOINT \
  -e AZURE_OPENAI_API_KEY \
  -e OPENAI_API_VERSION \
  -e AZURE_OPENAI_MODEL \
  -e PROMPT_REFINER_MCP_TRANSPORT=streamable-http \
  -e PROMPT_REFINER_MCP_HOST=0.0.0.0 \
  -e PROMPT_REFINER_MCP_PORT=8080 \
  -e PROMPT_REFINER_MCP_STREAMABLE_HTTP_PATH=/mcp \
  -e PROMPT_REFINER_MCP_STATELESS_HTTP=true \
  copilot-prompt-refiner-mcp:latest
```

## Microsoft Agent Framework Integration

`MicrosoftAgentFrameworkRuntime` in `src/copilot_prompt_refiner/agents/microsoft_agent_framework.py` calls Azure OpenAI directly.

- `PROMPT_REFINER_USE_MAF`: enable MAF (default `true`)
- `PROMPT_REFINER_STRICT_MAF`: strict MAF mode (default `true`)
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `OPENAI_API_VERSION`: API version (for example `2024-10-21`)
- `AZURE_OPENAI_MODEL`: model name
- `AZURE_OPENAI_SSL_VERIFY`: TLS certificate verification (default `true`)
- `AZURE_OPENAI_CA_BUNDLE`: custom/internal CA bundle path (optional)

Backward compatibility:
- `MAF_ENDPOINT`, `MAF_API_KEY`, `MAF_API_VERSION`, and `MAF_MODEL` are still supported as fallback env vars.

Default pipeline behavior uses both MAF Judge and MAF Refine.
To run only local heuristic mode, set `PROMPT_REFINER_USE_MAF=false`.

Judge output includes structured fields:
- `per_model_reviews`: per-model score (0-10), violation flag, failure tags, improvement suggestions
- `failure_cases`: failure type/evidence/repro input-output/required fix/success criteria
- `prioritized_actions`: actions Refine can apply directly
- `disagreement_flag`: indicates whether tie-breaker logic was needed due to model disagreement

## Troubleshooting
- `Could not infer system_prompt`
  - Provide `payload_input.system_prompt`, or pass `prompt_sources` / `context.current_system_prompts`.
- `user_input is required`
  - Pass `payload_input.user_input`, or set `require_user_input=false` with user logs (`logs`/`log_sources`).
- `[SSL: CERTIFICATE_VERIFY_FAILED]`
  - First set your organization CA bundle via `AZURE_OPENAI_CA_BUNDLE` and retry.
  - Use `AZURE_OPENAI_SSL_VERIFY=false` only for temporary diagnostics.
  - If Azure path instability should not block MCP tool execution, set `PROMPT_REFINER_STRICT_MAF=false` to allow heuristic fallback.

Some MCP clients may nest tool arguments as `{"payload_input": {"payload_input": {...}}}`; the server auto-unwraps one layer.
If `logs`, `log_sources`, or `ground_truth_content` are missing, the server auto-normalizes them to `{}`, `[]`, and `{}` to keep payload shape stable.

## Sample Data
- `samples/case_example.json`
- `samples/system_prompt.txt`
- `samples/mcp.external.example.json`
- `samples/mcp.remote.http.example.json`
- `samples/payload.resume_assistant.context.json`
