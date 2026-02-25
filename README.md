# Copilot Prompt Refiner

Microsoft Agent Framework 기반 Judge/Refine 에이전트 패턴을 사용해, 외부 에이전트 시스템 프롬프트를 평가/개선하는 플러그인형 프레임워크입니다.

## 목표
- VS Code Copilot 대화 로그/입력/파일 컨텍스트를 수집
- 복수 평가 모델(앙상블)로 품질 점수화
- Judge Agent가 개선 피드백 생성
- Refine Agent가 시스템 프롬프트 리비전 생성
- MCP 서버 툴로 외부 에이전트에서 호출 가능
- evaluate -> judge(ensemble/aggregation/tie-breaker) -> refine(small patch) 루프 지원

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`.env` 설정:

```bash
cp .env.example .env
```

필수 입력:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

권장 입력:
- `PROMPT_REFINER_MAX_ITERS`
- `PROMPT_REFINER_JUDGE_MODELS`
- `PROMPT_REFINER_REFINE_MAX_GROWTH` (옵션, 비우면 성장 제한 없음)

MCP 서버까지 사용하려면:

```bash
pip install -e .[mcp]
```

## CLI 사용

Payload 파일 예시:

```bash
cat > payload.json <<'JSON'
{
  "payload_input": {
    "workspace": "remote-repo",
    "user_input": "장애 원인을 요약해줘",
    "definition_py_content": "SYSTEM_PROMPT = \"You are an agent ...\"",
    "logs": [
      {"role": "user", "content": "장애 원인 요약해줘"},
      {"role": "assistant", "content": "분석 중"}
    ],
    "ground_truth_content": {"ground_truth": "DB migration 누락"},
    "context_files": ["definition.py", "logs/agent.json"]
  }
}
JSON
```

케이스 구성 확인:

```bash
prompt-refiner discover --payload-file payload.json
```

평가/개선 실행:

```bash
prompt-refiner evaluate --payload-file payload.json
prompt-refiner refine --payload-file payload.json
prompt-refiner run --payload-file payload.json
```

## MCP 서버

stdio transport로 실행:

```bash
prompt-refiner-mcp
```

또는 외부 레포에서 경로 문제 없이 실행하려면:

```bash
./scripts/run_mcp_server.sh
```

원격 HTTP(Streamable HTTP)로 실행:

```bash
PROMPT_REFINER_MCP_TRANSPORT=streamable-http \
PROMPT_REFINER_MCP_HOST=0.0.0.0 \
PROMPT_REFINER_MCP_PORT=8080 \
PROMPT_REFINER_MCP_STREAMABLE_HTTP_PATH=/mcp \
PROMPT_REFINER_MCP_STATELESS_HTTP=true \
./scripts/run_mcp_server_http.sh
```

Copilot Agent mode에서는 `command`(subprocess, stdio) 방식이 가장 안정적입니다.
HTTP를 사용할 때는 `PROMPT_REFINER_MCP_STATELESS_HTTP=true`를 권장합니다(세션 ID 없이 호출 가능).

외부 에이전트에서 사용할 MCP tools:
- `discover_case_input`
- `evaluate_prompt`
- `refine_prompt`
- `run_refinement_pipeline`

Copilot이 `definition.py`, `logs/*`, `ground_truth*` 파일 내용을 읽어 `payload_input`으로 전달하면 서버가 파일 경로 접근 없이 평가/개선을 수행합니다.
`payload_input.user_input`은 기본적으로 필수(현재 Copilot 채팅 입력값 전달 권장)이며, 필요 시 `require_user_input=false`로 로그 기반 추론을 허용할 수 있습니다.
`run_refinement_pipeline`은 `payload_input.max_iters`로 iteration 수를 오버라이드할 수 있습니다.

`run_refinement_pipeline` 예시:

```json
{
  "payload_input": {
    "workspace": "remote-repo",
    "user_input": "배포 실패 원인 요약해줘",
    "definition_py_content": "SYSTEM_PROMPT = \"You are an agent ...\"",
    "logs": "[{\"role\":\"user\",\"content\":\"배포 실패 원인 요약해줘\"},{\"role\":\"assistant\",\"content\":\"...\"}]",
    "ground_truth_content": "{\"ground_truth\":\"마이그레이션 누락이 원인\"}"
  }
}
```

외부 Copilot에서 자주 보내는 `context` 기반 payload도 지원합니다:

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

호환 매핑:
- `context.current_system_prompts` -> 내부 `prompt_sources`로 자동 변환
- `context.user_input` -> `payload_input.user_input` 대체 입력으로 사용
- `context.ground_truth`, `context.ground_truth_content`, `context.logs`, `context.log_sources` -> 동일 의미로 fallback 처리

멀티 에이전트 레포처럼 프롬프트가 여러 파일에 분산된 경우:
- `prompt_sources`(또는 `files`) 배열로 파일 내용들을 함께 전달
- 서버가 각 파일에서 시스템 프롬프트 후보를 추출하고 점수화해서 최적 후보를 선택

```json
{
  "payload_input": {
    "prompt_sources": [
      {"path": "agents/a/definition.py", "content": "SYSTEM_PROMPT = \"You are Agent A ...\""},
      {"path": "agents/b/reviewer.ts", "content": "export const systemPrompt = `You are Agent B ...`"}
    ],
    "user_input": "워크플로우를 분석해줘",
    "log_sources": [
      {"path": "logs/old.json", "modified_at": "2026-02-01T10:00:00Z", "content": "[{\"role\":\"user\",\"content\":\"old\"}]"},
      {"path": "logs/new.json", "modified_at": "2026-02-01T10:10:00Z", "content": "[{\"role\":\"user\",\"content\":\"new\"}]"}
    ]
  }
}
```

### VS Code `mcp.json` 예시

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

### VS Code 원격 HTTP MCP 예시

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

참고:
- URL은 `0.0.0.0`가 아니라 클라이언트에서 실제로 접근 가능한 주소를 사용해야 합니다.
- HTTP 서버는 stateless 모드로 띄우는 것을 권장합니다.

### 외부 레포에서 이 MCP 사용

외부 에이전트 레포에서 Copilot이 이 서버를 호출하려면:

1. 이 레포에서 서버 실행 준비
: `.venv` 생성 + `pip install -e ".[mcp]"`, `.env` 설정 완료

2. 외부 레포의 `mcp.json`에 절대 경로 등록
: `samples/mcp.external.example.json`의 `command`를 실제 경로로 교체

3. 외부 레포 Copilot에서 MCP tool 호출
: `discover_case_input`으로 입력 확인 후 `run_refinement_pipeline` 실행

4. Copilot이 외부 레포 파일을 읽어 payload 전달
: `prompt_sources`, `log_sources`, `ground_truth_content`, `user_input`를 `payload_input`으로 전달

참고:
- `run_refinement_pipeline`은 프롬프트를 개선하지만 외부 에이전트를 자동 재실행하지 않습니다.
- 개선 효과를 보려면 외부 에이전트를 재실행해 새 로그를 만들어 다시 평가해야 합니다.
- 샘플 파일:
- `samples/mcp.external.example.json` (로컬 stdio/command)
- `samples/mcp.remote.http.example.json` (원격 HTTP URL)
- `samples/payload.resume_assistant.context.json` (`context.current_system_prompts` 기반 payload 예시)

## Docker 배포(원격 공유)

이미지 빌드:

```bash
docker build -t copilot-prompt-refiner-mcp:latest .
```

컨테이너 실행(Streamable HTTP):

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

## Microsoft Agent Framework 연동

`src/copilot_prompt_refiner/agents/microsoft_agent_framework.py`의 `MicrosoftAgentFrameworkRuntime`는 Azure OpenAI를 직접 호출합니다.

- `PROMPT_REFINER_USE_MAF`: MAF 사용 여부 (기본 `true`)
- `PROMPT_REFINER_STRICT_MAF`: MAF 필수 모드 (기본 `true`)
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI 엔드포인트
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API 키
- `OPENAI_API_VERSION`: API 버전 (예: `2024-10-21`)
- `AZURE_OPENAI_MODEL`: 모델 이름
- `AZURE_OPENAI_SSL_VERIFY`: TLS 인증서 검증 여부 (기본 `true`)
- `AZURE_OPENAI_CA_BUNDLE`: 사내/커스텀 CA 번들 경로 (옵션)

하위 호환:
- 기존 `MAF_ENDPOINT`, `MAF_API_KEY`, `MAF_API_VERSION`, `MAF_MODEL`도 fallback으로 지원합니다.

파이프라인 기본 동작은 MAF Judge + MAF Refine를 사용합니다.
로컬 휴리스틱 모드로만 실행하려면 `PROMPT_REFINER_USE_MAF=false`로 설정하세요.

Judge 출력은 구조화된 필드를 포함합니다:
- `per_model_reviews`: 모델별 점수(0~10), 위반 여부, 실패 태그, 개선 제안
- `failure_cases`: 실패 유형/근거/재현 입력·출력/요구 수정/성공 조건
- `prioritized_actions`: Refine가 바로 적용할 우선순위 액션
- `disagreement_flag`: 모델 간 점수 불일치가 큰 경우 tie-breaker 판정 여부

## 트러블슈팅
- `Could not infer system_prompt`
  - `payload_input.system_prompt`를 명시하거나, `prompt_sources`/`context.current_system_prompts`를 전달하세요.
- `user_input is required`
  - `payload_input.user_input`을 전달하거나 `require_user_input=false` + user 로그(`logs`/`log_sources`)를 함께 전달하세요.
- `[SSL: CERTIFICATE_VERIFY_FAILED]`
  - 우선 `AZURE_OPENAI_CA_BUNDLE`에 조직의 CA 번들을 설정해 재시도하세요.
  - 임시 점검용으로만 `AZURE_OPENAI_SSL_VERIFY=false`를 사용할 수 있습니다(운영 비권장).
  - Azure 경로 장애 중에도 MCP 도구 실행을 지속하려면 `PROMPT_REFINER_STRICT_MAF=false`로 설정해 휴리스틱 fallback을 허용하세요.

일부 MCP 클라이언트가 툴 인자를 `{"payload_input": {"payload_input": {...}}}` 형태로 중첩 전달해도 서버가 자동으로 1단계 언랩하여 처리합니다.

## 샘플 데이터
- `samples/case_example.json`
- `samples/system_prompt.txt`
- `samples/mcp.external.example.json`
- `samples/mcp.remote.http.example.json`
- `samples/payload.resume_assistant.context.json`
