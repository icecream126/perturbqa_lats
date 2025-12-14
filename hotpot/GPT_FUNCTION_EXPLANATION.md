# `gpt()` 함수 설명

## 개요

`gpt()` 함수는 `models.py`에 정의된 **LLM 호출을 위한 통합 인터페이스**입니다.
이 함수는 OpenAI API 또는 로컬 모델(HuggingFace)을 호출하여 텍스트를 생성합니다.

## 위치

- **정의**: `hotpot/models.py` (line 257)
- **사용**: `hotpot/lats.py`에서 `from models import gpt`로 import하여 사용

## 함수 시그니처

```python
def gpt(prompt, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None, local_model_name=None) -> list:
```

## 주요 파라미터

| 파라미터 | 타입 | 설명 | 기본값 |
|---------|------|------|--------|
| `prompt` | str | LLM에 전달할 프롬프트 문자열 | (필수) |
| `model` | str | 사용할 모델 이름 | "gpt-3.5-turbo" |
| `temperature` | float | 생성 온도 (0.0 ~ 2.0, 높을수록 다양성 증가) | 1.0 |
| `max_tokens` | int | 최대 생성 토큰 수 | 100 |
| `n` | int | 생성할 샘플 수 (여러 응답 생성) | 1 |
| `stop` | str | 생성 중단 토큰 (이 문자열이 나타나면 생성 중단) | None |
| `local_model_name` | str | 로컬 모델 이름 (model="local"일 때 사용) | None |

## 반환값

- **타입**: `list[str]`
- **내용**: LLM이 생성한 응답 문자열들의 리스트
- **예시**: `["Thought 1: I need to search...", "Thought 1: Let me think about..."]`

## 동작 방식

### 1. 모델 타입에 따른 분기

```python
if model == "local":
    # 로컬 모델 사용 (HuggingFace)
    return _chat_local(prompt, target_model, ...)
else:
    # OpenAI API 사용
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, ...)
```

### 2. OpenAI API 호출 시

- 프롬프트를 `{"role": "user", "content": prompt}` 형식의 메시지로 변환
- `chatgpt()` 함수를 통해 OpenAI ChatCompletion API 호출
- 응답에서 텍스트를 추출하여 리스트로 반환

### 3. 로컬 모델 사용 시

- HuggingFace 파이프라인을 사용하여 로컬에서 모델 실행
- 동일한 인터페이스로 응답 리스트 반환

## `lats.py`에서의 사용 예시

### 1. 액션 생성 (`get_samples()`)

```python
# LLM을 사용하여 다음 Thought와 Action 후보들을 생성
samples = gpt(prompt, n=n_generate_sample, stop=stop)
# n: 생성할 후보 수 (예: 5개)
# stop: "Observation" 토큰에서 생성 중단
```

**사용 목적**: 현재 상태에서 다음에 수행할 수 있는 액션 후보들을 생성

### 2. 가치 평가 (`get_value()`)

```python
# LLM을 사용하여 현재 궤적의 가치를 평가
value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
# n: 평가 샘플 수 (여러 샘플의 평균 사용)
```

**사용 목적**: 현재까지의 추론 궤적이 얼마나 올바른지 평가

## `lats_search()`에서의 설정

```python
# lats_search() 함수 시작 부분에서
gpt = partial(gpt, model=args.backend, temperature=args.temperature, local_model_name=getattr(args, "local_model_name", None))
```

**설명**:
- `partial()`을 사용하여 모델, temperature 등을 미리 설정
- 이후 `gpt()` 호출 시 이 설정들이 자동으로 적용됨
- 예: `gpt(prompt, n=5)` 호출 시 `model=args.backend`, `temperature=args.temperature`가 자동 적용

## 주의사항

1. **프롬프트 포맷**: 
   - PerturbQA의 경우 `<|SYSTEM|>`, `<|USER|>`, `<|ASSISTANT|>` 토큰이 포함된 프롬프트가 전달됨
   - OpenAI API는 이를 단순 문자열로 처리하므로, 모델이 이를 제대로 파싱하지 못할 수 있음

2. **비용**: 
   - OpenAI API 사용 시 토큰 사용량에 따라 비용 발생
   - `gpt_usage()` 함수로 사용량 확인 가능

3. **재시도**: 
   - `chatgpt()` 내부에서 `completions_with_backoff()`를 사용하여 자동 재시도 수행

## 관련 함수

- `chatgpt()`: OpenAI ChatCompletion API를 직접 호출
- `_chat_local()`: 로컬 모델(HuggingFace)을 사용하여 텍스트 생성
- `gpt_usage()`: 토큰 사용량 및 비용 조회

