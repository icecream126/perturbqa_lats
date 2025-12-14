# PerturbQA 프롬프트 디버깅 가이드

## 프롬프트 생성 및 전달 흐름

### 1. 초기 프롬프트 생성 (`perturbqa.py`)

#### 위치: `PerturbQATask._build_full_prompt()` (line 280-335)

이 메서드에서 실제 질문 프롬프트가 생성됩니다.

**프롬프트 구조:**
```
<|SYSTEM|>
{system_query}
<|END_SYSTEM|>
<|USER|>
{prompt_context}
{user_query}
<|END_USER|>
<|ASSISTANT|>
```

**포함되는 내용:**
- `system_query`: `prompt_utils_return_prompt("gene_perturb_system_prompt")`에서 가져옴
- `prompt_context`: 
  - Cell line 정보
  - Knockdown gene에 대한 도메인 지식 (pert_descriptions)
  - Target gene에 대한 도메인 지식 (target_descriptions)
  - Interaction Metrics (STRINGdb score, GO semantic similarity)
- `user_query`: `prompt_utils_return_prompt("gene_perturb_user_prompt")`에서 가져와서 포맷팅

**예시 프롬프트:**
```
<|SYSTEM|>
You are a scientific reasoning agent...
<|END_SYSTEM|>
<|USER|>

Cell line: A549

Domain knowledge for knockdown gene GENE1:
- Description 1
- Description 2
...

Domain knowledge for target gene GENE2:
- Description 1
- Description 2
...

Interaction Metrics:
- STRINGdb score: 0.8500
- GO semantic similarity: 0.7500

What genes are affected when GENE1 is perturbed in A549 cell line?
<|END_USER|>
<|ASSISTANT|>
```

### 2. 프롬프트 래핑 (`perturbqa.py`)

#### 위치: `PerturbQATask.standard_prompt_wrap()` (line 123-124)
```python
def standard_prompt_wrap(x: str, y:str='') -> str:
    return standard_prompt.format(input=x) + y
```

**문제점:** `standard_prompt`는 HotPotQA용으로 설계되어 있어서 PerturbQA의 특수 포맷(`<|SYSTEM|>`, `<|USER|>` 등)과 맞지 않을 수 있습니다.

#### 위치: `PerturbQATask.cot_prompt_wrap()` (line 153-176)
```python
def cot_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[]):
    # reflection_mapping_list가 있으면 cot_prompt_feedback 사용
    # 없으면 cot_prompt 사용
    prompt = cot_prompt.format(input=input)
    return prompt
```

**문제점:** `cot_prompt`도 HotPotQA용이며, PerturbQA의 특수 포맷을 고려하지 않습니다.

### 3. LLM 호출 (`lats.py`)

#### 위치: `get_samples()` (line 82-97)
```python
if prompt_sample == 'standard':
    prompt = task.standard_prompt_wrap(x, y)
elif prompt_sample == 'cot':
    prompt = task.cot_prompt_wrap(x, y, reflection_map)
logging.info(f"PROMPT: {prompt}")
samples = gpt(prompt, n=n_generate_sample, stop=stop)
```

여기서 `x`는 `generate_prompt(node)`의 결과로, `node.question` (즉, `_build_full_prompt()`의 결과) + trajectory입니다.

### 4. 실제 LLM API 호출 (`models.py`)

#### 위치: `gpt()` (line 257-267)
```python
def gpt(prompt, model="gpt-3.5-turbo", ...):
    if model == "local":
        return _chat_local(prompt, ...)
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, ...)
```

**중요한 문제점:**
- `_build_full_prompt()`에서 생성한 특수 포맷(`<|SYSTEM|>`, `<|USER|>`, `<|ASSISTANT|>`)이 그대로 문자열로 전달됩니다.
- `gpt()` 함수는 이를 단순히 `{"role": "user", "content": prompt}`로 래핑합니다.
- LLM이 이 특수 토큰들을 제대로 파싱하지 못할 수 있습니다.

## 디버깅 포인트

### 1. 프롬프트 생성 확인
**위치:** `perturbqa.py` line 84

```python
# line 81-84
question = self._build_full_prompt(pert_gene, records)
# DEBUG: 프롬프트 생성 확인
# 확인할 값: question (생성된 전체 프롬프트), pert_gene, records
import pdb; pdb.set_trace()  # ✅ 이미 추가됨
```

**확인할 값:**
- `question`: 생성된 전체 프롬프트 문자열
- `pert_gene`: perturbation gene 이름
- `records`: 관련 레코드 데이터

### 2. 프롬프트 래핑 확인
**위치:** `lats.py` line 97

```python
if prompt_sample == 'standard':
    prompt = task.standard_prompt_wrap(x, y)
elif prompt_sample == 'cot':
    prompt = task.cot_prompt_wrap(x, y, reflection_map)
# DEBUG: 프롬프트 래핑 확인
# 확인할 값: x (입력 질문), y (현재 trajectory), prompt (최종 래핑된 프롬프트), prompt_sample
import pdb; pdb.set_trace()  # ✅ 이미 추가됨
logging.info(f"PROMPT: {prompt}")
```

**확인할 값:**
- `x`: 입력 질문 (원래 `_build_full_prompt()`의 결과 + trajectory)
- `y`: 현재까지의 trajectory
- `prompt`: 최종 래핑된 프롬프트
- `prompt_sample`: 사용되는 프롬프트 타입 ('standard' 또는 'cot')

### 3. LLM API 호출 직전 확인
**위치:** `models.py` line 268

```python
def gpt(prompt, model="gpt-3.5-turbo", ...):
    if model == "local":
        return _chat_local(prompt, ...)
    # DEBUG: LLM API 호출 직전 확인
    # 확인할 값: prompt (최종 프롬프트 문자열), model, messages 구조
    import pdb; pdb.set_trace()  # ✅ 이미 추가됨
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, ...)
```

**확인할 값:**
- `prompt`: LLM에 전달되는 최종 프롬프트 문자열
- `model`: 사용되는 모델 이름
- `messages`: 실제로 API에 전달되는 메시지 구조

### 4. 초기 질문 확인
**위치:** `lats.py` line 189

```python
x = local_env.reset(idx=idx)
import pdb; pdb.set_trace()  # 여기에 추가
if to_print:
    print(idx, x)
```

**확인할 값:**
- `x`: 초기 질문 (환경에서 리셋된 질문)
- `idx`: 데이터 인덱스

## 예상되는 문제점

1. **특수 토큰 처리 문제:**
   - `_build_full_prompt()`에서 생성한 `<|SYSTEM|>`, `<|USER|>`, `<|ASSISTANT|>` 토큰이 LLM API에 그대로 전달됨
   - OpenAI API는 이런 토큰을 인식하지 못할 수 있음
   - 올바른 방법: `messages` 리스트에 `{"role": "system", "content": ...}`와 `{"role": "user", "content": ...}`로 분리

2. **프롬프트 템플릿 불일치:**
   - `standard_prompt`와 `cot_prompt`는 HotPotQA용으로 설계됨
   - PerturbQA의 특수 포맷과 호환되지 않을 수 있음

3. **프롬프트 중복:**
   - `_build_full_prompt()`에서 이미 완전한 프롬프트를 생성
   - `standard_prompt_wrap()`이나 `cot_prompt_wrap()`에서 다시 래핑하면서 구조가 깨질 수 있음

## 권장 디버깅 순서

1. **첫 번째 브레이크포인트** (`perturbqa.py` line 84): 생성된 원본 프롬프트 확인
   - `question` 변수에 생성된 전체 프롬프트가 들어있음
   - `<|SYSTEM|>`, `<|USER|>`, `<|ASSISTANT|>` 토큰 포함 여부 확인
   
2. **두 번째 브레이크포인트** (`lats.py` line 97): 래핑된 프롬프트 확인
   - `x`: 원본 질문 + trajectory
   - `prompt`: `standard_prompt_wrap()` 또는 `cot_prompt_wrap()`으로 래핑된 최종 프롬프트
   - 원본 프롬프트가 어떻게 변형되었는지 확인
   
3. **세 번째 브레이크포인트** (`models.py` line 268): LLM API 호출 직전 최종 프롬프트 확인
   - `prompt`: 실제로 LLM에 전달되는 문자열
   - `messages`: API에 전달되는 메시지 구조 확인
   - 특수 토큰이 제대로 처리되는지 확인

## 각 브레이크포인트에서 확인할 명령어

### pdb 명령어:
```python
# 프롬프트 전체 내용 확인
print(prompt)
# 또는 더 자세히
print(repr(prompt))

# 프롬프트 길이 확인
len(prompt)

# 특수 토큰 포함 여부 확인
'<|SYSTEM|>' in prompt
'<|USER|>' in prompt
'<|ASSISTANT|>' in prompt

# 프롬프트의 처음 500자 확인
prompt[:500]

# 프롬프트의 마지막 500자 확인
prompt[-500:]

# 변수 타입 확인
type(prompt)
type(x)
type(y)

# 다음 줄로 진행
n (next)
# 또는 함수 내부로 들어가기
s (step)
# 계속 실행
c (continue)
```

## 요약

**현재 설정된 디버깅 포인트:**
1. ✅ `perturbqa.py:84` - 프롬프트 생성 직후
2. ✅ `lats.py:97` - 프롬프트 래핑 직후  
3. ✅ `models.py:268` - LLM API 호출 직전

**각 포인트에서 확인할 핵심 사항:**
- 프롬프트가 올바르게 생성되는가?
- 특수 토큰(`<|SYSTEM|>`, `<|USER|>`, `<|ASSISTANT|>`)이 포함되어 있는가?
- 프롬프트 래핑 과정에서 구조가 깨지지 않았는가?
- 최종적으로 LLM에 전달되는 프롬프트가 올바른 형식인가?

각 브레이크포인트에서:
- `print(prompt)` 또는 `print(repr(prompt))`로 전체 프롬프트 출력
- 프롬프트 길이 확인: `len(prompt)`
- 특수 토큰 포함 여부 확인: `'<|SYSTEM|>' in prompt`
- 프롬프트의 처음 500자와 마지막 500자 확인

