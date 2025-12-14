# `node.state`의 `thought`, `action`, `observation` 생성 과정

## 개요

`node.state`는 딕셔너리 형태로, 각 노드의 현재 상태를 나타냅니다:
```python
{
    'thought': '생성된 추론 내용',
    'action': '수행할 액션',
    'observation': '액션 실행 결과'
}
```

이 세 가지 요소는 `generate_new_states()` 함수에서 생성됩니다.

## 생성 흐름

### 1. `generate_new_states()` 함수 호출

**위치**: `hotpot/lats.py` line 633

**역할**: 현재 노드에서 가능한 새로운 상태(자식 노드)들을 생성

**주요 단계**:
1. 노드의 현재 상태를 프롬프트로 변환
2. LLM을 호출하여 Thought와 Action 생성
3. 생성된 텍스트에서 Thought와 Action 파싱
4. 환경에서 Action 실행하여 Observation 획득
5. 새로운 노드 생성

### 2. 프롬프트 생성 (`generate_prompt()`)

**위치**: `hotpot/lats.py` line 781

```python
prompt = generate_prompt(node)
```

**동작**:
- 노드의 현재 상태(`node.state`)와 질문(`node.question`)을 기반으로 프롬프트 생성
- 루트 노드부터 현재 노드까지의 모든 `thought`, `action`, `observation`을 포함한 trajectory 생성
- 예시:
  ```
  Question: ...
  Thought 1: ...
  Action: ...
  Observation: ...
  Thought 2: ...
  Action: ...
  Observation: ...
  ```

### 3. LLM을 통한 Thought와 Action 생성 (`get_samples()`)

**위치**: `hotpot/lats.py` line 653

```python
sampled_actions = get_samples(
    task, 
    prompt,  # generate_prompt()에서 생성한 프롬프트
    f"Thought {node.depth + 1}: ",  # 시작 문자열 (예: "Thought 2: ")
    n,  # 생성할 후보 수
    prompt_sample=args.prompt_sample,  # 'standard' 또는 'cot'
    stop="Observation"  # 생성 중단 토큰
)
```

**동작**:
1. `get_samples()` 내부에서:
   - `task.cot_prompt_wrap()` 또는 `task.standard_prompt_wrap()`으로 프롬프트 래핑
   - `gpt()` 함수 호출하여 LLM 응답 생성
   - `stop="Observation"` 토큰이 나타나면 생성 중단

2. **LLM 응답 예시**:
   ```
   Thought 2: I need to determine whether CRISPRi knockdown of ABCF1 in HepG2 cells leads to a change in EIF1 expression.
   Action: Search for any known interactions or regulatory relationships between ABCF1 and EIF1 in HepG2 cells or other relevant contexts.
   ```

3. **반환값**: `sampled_actions`는 문자열 리스트
   ```python
   [
       "Thought 2: ...\nAction: ...",
       "Thought 2: ...\nAction: ...",
       ...
   ]
   ```

### 4. Thought와 Action 파싱

**위치**: `hotpot/lats.py` line 662-663

```python
# 생성된 액션 텍스트에서 Thought 라인 추출
thought_line = next(
    (line.split(":")[1].strip() 
     for line in action.split("\n") 
     if line.startswith(f"Thought {node.depth + 1}")), 
    ''
)

# 생성된 액션 텍스트에서 Action 라인 추출
action_line = next(
    (line.split(":")[1].strip() 
     for line in action.split("\n") 
     if line.startswith("Action") and ":" in line), 
    None
)
```

**동작**:
- LLM이 생성한 텍스트를 줄 단위로 분리
- `"Thought {depth}: "`로 시작하는 줄에서 콜론 뒤의 내용 추출 → `thought_line`
- `"Action: "`로 시작하는 줄에서 콜론 뒤의 내용 추출 → `action_line`

**예시**:
- 입력 텍스트: `"Thought 2: I need to determine...\nAction: Search for..."`
- `thought_line`: `"I need to determine..."`
- `action_line`: `"Search for..."`

### 5. 환경에서 Action 실행 (`step()`)

**위치**: `hotpot/lats.py` line 679

```python
# 액션 타입과 파라미터 추출
action_type = action_line.split('[')[0] if '[' in action_line else action_line
action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

# 환경에서 액션 실행
obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
```

**동작**:
1. `action_line`에서 액션 타입과 파라미터 추출
   - 예: `"Search[ABCF1]"` → `action_type="Search"`, `action_param="ABCF1"`
2. `step()` 함수 호출:
   - 환경(`env`)에서 액션 실행
   - 재시도 로직 포함 (타임아웃 시 최대 10번 재시도)
3. **반환값**:
   - `obs`: Observation (액션 실행 결과)
   - `r`: Reward (0 또는 1)
   - `done`: 에피소드 종료 여부
   - `info`: 추가 정보 (예: `em` - Exact Match)

**예시**:
- `action_line`: `"Search for any known interactions or regulatory relationships between ABCF1 and EIF1 in HepG2 cells or other relevant contexts."`
- 파싱 결과: 액션 형식이 맞지 않아 `action_type`과 `action_param` 추출 실패
- `step()` 호출: `step(env, "search for any known interactions...")`
- `obs`: `"Invalid action: search for any known interactions..."`

### 6. 상태 딕셔너리 업데이트 및 노드 생성

**위치**: `hotpot/lats.py` line 682-687

```python
# 새로운 상태 딕셔너리 업데이트
new_state['thought'] = thought_line
new_state['action'] = action_line
new_state['observation'] = obs

# 새로운 노드 생성
new_node = Node(state=new_state, question=node.question, parent=node)
new_node.is_terminal = r == 1 or done
new_node.reward = r
new_node.depth = node.depth + 1
```

**동작**:
- 부모 노드의 상태를 복사한 `new_state`에 세 가지 요소 저장
- 새로운 `Node` 객체 생성
- 노드의 종료 여부, 보상, 깊이 설정

## 전체 흐름 다이어그램

```
generate_new_states(node)
    │
    ├─> generate_prompt(node)
    │   └─> 노드의 trajectory를 프롬프트로 변환
    │
    ├─> get_samples(task, prompt, ...)
    │   ├─> task.cot_prompt_wrap() 또는 task.standard_prompt_wrap()
    │   ├─> gpt() 호출
    │   └─> LLM이 생성한 Thought + Action 텍스트 리스트 반환
    │
    ├─> 각 sampled_action에 대해:
    │   ├─> thought_line 파싱 (텍스트에서 "Thought X: " 추출)
    │   ├─> action_line 파싱 (텍스트에서 "Action: " 추출)
    │   │
    │   ├─> step(env, action)
    │   │   └─> 환경에서 액션 실행 → observation 획득
    │   │
    │   └─> new_state 생성:
    │       ├─> new_state['thought'] = thought_line
    │       ├─> new_state['action'] = action_line
    │       └─> new_state['observation'] = obs
    │
    └─> 새로운 노드들 반환
```

## 예시: 실제 생성 과정

### 입력 (노드 상태)
```python
node.state = {
    'thought': 'I need to determine whether CRISPRi knockdown of ABCF1...',
    'action': 'Search for any known interactions...',
    'observation': 'Invalid action: search for any known interactions...'
}
node.depth = 1
```

### 1단계: 프롬프트 생성
```
Question: In hepg2 cells, if a CRISPRi knockdown of ABCF1 is done...
Thought 1: I need to determine whether CRISPRi knockdown of ABCF1...
Action: Search for any known interactions...
Observation: Invalid action: search for any known interactions...
```

### 2단계: LLM 응답 생성
```
Thought 2: I need to analyze how CRISPRi knockdown of ABCF1 affects EIF1 expression...
Action: Search[ABCF1 EIF1 interaction]
```

### 3단계: 파싱
- `thought_line`: `"I need to analyze how CRISPRi knockdown of ABCF1 affects EIF1 expression..."`
- `action_line`: `"Search[ABCF1 EIF1 interaction]"`

### 4단계: 환경 실행
- `step(env, "search[ABCF1 EIF1 interaction]")`
- `obs`: `"ABCF1 and EIF1 are both involved in translation initiation..."`

### 5단계: 새 상태 생성
```python
new_state = {
    'thought': 'I need to analyze how CRISPRi knockdown of ABCF1 affects EIF1 expression...',
    'action': 'Search[ABCF1 EIF1 interaction]',
    'observation': 'ABCF1 and EIF1 are both involved in translation initiation...'
}
```

## 주의사항

1. **액션 형식**: 
   - 올바른 형식: `"Search[entity]"`, `"Lookup[entity]"` 등
   - 잘못된 형식: `"Search for any known interactions..."` (자연어)
   - 잘못된 형식의 경우 환경에서 "Invalid action" 오류 반환

2. **중복 제거**:
   - `unique_key = f"{thought_line}::{action_line}"`로 고유 키 생성
   - 동일한 Thought와 Action 조합은 한 번만 생성

3. **종료 조건**:
   - `r == 1`: 정답을 찾은 경우
   - `done == True`: 에피소드가 종료된 경우
   - 종료된 노드는 `is_terminal = True`로 설정

4. **실패한 궤적 기록**:
   - `is_terminal == True`이고 `r == 0`인 경우
   - `failed_trajectories`에 추가되어 자가 반성(self-reflection)에 활용

