# HotPotQA LATS 실행 흐름 정리

## 1. 진입점: `hotpot/lats.sh`

```bash
python run.py \
    --backend gpt-3.5-turbo \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/tot_10k.log \
    ${@}
```

**주요 파라미터:**
- `--backend`: 사용할 GPT 모델 (gpt-3.5-turbo)
- `--task_start_index`: 시작 문제 인덱스 (0)
- `--task_end_index`: 종료 문제 인덱스 (100)
- `--n_generate_sample`: 확장 시 생성할 샘플 수 (5)
- `--n_evaluate_sample`: 평가 시 생성할 샘플 수 (1)
- `--prompt_sample`: 프롬프트 타입 (cot = Chain of Thought)
- `--iterations`: 최대 반복 횟수 (30)
- `--log`: 로그 파일 경로

**참고:** `--algorithm lats` 파라미터가 스크립트에 없지만, `${@}`를 통해 전달되거나 `run.py`에서 기본값으로 처리될 수 있습니다.

---

## 2. `hotpot/run.py` - 메인 실행 로직

### 2.1 `parse_args()` - 인자 파싱
- 명령줄 인자를 파싱하여 `args` 객체 생성
- `--algorithm` 파라미터가 없으면 오류 발생 가능

### 2.2 `run(args)` - 메인 실행 함수

**초기화:**
1. `HotPotQATask()` 객체 생성
2. 로그 디렉토리 생성 (`logs/` 디렉토리)
3. 로깅 설정 (파일 로깅)
4. `task_accs` 리스트 초기화 (정확도 추적용)

**메인 루프 (각 문제마다):**
```python
for i in range(args.task_start_index, args.task_end_index):
    # LATS 알고리즘 실행
    state, value, all_nodes, reward, em = lats_search(args, task, i, args.iterations, True)
    
    # 정확도 기록 및 평균 계산
    task_accs.append(em)
    cnt_avg = sum(task_accs) / len(task_accs)
```

**종료:**
- GPT 사용량 출력 (`gpt_usage()`)

---

## 3. `hotpot/lats.py` - LATS 알고리즘 구현

### 3.1 `lats_search()` - LATS 검색 메인 함수

**초기화:**
1. GPT 함수 설정 (`partial(gpt, model=args.backend, temperature=args.temperature)`)
2. 환경 초기화: `env.reset(idx=idx)` - HotPotQA 문제 로드
3. 루트 노드 생성: `Node(state=None, question=x)`
4. 전역 변수 초기화:
   - `failed_trajectories`: 실패한 궤적 저장
   - `reflection_map`: 자기 반성 맵핑
   - `terminal_nodes`: 터미널 노드 리스트

**메인 반복 루프 (최대 `iterations`번):**

```python
for i in range(iterations):
    1. select_node(root) - UCT 기반 노드 선택
    2. 터미널 노드 체크 (reward == 1이면 즉시 반환)
    3. expand_node(node) - 노드 확장 (자식 노드 생성)
    4. evaluate_node(node) - 노드 평가 (자식 노드들의 가치 평가)
    5. rollout() - 시뮬레이션 (최고 가치 자식 노드로 롤아웃)
    6. backpropagate() - 역전파 (가치를 부모 노드로 전파)
    7. 성공 노드 체크 (reward == 1인 터미널 노드 찾기)
```

**종료 조건:**
- `reward == 1`인 터미널 노드 발견 시 즉시 반환
- 모든 반복 완료 시 최고 노드 반환

---

### 3.2 핵심 함수들

#### `select_node(node)` - UCT 기반 노드 선택
- **목적:** 탐험과 활용의 균형을 맞춘 노드 선택
- **알고리즘:**
  1. 루트에서 시작하여 자식 노드 중 UCT 값이 가장 높은 노드 선택
  2. UCT 공식: `value/visits + sqrt(2 * log(parent.visits) / visits)`
  3. 모든 자식이 터미널이면 백트래킹
  4. `reward == 1`인 터미널 노드 발견 시 즉시 반환

#### `expand_node(node, args, task)` - 노드 확장
- **목적:** 현재 노드에서 가능한 다음 상태들을 생성
- **프로세스:**
  1. 깊이 제한 체크 (depth >= 7이면 터미널로 표시)
  2. `generate_new_states()` 호출:
     - `get_samples()`: GPT로 다음 액션 샘플링 (n_generate_sample개)
     - 각 샘플에 대해:
       - Thought와 Action 파싱
       - 환경에 액션 실행: `step(env, action)`
       - 새로운 노드 생성 및 상태 업데이트
       - 터미널 여부 및 reward 설정
     - 중복 상태 제거

#### `evaluate_node(node, args, task)` - 노드 평가
- **목적:** 자식 노드들의 가치를 평가하여 우선순위 결정
- **프로세스:**
  1. 각 자식 노드에 대한 프롬프트 생성
  2. `get_values()` 호출:
     - `get_value()`: GPT로 각 자식 노드의 가치 평가
     - `value_prompt_wrap()`: 평가 프롬프트 생성 (실패한 궤적 포함 가능)
     - `value_outputs_unwrap()`: GPT 출력을 0.1~1.0 점수로 변환
  3. 각 자식 노드의 `value` 속성 업데이트
  4. 평균 가치 반환

#### `rollout(node, args, task, idx, max_depth=4)` - 시뮬레이션
- **목적:** 선택된 노드에서 깊이 제한까지 빠르게 시뮬레이션
- **프로세스:**
  1. 최대 깊이(max_depth=4)까지 반복:
     - `generate_new_states()`: 다음 상태들 생성
     - `get_values()`: 각 상태 평가
     - 최고 가치 상태 선택하여 계속 진행
  2. 터미널 노드 도달 시 즉시 반환
  3. 깊이 제한 도달 시 reward = -1
  4. 평균 reward 반환

#### `backpropagate(node, value)` - 역전파
- **목적:** 시뮬레이션 결과를 부모 노드들로 전파
- **프로세스:**
  1. 현재 노드부터 루트까지 역순으로:
     - `visits` 증가
     - `value` 업데이트: `(value * (visits-1) + new_value) / visits`
  2. 터미널 노드의 경우 reward 값 반영

#### `generate_new_states(node, args, task, n)` - 새 상태 생성
- **프로세스:**
  1. `generate_prompt(node)`: 현재 노드의 궤적을 프롬프트로 변환
  2. `get_samples()`: GPT로 다음 Thought와 Action 샘플링
  3. 각 샘플에 대해:
   - Thought와 Action 파싱
   - 환경에 액션 실행 (`step(env, action)`)
   - 새 노드 생성 및 상태 업데이트
   - 터미널 여부 및 reward 설정
  4. 중복 상태 제거 후 반환

---

## 4. `hotpot/hotpotqa.py` - HotPotQA 태스크 정의

### 주요 메서드:

#### `cot_prompt_wrap(x, y, reflection_mapping_list)` - CoT 프롬프트 생성
- Chain of Thought 프롬프트 생성
- 실패한 궤적과 반성이 있으면 피드백 포함

#### `value_prompt_wrap(x, y, z, reflections)` - 평가 프롬프트 생성
- 노드 가치 평가를 위한 프롬프트 생성
- 실패한 궤적과 반성을 포함하여 더 나은 평가 제공

#### `generate_self_reflection(z, question)` - 자기 반성 생성
- 실패한 궤적들을 분석하여 반성 생성
- GPT를 사용하여 각 궤적의 문제점 파악

---

## 5. 환경 및 래퍼

### `wikienv.WikiEnv()` - 위키 환경
- HotPotQA 문제를 위한 위키 검색 환경

### `wrappers.HotPotQAWrapper()` - HotPotQA 래퍼
- 환경을 HotPotQA 태스크에 맞게 래핑

### `wrappers.LoggingWrapper()` - 로깅 래퍼
- 환경 액션과 관찰을 로깅

---

## 6. 전체 실행 흐름 요약

```
lats.sh 실행
    ↓
run.py: parse_args() → 인자 파싱
    ↓
run.py: run() → HotPotQATask 초기화
    ↓
[각 문제마다 반복]
    ↓
lats.py: lats_search() → 환경 초기화, 루트 노드 생성
    ↓
[최대 iterations번 반복]
    ↓
    1. select_node() → UCT 기반 노드 선택
    2. expand_node() → GPT로 자식 노드 생성
    3. evaluate_node() → GPT로 자식 노드 평가
    4. rollout() → 시뮬레이션 실행
    5. backpropagate() → 가치 역전파
    ↓
[성공 노드 발견 시 즉시 반환]
    ↓
run.py: 정확도 기록 및 평균 계산
    ↓
모든 문제 완료 후 GPT 사용량 출력
```

---

## 7. 주요 데이터 구조

### `Node` 클래스
- `state`: {'thought', 'action', 'observation'} 딕셔너리
- `parent`: 부모 노드
- `question`: 원본 질문
- `children`: 자식 노드 리스트
- `visits`: 방문 횟수
- `value`: 노드 가치
- `depth`: 트리 깊이
- `is_terminal`: 터미널 노드 여부
- `reward`: 보상 (0 또는 1)
- `em`: Exact Match 점수

### 전역 변수
- `failed_trajectories`: 실패한 궤적 리스트
- `reflection_map`: 자기 반성 맵핑 리스트

---

## 8. GPT 호출 지점

1. **상태 생성 (`get_samples`)**: 다음 Thought와 Action 생성
2. **노드 평가 (`get_value`)**: 노드의 가치 평가
3. **자기 반성 (`generate_self_reflection`)**: 실패한 궤적 분석

각 호출은 `models.py`의 `gpt()` 함수를 통해 OpenAI API를 호출합니다.

