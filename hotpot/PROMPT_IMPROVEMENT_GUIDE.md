# 프롬프트 개선 가이드

## 현재 프롬프트 흐름

### 1. 초기 프롬프트 생성 (`perturbqa.py`)

**위치**: `PerturbQATask._build_full_prompt()` (line 280-335)

**생성되는 프롬프트 구조**:
```
<|SYSTEM|>
{system_query}
<|END_SYSTEM|>
<|USER|>
Cell line: {cell_line}
{cell_info}

Domain knowledge for knockdown gene {perturb_gene}:
- {pert_descriptions}

Domain knowledge for target gene {target_gene}:
- {target_descriptions}

Interaction Metrics:
- STRINGdb score: {score}
- GO semantic similarity: {go_similarity}

Question: {question}
<|END_USER|>
<|ASSISTANT|>
```

### 2. Trajectory 추가 (`lats.py`)

**위치**: `generate_prompt()` (line 781-808)

**동작**:
- 루트 노드부터 현재 노드까지의 모든 `thought`, `action`, `observation`을 수집
- 형식:
  ```
  {question}
  Thought 1: {thought}
  Action 1: {action}
  Observation 1: {observation}
  Thought 2: {thought}
  ...
  ```

### 3. 프롬프트 래핑 (`perturbqa.py`)

**위치**: `PerturbQATask.cot_prompt_wrap()` (line 197-220)

**이전 방식 (HotPotQA용)**:
```python
prompt = cot_prompt.format(input=input)  # HotPotQA용 프롬프트
```

**문제점**:
- Wikipedia Search/Lookup 액션에 맞춰져 있음
- Thought가 간단함 (1-2 문장)
- PerturbQA의 생물학적 도메인 지식을 활용하지 못함
- 액션 형식이 명확하지 않음

**개선된 방식 (PerturbQA 전용)**:
```python
prompt = cot_prompt_perturbqa.format(input=input)  # PerturbQA 전용 프롬프트
```

### 4. LLM 호출 (`lats.py`)

**위치**: `get_samples()` (line 200-205)

```python
samples = gpt(prompt, n=n_generate_sample, stop="Observation", max_tokens=2000)
```

## 개선 사항

### 1. PerturbQA 전용 프롬프트 생성

**파일**: `perturbqa_prompts.py`

**주요 개선점**:

#### a) Thought 생성 가이드라인 강화
- **이전**: 간단한 1-2 문장
- **개선**: 최소 2-3 문장, 상세한 생물학적 추론 요구
  - 세포주 특성 분석
  - 유전자 기능 및 상호작용 고려
  - 분자 메커니즘 평가
  - 논리적 예측 근거 제시

#### b) Action 형식 명확화
- **이전**: `Search[entity]`, `Lookup[keyword]`, `Finish[answer]` (Wikipedia용)
- **개선**: 
  - `Search[gene_name or pathway_name]` - 특정 유전자/경로 정보 검색
  - `Analyze[relationship_type]` - 특정 관계 유형 분석
  - `Finish[upregulated|downregulated|uncertain]` - 최종 답변

#### c) 도메인 지식 활용 안내
- 제공된 도메인 지식을 적극 활용하도록 명시
- 특정 기능, 상호작용, 경로를 참조하도록 요구
- 여러 메커니즘 고려 (직접 상호작용, 경로 효과, 보상 메커니즘)

#### d) 상세한 예시 제공
- 실제 PerturbQA 질문에 대한 완전한 예시
- 각 Thought가 어떻게 상세하게 작성되어야 하는지 보여줌
- 생물학적 추론 과정을 단계별로 설명

### 2. 프롬프트 구조 비교

#### 이전 (HotPotQA용)
```
Solve a question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Search[entity]
(2) Lookup[keyword]
(3) Finish[answer]

Example:
Thought 1: I need to search Colorado orogeny...
Action 1: Search[Colorado orogeny]
```

#### 개선 (PerturbQA 전용)
```
You are an expert molecular and cellular biology researcher...
Your Thought should be DETAILED and BIOLOGICALLY INFORMED:
1. Analyzes the biological context
2. Considers the relationship between genes
3. Evaluates available evidence
4. Makes logical predictions based on mechanisms
5. Should be comprehensive (at least 2-3 sentences)

Example:
Thought 1: ABCF1 is involved in translation initiation and is part of the 43S 
pre-initiation complex. EIF1 is also a component of the same 43S complex...
[상세한 생물학적 추론 2-3 문장]
Action 1: Search[ABCF1 EIF1 interaction]
```

### 3. 사용 방법

**자동 적용**: `PerturbQATask.cot_prompt_wrap()`이 자동으로 PerturbQA 전용 프롬프트를 사용합니다.

**수동 확인**: 
```python
# lats.py의 get_samples()에서
prompt = task.cot_prompt_wrap(x, y, reflection_map)
# 이제 cot_prompt_perturbqa가 사용됨
```

## 예상 효과

1. **Thought 품질 향상**:
   - 더 상세하고 생물학적으로 정확한 추론
   - 도메인 지식 적극 활용
   - 메커니즘 기반 논리적 추론

2. **Action 품질 향상**:
   - 더 구체적이고 관련성 높은 액션
   - PerturbQA 도메인에 맞는 액션 형식

3. **Observation 활용**:
   - 더 나은 액션으로 인한 더 유용한 관찰 결과

## 디버깅

프롬프트가 제대로 적용되었는지 확인:

1. **`lats.py` line 200**: `pdb.set_trace()`에서 `prompt` 변수 확인
   - `cot_prompt_perturbqa`의 내용이 포함되어 있는지 확인
   - "You are an expert molecular and cellular biology researcher"로 시작하는지 확인

2. **생성된 Thought 확인**:
   - 최소 2-3 문장인지 확인
   - 생물학적 추론이 포함되어 있는지 확인
   - 도메인 지식을 참조하고 있는지 확인

3. **생성된 Action 확인**:
   - `Search[...]`, `Analyze[...]`, `Finish[...]` 형식인지 확인
   - 자연어가 아닌 구조화된 형식인지 확인

