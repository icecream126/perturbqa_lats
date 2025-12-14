"""
LATS (Language Agent Tree Search) 알고리즘 구현

이 모듈은 MCTS(Monte Carlo Tree Search) 기반의 언어 에이전트 트리 탐색 알고리즘을 구현합니다.
질문 답변 작업에서 LLM을 사용하여 Thought-Action-Observation 시퀀스를 탐색하고 최적의 답을 찾습니다.

주요 구성 요소:
- Node: 탐색 트리의 노드를 나타내는 클래스
- lats_search: 메인 검색 함수 (Selection, Expansion, Simulation, Backpropagation 단계 수행)
- get_samples: LLM을 사용하여 다음 액션 후보 생성
- get_value/get_values: 노드의 가치 평가
- select_node: UCT를 사용한 노드 선택
- expand_node: 노드 확장
- rollout: 시뮬레이션 수행
- backpropagate: 결과 역전파
"""

import itertools
import numpy as np
from functools import partial
from models import gpt as _gpt_base
import wikienv, wrappers

# 전역 gpt 함수 - lats_search에서 partial로 설정됨
# 다른 모듈(perturbqa.py 등)에서도 접근 가능하도록 모듈 레벨 변수로 설정
gpt = _gpt_base
import requests
import logging
import random
import os

# 환경은 지연 초기화되므로 CLI 인자가 환경 변수를 덮어쓸 수 있습니다.
# 전역 변수로 저장하여 여러 번 초기화되는 것을 방지합니다.
env = None


def _ensure_env(args):
    """
    환경(Environment)을 한 번만 생성합니다. CLI 인자를 우선 사용하고, 없으면 환경 변수를 사용합니다.
    
    이 함수는 싱글톤 패턴을 사용하여 환경을 한 번만 초기화하고 재사용합니다.
    데이터셋 타입에 따라 HotPotQA 또는 PerturbQA 환경을 생성합니다.
    
    Args:
        args: 명령줄 인자 객체 (dataset_type, perturbqa_data_dir 등을 포함)
    
    Returns:
        초기화된 환경 객체 (LoggingWrapper로 감싸진 환경)
    """
    global env
    if env is not None:
        return env

    dataset_type = getattr(args, "dataset_type", None) or os.getenv("DATASET_TYPE", "hotpotqa")
    base_env = wikienv.WikiEnv()

    
    if dataset_type == "perturbqa":
        sorted_genes_dir = getattr(args, "perturbqa_data_dir", None) or os.getenv("PERTURBQA_DATA_DIR")
        if sorted_genes_dir is None:
            raise ValueError(
                "PERTURBQA_DATA_DIR must be provided (env var or --perturbqa_data_dir) for perturbqa dataset"
            )
        base_env = wrappers.PerturbQAWrapper(base_env, sorted_genes_dir=sorted_genes_dir)
    else:
        # Default to HotPotQA for backward compatibility
        base_env = wrappers.HotPotQAWrapper(base_env, split="train")

    env = wrappers.LoggingWrapper(base_env)
    return env

# 전역 변수: 실패한 궤적과 자가 반성 정보를 저장
# - reflection_map: 실패한 궤적에 대한 LLM의 자가 반성 결과를 저장
# - failed_trajectories: 보상이 0인 종료 노드들의 궤적을 저장 (학습에 활용)
global reflection_map
global failed_trajectories
reflection_map = []  # 자가 반성 맵: 실패한 궤적에 대한 반성 정보
failed_trajectories = []  # 실패한 궤적 리스트: {'trajectory': str, 'final_answer': str} 형식

def step(env, action):
    """
    환경에서 액션을 실행합니다. 타임아웃 에러가 발생하면 최대 10번까지 재시도합니다.
    
    네트워크 요청이 실패하거나 타임아웃이 발생할 수 있으므로, 
    안정성을 위해 재시도 로직을 포함합니다.
    
    Args:
        env: 실행할 환경 객체
        action: 실행할 액션 (예: "search[entity]", "lookup[keyword]", "finish[answer]")
    
    Returns:
        (observation, reward, done, info) 튜플
        - observation: 액션 실행 후 관찰된 결과
        - reward: 보상 값 (0 또는 1)
        - done: 에피소드 종료 여부
        - info: 추가 정보 (정답, 평가 메트릭 등)
    """
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    """
    단일 노드(부분 궤적)의 가치를 LLM을 사용하여 평가합니다.
    
    이 함수는 현재까지의 추론 궤적(y)이 얼마나 올바른지 평가합니다.
    실패한 궤적들과 반성(reflection) 정보를 포함하여 더 정확한 평가를 수행합니다.
    캐싱을 통해 동일한 프롬프트에 대한 반복 평가를 방지합니다.
    
    Args:
        task: Task 객체 (value_prompt_wrap, value_outputs_unwrap 메서드 포함)
        x: 원본 질문 또는 프롬프트
        y: 평가할 부분 궤적 (현재까지의 Thought, Action, Observation 시퀀스)
        n_evaluate_sample: LLM으로부터 생성할 평가 샘플 수 (여러 샘플의 평균 사용)
        cache_value: 가치 캐싱 사용 여부 (기본값: True)
    
    Returns:
        float: 노드의 가치 (0.0 ~ 1.0 사이의 값, 또는 -1.0 if 평가 실패)
    """
    global reflection_map
    global failed_trajectories
    
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y, unique_trajectories, reflection_map)
    logging.info(f"Current: {x}")
    logging.info(f"Current: {y}")
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    logging.info(f"VALUE PROMPT: {value_prompt}")
    # max_tokens를 충분히 크게 설정하여 응답이 잘리지 않도록 함
    # 가치 평가는 일반적으로 짧은 응답이지만, 안전을 위해 500 토큰 설정
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None, max_tokens=500)
    logging.info(f"VALUE OUTPUTS: {value_outputs}")
    value = task.value_outputs_unwrap(value_outputs)
    logging.info(f"VALUES: {value}")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    """
    여러 노드(부분 궤적들)의 가치를 일괄 평가합니다.
    
    여러 후보 궤적들에 대해 각각의 가치를 평가하여 반환합니다.
    중복된 궤적에 대해서는 재평가를 피하기 위해 로컬 캐시를 사용합니다.
    
    Args:
        task: Task 객체
        x: 원본 질문 또는 프롬프트
        ys: 평가할 부분 궤적들의 리스트
        n_evaluate_sample: LLM으로부터 생성할 평가 샘플 수
        cache_value: 가치 캐싱 사용 여부
    
    Returns:
        list: 각 궤적에 대한 가치 점수 리스트 (길이는 ys와 동일)
    """
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    """
    LLM을 사용하여 다음 단계의 액션 후보들을 생성합니다.
    
    현재 상태(x, y)에서 다음에 수행할 수 있는 Thought와 Action을 생성합니다.
    실패한 궤적이 있으면 자가 반성(self-reflection)을 생성하여 더 나은 액션을 생성하도록 합니다.
    프롬프트 타입에 따라 standard 또는 chain-of-thought (cot) 방식을 사용합니다.
    
    Args:
        task: Task 객체 (standard_prompt_wrap, cot_prompt_wrap 메서드 포함)
        x: 원본 질문 또는 프롬프트
        y: 현재까지의 궤적 (다음 액션을 생성할 기준점)
        n_generate_sample: 생성할 액션 후보 수
        prompt_sample: 프롬프트 타입 ('standard' 또는 'cot')
        stop: 생성 중단 토큰 (예: "Observation")
    
    Returns:
        list: 생성된 액션 후보들의 리스트 (각각은 y에 이어지는 형태)
    """
    global failed_trajectories
    global reflection_map
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
        print("generating reflections")
        reflection_map = task.generate_self_reflection(unique_trajectories, x)
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # DEBUG: 프롬프트 래핑 확인
    # 확인할 값: x (입력 질문), y (현재 trajectory), prompt (최종 래핑된 프롬프트), prompt_sample
    # import pdb; pdb.set_trace()
    logging.info(f"PROMPT: {prompt}")
    # max_tokens를 충분히 크게 설정하여 Thought와 Action이 완전히 생성되도록 함
    # Thought + Action을 생성하려면 최소 1000-2000 토큰이 필요하므로 2000으로 설정
    # stop 토큰("Observation")이 나타나면 자동으로 중단되므로 안전함
    samples = gpt(prompt, n=n_generate_sample, stop=stop, max_tokens=2000)
    return [y + _ for _ in samples]

def get_unique_trajectories(failed_trajectories, num=5):
    """
    실패한 궤적들 중에서 고유한 것들만 추출합니다.
    
    동일한 최종 답변을 가진 궤적들은 중복으로 간주하여 제거합니다.
    반성(reflection) 생성 시 사용하기 위해 고유한 실패 사례만 선택합니다.
    
    Args:
        failed_trajectories: 실패한 궤적들의 리스트 (각각은 'trajectory'와 'final_answer' 키를 가짐)
        num: 반환할 최대 고유 궤적 수 (기본값: 5)
    
    Returns:
        list: 고유한 궤적들의 텍스트 표현 리스트
    """
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories

class Node:
    """
    탐색 트리의 노드를 나타내는 클래스입니다.
    
    각 노드는 하나의 상태(state)를 나타내며, Thought-Action-Observation 시퀀스의 한 단계를 표현합니다.
    MCTS(Monte Carlo Tree Search) 알고리즘에서 사용되며, UCT(Upper Confidence Bound for Trees) 값을 계산합니다.
    """
    def __init__(self, state, question, parent=None):
        """
        노드를 초기화합니다.
        
        Args:
            state: 노드의 상태 딕셔너리 {'thought': str, 'action': str, 'observation': str}
            question: 원본 질문 또는 프롬프트
            parent: 부모 노드 (None이면 루트 노드)
        """
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []  # 자식 노드들의 리스트
        self.visits = 0  # 이 노드가 방문된 횟수
        self.value = 0  # 노드의 평균 가치
        self.depth = 0 if parent is None else parent.depth + 1  # 트리에서의 깊이
        self.is_terminal = False  # 종료 노드 여부 (답을 찾았거나 실패)
        self.reward = 0  # 보상 값 (0 또는 1)
        self.exhausted = False  # 모든 자식이 종료 노드인지 여부
        self.em = 0  # Exact Match, 평가 메트릭 (정확히 일치하는지 여부)

    def uct(self):
        """
        UCT(Upper Confidence Bound for Trees) 값을 계산합니다.
        
        탐색과 활용(exploration vs exploitation)의 균형을 맞추기 위한 값입니다.
        높은 UCT 값을 가진 노드가 우선적으로 선택됩니다.
        
        Returns:
            float: UCT 값 (visits가 0이면 value를 그대로 반환)
        """
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def __str__(self):
        """
        노드의 문자열 표현을 반환합니다.
        
        디버깅이나 로깅에 사용되는 사람이 읽기 쉬운 형식입니다.
        
        Returns:
            str: 노드의 주요 정보를 포함한 문자열
        """
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        """
        노드를 딕셔너리 형식으로 변환합니다.
        
        직렬화나 저장 목적으로 사용됩니다. 재귀적으로 자식 노드들도 변환합니다.
        
        Returns:
            dict: 노드의 모든 정보를 포함한 딕셔너리
        """
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }
    
def node_trajectory_to_text(node_string):
    """
    노드의 문자열 표현을 사람이 읽기 쉬운 텍스트 형식으로 변환합니다.
    
    노드의 __str__ 메서드로 생성된 문자열을 파싱하여
    "Thought N: ...", "Action N: ...", "Observation N: ..." 형식으로 변환합니다.
    
    Args:
        node_string: 노드의 문자열 표현 (예: "Node(depth=1, thought=..., action=..., observation=...)")
    
    Returns:
        str: 포맷팅된 궤적 텍스트
    """
    lines = node_string.split('\n')
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            thought = line.split(", thought=")[1].split(", action=")[0].strip()
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue
        
        if depth != 0:
            if thought:
                formatted_lines.append(f"Thought {depth}: {thought}")
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    
    return '\n'.join(formatted_lines)

def collect_all_nodes(node):
    """
    주어진 노드부터 시작하여 모든 하위 노드들을 재귀적으로 수집합니다.
    
    트리의 특정 노드를 루트로 하는 서브트리의 모든 노드를 리스트로 반환합니다.
    디버깅이나 통계 수집에 사용됩니다.
    
    Args:
        node: 수집을 시작할 노드
    
    Returns:
        list: 노드와 그 모든 자식 노드들의 리스트
    """
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_all_nodes(child))
    return nodes

def collect_trajectory(node):
    """
    노드에서 루트까지의 전체 궤적을 수집합니다.
    
    특정 노드에서 시작하여 부모 노드를 따라 올라가며 루트까지의 경로를 수집합니다.
    실패한 궤적을 기록하거나 디버깅에 사용됩니다.
    
    Args:
        node: 궤적을 수집할 시작 노드
    
    Returns:
        str: 루트부터 해당 노드까지의 궤적을 나타내는 문자열
    """
    trajectory = []
    while node:
        trajectory.append(str(node))
        node = node.parent
    return '\n'.join(reversed(trajectory))

def lats_search(args, task, idx, iterations=30, to_print=True):
    """
    LATS (Language Agent Tree Search) 알고리즘의 메인 검색 함수입니다.
    
    MCTS 기반의 트리 탐색 알고리즘을 사용하여 질문에 대한 답을 찾습니다.
    각 반복에서 Selection, Expansion, Simulation, Backpropagation 단계를 수행합니다.
    
    알고리즘 흐름:
    1. 루트 노드 생성 (초기 질문)
    2. 반복 (iterations 횟수만큼):
       - Selection: UCT를 사용하여 탐색할 노드 선택
       - Expansion: 선택된 노드를 확장하여 자식 노드 생성
       - Simulation: 시뮬레이션을 통해 노드 가치 추정
       - Backpropagation: 결과를 부모 노드들로 역전파
    3. 최종적으로 가장 좋은 노드 반환
    
    Args:
        args: 명령줄 인자 객체 (backend, temperature, n_generate_sample 등 포함)
        task: Task 객체 (프롬프트 생성, 가치 평가 메서드 포함)
        idx: 데이터셋에서의 인덱스
        iterations: 최대 반복 횟수 (기본값: 30)
        to_print: 결과를 출력할지 여부 (기본값: True)
    
    Returns:
        tuple: (최종 상태, 가치, 모든 노드, 보상, 정확도)
            - 최종 상태: 찾은 답의 상태
            - 가치: 최종 노드의 가치
            - 모든 노드: 탐색 중 생성된 모든 노드
            - 보상: 최종 보상 (0 또는 1)
            - 정확도: Exact Match 점수
    """
    global gpt
    global failed_trajectories
    global reflection_map
    local_env = _ensure_env(args)
    gpt = partial(gpt, model=args.backend, temperature=args.temperature, local_model_name=getattr(args, "local_model_name", None))
    
    # 데이터셋 타입에 따라 입력 처리 방식 결정
    dataset_type = getattr(args, "dataset_type", None) or os.getenv("DATASET_TYPE", "hotpotqa")
    
    if dataset_type == "perturbqa":
        # PerturbQA의 경우: task.get_input()으로 _build_full_prompt()에서 생성한 전체 프롬프트 사용
        # 시스템/사용자/어시스턴트 태그와 컨텍스트가 포함된 완전한 프롬프트를 얻음
        x = task.get_input(idx)
        # 액션 실행을 위해 환경 초기화 필요
        local_env.reset(idx=idx)
    else:
        # HotPotQA의 경우: env.reset()이 "Question: {question}" 형식 반환
        # env.reset()이 이미 환경을 초기화하므로 반환값 사용
        x = local_env.reset(idx=idx)
    
    # import pdb; pdb.set_trace()
    if to_print:
        print(idx, x)
    root = Node(state=None, question=x)  # 루트 노드 생성
    all_nodes = []  # 모든 노드를 저장할 리스트
    failed_trajectories = []  # 실패한 궤적들을 저장할 리스트
    terminal_nodes = []  # 종료 노드들을 저장할 리스트
    reflection_map = []  # 자가 반성 맵 (실패한 궤적에 대한 반성)
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    
    # 초기 상태 로깅
    logging.info("=" * 80)
    logging.info(f"LATS SEARCH STARTED - Question Index: {idx}")
    logging.info("=" * 80)
    logging.info(f"Question Preview: {x[:200]}..." if len(x) > 200 else f"Question: {x}")
    logging.info(f"Total Iterations: {iterations}")
    logging.info(f"Root Node Created: {format_node_detail(root)}")
    
    for i in range(iterations):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f"ITERATION {i + 1}/{iterations}")
        logging.info("=" * 80)
        
        # Selection 단계: UCT를 사용하여 탐색할 노드 선택
        logging.info("─" * 80)
        logging.info("STEP 1: SELECTION")
        logging.info("─" * 80)
        node = select_node(root)

        # 종료 노드이거나 보상이 0인 경우 다시 선택 (백트래킹)
        while node is None or (node.is_terminal and node.reward != 1):
            if node is None:
                logging.warning(f"  ⚠️  Selected node is None, reselecting...")
            elif node.is_terminal and node.reward != 1:
                logging.warning(f"  ⚠️  Terminal node with reward 0 found at depth {node.depth}, reselecting...")
                logging.info(f"  Node details: {format_node_detail(node)}")
            node = select_node(root)
        
        # 모든 경로가 고갈된 경우 검색 종료
        if node is None:
            logging.warning("=" * 80)
            logging.warning("SEARCH TERMINATED: All paths lead to terminal nodes with reward 0")
            logging.warning("=" * 80)
            log_tree_statistics(root)
            break

        # 선택된 노드 상세 정보 로깅
        logging.info(f"  ✓ Selected Node:")
        logging.info(format_node_detail(node))
        
        # 성공한 종료 노드를 찾은 경우 즉시 반환
        if node.is_terminal and node.reward == 1:
            logging.info("=" * 80)
            logging.info(f"🎉 SUCCESS! Terminal node with reward 1 found at iteration {i + 1}")
            logging.info("=" * 80)
            logging.info(f"Final Node: {format_node_detail(node)}")
            log_tree_statistics(root)
            return node.state, node.value, all_nodes, node.reward, node.em
        
        # Expansion 단계: 선택된 노드를 확장하여 자식 노드 생성
        logging.info("─" * 80)
        logging.info("STEP 2: EXPANSION")
        logging.info("─" * 80)
        logging.info(f"  Expanding node at depth {node.depth}...")
        expand_node(node, args, task)
        logging.info(f"  ✓ Expanded: {len(node.children)} new children created")

        # 깊이 제한에 도달했거나 자식이 없는 경우 다시 선택
        while node.is_terminal or not node.children:
            if node.is_terminal:
                logging.warning(f"  ⚠️  Node is terminal, reselecting...")
            elif not node.children:
                logging.warning(f"  ⚠️  Node has no children, reselecting...")
            node = select_node(root)
            expand_node(node, args, task)

        # 자식 노드들 정보 로깅
        logging.info(f"  Children created:")
        for j, child in enumerate(node.children):
            logging.info(f"    Child {j+1}: Depth={child.depth}, Value={child.value:.4f}, Terminal={child.is_terminal}, Reward={child.reward}")
            if child.state.get('thought'):
                thought_short = child.state['thought'][:80] + '...' if len(child.state['thought']) > 80 else child.state['thought']
                logging.info(f"      Thought: {thought_short}")

        # Evaluation 단계: 자식 노드들의 가치 평가
        logging.info("─" * 80)
        logging.info("STEP 3: EVALUATION")
        logging.info("─" * 80)
        logging.info(f"  Evaluating {len(node.children)} children...")
        value = evaluate_node(node, args, task)
        logging.info(f"  ✓ Evaluation complete. Average value: {value:.4f}")
        
        # Simulation 단계: 가장 높은 가치를 가진 자식 노드에서 시뮬레이션 수행
        best_child = max(node.children, key=lambda child: child.value)
        logging.info("─" * 80)
        logging.info("STEP 4: SIMULATION (ROLLOUT)")
        logging.info("─" * 80)
        logging.info(f"  Starting rollout from best child (value={best_child.value:.4f}):")
        logging.info(format_node_detail(best_child))
        reward, terminal_node = rollout(best_child, args, task, idx, max_depth=4)

        terminal_nodes.append(terminal_node)
        
        logging.info(f"  ✓ Rollout complete. Reward: {reward}, Terminal depth: {terminal_node.depth}")
        logging.info(f"  Terminal node: {format_node_detail(terminal_node)}")

        # 시뮬레이션 중 성공한 경로를 찾은 경우 즉시 반환
        if terminal_node.reward == 1:
            logging.info("=" * 80)
            logging.info("🎉 SUCCESS! Successful trajectory found during simulation")
            logging.info("=" * 80)
            log_tree_statistics(root)
            return terminal_node.state, terminal_node.value, [], terminal_node.reward, terminal_node.em

        # Backpropagation 단계: 시뮬레이션 결과를 부모 노드들로 역전파
        logging.info("─" * 80)
        logging.info("STEP 5: BACKPROPAGATION")
        logging.info("─" * 80)
        logging.info(f"  Backpropagating reward {reward} from depth {terminal_node.depth}...")
        backpropagate(terminal_node, reward)
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        # 트리 전체에서 보상이 1인 종료 노드 확인 (성공한 경로가 있는지 체크)
        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logging.info("=" * 80)
            logging.info(f"🎉 SUCCESS! Terminal node with reward 1 found at iteration {i + 1}")
            logging.info("=" * 80)
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            logging.info(f"Best node: {format_node_detail(best_node)}")
            log_tree_statistics(root)
            return best_node.state, best_node.value, all_nodes, best_node.reward, best_node.em
    
        # 반복 종료 시 트리 상태 요약
        logging.info("─" * 80)
        logging.info(f"Iteration {i + 1} Summary:")
        logging.info(f"  Total nodes in tree: {len(all_nodes)}")
        logging.info(f"  Terminal nodes: {len(terminal_nodes)}")
        logging.info(f"  Failed trajectories: {len(failed_trajectories)}")
        
        # 주기적으로 트리 구조 출력 (매 5번째 반복마다)
        if (i + 1) % 5 == 0:
            log_tree_structure(root, max_depth=3)
            log_tree_statistics(root)
    
    # 모든 반복이 끝난 후 최종 결과 선택
    # import pdb; pdb.set_trace()
    logging.info("")
    logging.info("=" * 80)
    logging.info("FINAL RESULT SELECTION")
    logging.info("=" * 80)
    
    all_nodes_list = collect_all_nodes(root)
    all_nodes_list.extend(terminal_nodes)
    
    # 최종 트리 구조 및 통계 출력
    log_tree_structure(root, max_depth=5)
    log_tree_statistics(root)
    
    # 보상이 가장 높은 노드 선택 (보상이 1인 노드가 있으면 그것을, 없으면 가장 높은 보상)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    failed_trajectories = []
    
    logging.info("─" * 80)
    if best_child.reward == 1:
        logging.info("✅ FINAL RESULT: Successful trajectory found")
    else:
        logging.warning("❌ FINAL RESULT: Unsuccessful trajectory found")
    logging.info("─" * 80)
    logging.info(f"Best Node Selected:")
    logging.info(format_node_detail(best_child))
    
    if best_child is None:
        best_child = root
        logging.warning("  ⚠️  Best child was None, using root node")
    
    logging.info("=" * 80)
    logging.info(f"LATS SEARCH COMPLETED - Question Index: {idx}")
    logging.info("=" * 80)
    logging.info("")
    
    return best_child.state, best_child.value, all_nodes, best_child.reward, best_child.em

def select_node(node):
    """
    UCT 값을 사용하여 다음으로 탐색할 노드를 선택합니다.
    
    MCTS 알고리즘의 Selection 단계입니다. 탐색과 활용의 균형을 맞추기 위해
    UCT 값이 가장 높은 노드를 선택합니다. 종료 노드나 모든 자식이 종료인 경우
    백트래킹을 수행합니다.
    
    Args:
        node: 선택을 시작할 노드 (보통 루트 노드)
    
    Returns:
        Node: 선택된 노드 (모든 경로가 고갈되면 None)
    """
    # import pdb; pdb.set_trace()
    while node and node.children:
        logging.info(f"  Selecting from {len(node.children)} children at depth {node.depth}")
        
        # 자식 노드들의 UCT 값 로깅
        for j, child in enumerate(node.children):
            uct_val = child.uct() if child.parent else 0.0
            status = "TERMINAL" if child.is_terminal else "ACTIVE"
            reward_info = f"R:{child.reward}" if child.is_terminal else ""
            logging.info(f"    Child {j+1}: {status} | UCT:{uct_val:.4f} | Value:{child.value:.4f} | Visits:{child.visits} {reward_info}")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]
        
        # 모든 자식이 종료 노드인 경우 백트래킹
        if len(terminal_children) == len(node.children):
            logging.warning(f"  ⚠️  All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        # 보상이 1인 종료 노드가 있으면 즉시 반환 (성공한 경로 발견)
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"  ✓ Found terminal node with reward 1 at depth {node.depth}")
            return node_with_reward_1
        
        # UCT 값이 가장 높은 비종료 자식 노드 선택
        non_terminal_children = [child for child in node.children if not child.is_terminal]
        if non_terminal_children:
            node = max(non_terminal_children, key=lambda child: child.uct())
        else:
            node = None

        # 선택된 노드가 종료 노드이고 보상이 1이 아닌 경우, 다시 선택
        while node and node.is_terminal and node.reward != 1:
            non_terminal_children = [child for child in node.parent.children if not child.is_terminal]
            node = max(non_terminal_children, key=lambda child: child.uct(), default=None) if non_terminal_children else None
            
        logging.info(f"  ✓ Selected node at depth {node.depth} with UCT {node.uct():.4f}")
        logging.info(format_node_detail(node))
        
    return node  # 모든 경로가 고갈되면 None 반환

def expand_node(node, args, task):
    """
    노드를 확장하여 자식 노드들을 생성합니다.
    
    MCTS 알고리즘의 Expansion 단계입니다. 선택된 노드에서 LLM을 사용하여
    다음 가능한 액션들을 생성하고, 각 액션에 대해 새로운 노드를 만듭니다.
    최대 깊이(7)에 도달하면 노드를 종료 노드로 표시합니다.
    
    Args:
        node: 확장할 노드
        args: 명령줄 인자 객체 (n_generate_sample 등 포함)
        task: Task 객체
    """
    # import pdb; pdb.set_trace()
    if node.depth >= 7:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    node.children.extend(new_nodes)

def rollout(node, args, task, idx, max_depth=4):
    """
    시뮬레이션을 통해 노드의 가치를 추정합니다.
    
    MCTS 알고리즘의 Simulation 단계입니다. 선택된 노드에서 시작하여
    탐욕적(greedy) 방식으로 최고 가치의 자식을 선택하며 진행합니다.
    종료 노드에 도달하거나 최대 깊이에 도달할 때까지 시뮬레이션을 수행합니다.
    
    Args:
        node: 시뮬레이션을 시작할 노드
        args: 명령줄 인자 객체
        task: Task 객체
        idx: 데이터 인덱스
        max_depth: 시뮬레이션의 최대 깊이 (기본값: 4)
    
    Returns:
        tuple: (평균 보상, 종료 노드)
            - 평균 보상: 시뮬레이션 중 얻은 보상들의 평균
            - 종료 노드: 시뮬레이션이 종료된 노드
    """
    # import pdb; pdb.set_trace()
    logging.info("ROLLING OUT")
    depth = node.depth
    n = 5
    rewards = [0]
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        logging.info(f"ROLLING OUT {depth}")
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, n)

        for state in new_states:
            if state.is_terminal:
                return state.reward, state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        #new_state = new_state[0]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        max_value_index = values.index(max(values))
        rewards.append(max(values))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            rewards = [-1]
    
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    logging.info(f"  ✓ Rollout finished. Depth reached: {depth}, Average reward: {avg_reward:.4f}")
    logging.info(f"  Final node: {format_node_detail(node)}")
    return avg_reward, node

def generate_new_states(node, args, task, n):
    """
    LLM을 사용하여 현재 노드에서 가능한 새로운 상태들을 생성합니다.
    
    노드의 현재 상태를 프롬프트로 변환하고, LLM을 호출하여 다음 Thought와 Action을 생성합니다.
    각 생성된 액션을 환경에서 실행하여 Observation을 얻고, 새로운 노드를 생성합니다.
    중복된 상태는 제거하여 고유한 노드만 반환합니다.
    
    Args:
        node: 새로운 상태를 생성할 기준 노드
        args: 명령줄 인자 객체 (prompt_sample 등 포함)
        task: Task 객체
        n: 생성할 액션 후보 수
    
    Returns:
        list: 생성된 새로운 노드들의 리스트 (중복 제거됨)
    """
    global failed_trajectories
    # import pdb; pdb.set_trace()
    prompt = generate_prompt(node)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample, stop="Observation")
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    tried_actions = []
    
    unique_states = {}  # 고유한 상태를 저장하기 위한 딕셔너리
    for action in sampled_actions:
        new_state = node.state.copy()  # 부모 노드의 상태를 복사

        # 생성된 액션에서 Thought와 Action 라인 추출
        thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Thought와 Action을 조합하여 고유 키 생성
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # 이미 존재하는 상태는 건너뜀

        tried_actions.append(action_line)
        
        if action_line:
            # 액션 타입과 파라미터 추출 (예: "Search[entity]" -> "Search", "entity")
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            # 환경에서 액션 실행
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")

            # 새로운 상태 딕셔너리 업데이트
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = obs

            # 새로운 노드 생성
            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done  # 보상이 1이거나 에피소드가 끝나면 종료
            new_node.reward = r
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')  # 정확히 일치하는 경우 EM 저장
            unique_states[unique_key] = new_node  # 고유 상태 딕셔너리에 추가
            logging.info(f"  ✓ New Node Created:")
            logging.info(format_node_detail(new_node))
            logging.info(f"  Environment Feedback: {info}")

            # 실패한 궤적 기록 (보상이 0이고 종료된 경우)
            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory(new_node)
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # 고유한 노드들의 리스트 반환


def evaluate_node(node, args, task):
    """
    노드의 자식 노드들을 평가하여 각각에 가치를 할당합니다.
    
    MCTS 알고리즘의 Evaluation 단계입니다. 노드의 모든 비종료 자식 노드들에 대해
    LLM을 사용하여 가치를 평가하고, 각 자식 노드의 value 속성에 할당합니다.
    노드의 전체 가치는 자식들의 가치 평균으로 계산됩니다.
    
    Args:
        node: 평가할 노드 (자식 노드들의 가치를 평가)
        args: 명령줄 인자 객체 (n_evaluate_sample 등 포함)
        task: Task 객체
    
    Returns:
        float: 자식 노드들의 평균 가치
    """
    # # import pdb; pdb.set_trace()
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
    
    logging.info(f"Length of votes: {len(votes)}")
    logging.info(f"Length of node.children: {len(node.children)}")
    
    # votes 리스트를 미리 할당 (종료 노드에 대해서는 0으로 채움)
    votes = votes + [0] * (len(node.children) - len(votes))
    for i, child in enumerate(node.children):
        child.value = votes[i]  # 각 자식 노드에 가치 할당
    
    return sum(votes) / len(votes) if votes else 0  # 평균 가치 반환


def format_node_detail(node):
    """
    노드의 상세 정보를 포맷팅합니다.
    
    Args:
        node: 포맷팅할 노드
    
    Returns:
        str: 노드의 상세 정보 문자열
    """
    if node is None:
        return "None"
    
    thought_preview = (node.state.get('thought', '')[:100] + '...') if len(node.state.get('thought', '')) > 100 else node.state.get('thought', '')
    action_preview = (node.state.get('action', '')[:80] + '...') if len(node.state.get('action', '')) > 80 else node.state.get('action', '')
    obs_preview = (node.state.get('observation', '')[:80] + '...') if len(node.state.get('observation', '')) > 80 else node.state.get('observation', '')
    
    uct_val = node.uct() if node.parent else 0.0
    
    detail = f"""
    ┌─ Node Details ─────────────────────────────────────────────
    │ Depth: {node.depth} | Visits: {node.visits} | Value: {node.value:.4f} | UCT: {uct_val:.4f}
    │ Terminal: {node.is_terminal} | Reward: {node.reward} | EM: {node.em}
    │ Children: {len(node.children)} | Exhausted: {node.exhausted}
    │
    │ Thought: {thought_preview}
    │ Action:  {action_preview}
    │ Obs:     {obs_preview}
    └─────────────────────────────────────────────────────────────"""
    return detail

def log_tree_structure(root, max_depth=5):
    """
    트리 구조를 시각적으로 로깅합니다.
    
    Args:
        root: 루트 노드
        max_depth: 최대 출력 깊이
    """
    def _log_tree_recursive(node, level=0, prefix="", is_last=True):
        if node is None or level > max_depth:
            return
        
        # 현재 노드 정보
        connector = "└── " if is_last else "├── "
        node_info = f"Depth:{node.depth} V:{node.value:.2f} Visits:{node.visits} UCT:{node.uct():.2f}"
        if node.is_terminal:
            node_info += f" [TERMINAL R:{node.reward}]"
        
        logging.info(f"{prefix}{connector}{node_info}")
        
        # Thought/Action 미리보기
        if node.state.get('thought'):
            thought_short = node.state['thought'][:60] + '...' if len(node.state['thought']) > 60 else node.state['thought']
            logging.info(f"{prefix}{'    ' if is_last else '│   '}  └─ Thought: {thought_short}")
        if node.state.get('action'):
            action_short = node.state['action'][:60] + '...' if len(node.state['action']) > 60 else node.state['action']
            logging.info(f"{prefix}{'    ' if is_last else '│   '}  └─ Action: {action_short}")
        
        # 자식 노드들
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            extension = "    " if is_last else "│   "
            _log_tree_recursive(child, level + 1, prefix + extension, is_last_child)
    
    logging.info("=" * 80)
    logging.info("TREE STRUCTURE:")
    logging.info("=" * 80)
    _log_tree_recursive(root, 0, "", True)
    logging.info("=" * 80)

def log_tree_statistics(root):
    """
    트리의 통계 정보를 로깅합니다.
    
    Args:
        root: 루트 노드
    """
    all_nodes = collect_all_nodes(root)
    
    if not all_nodes:
        logging.info("Tree Statistics: No nodes found")
        return
    
    # 깊이별 통계
    depth_stats = {}
    terminal_count = 0
    reward_1_count = 0
    total_visits = 0
    total_value = 0
    
    for node in all_nodes:
        depth = node.depth
        if depth not in depth_stats:
            depth_stats[depth] = {'count': 0, 'avg_value': 0, 'total_visits': 0}
        depth_stats[depth]['count'] += 1
        depth_stats[depth]['total_visits'] += node.visits
        total_visits += node.visits
        total_value += node.value
        
        if node.is_terminal:
            terminal_count += 1
            if node.reward == 1:
                reward_1_count += 1
    
    # 통계 로깅
    logging.info("=" * 80)
    logging.info("TREE STATISTICS:")
    logging.info("=" * 80)
    logging.info(f"Total Nodes: {len(all_nodes)}")
    logging.info(f"Total Visits: {total_visits}")
    logging.info(f"Average Value: {total_value / len(all_nodes):.4f}" if all_nodes else "N/A")
    logging.info(f"Terminal Nodes: {terminal_count} ({reward_1_count} with reward=1)")
    logging.info(f"Max Depth: {max(depth_stats.keys()) if depth_stats else 0}")
    logging.info("")
    logging.info("Depth Distribution:")
    for depth in sorted(depth_stats.keys()):
        stats = depth_stats[depth]
        avg_value = stats['total_visits'] / stats['count'] if stats['count'] > 0 else 0
        logging.info(f"  Depth {depth}: {stats['count']} nodes, {stats['total_visits']} visits, avg_value: {avg_value:.4f}")
    logging.info("=" * 80)

def print_tree(node, level=0):
    """
    트리 구조를 들여쓰기를 사용하여 출력합니다.
    
    디버깅 목적으로 노드와 그 자식들을 계층적으로 출력합니다.
    
    Args:
        node: 출력을 시작할 노드
        level: 현재 깊이 (들여쓰기 레벨)
    """
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)

def backpropagate(node, value):
    """
    시뮬레이션 결과를 부모 노드들로 역전파합니다.
    
    MCTS 알고리즘의 Backpropagation 단계입니다. 시뮬레이션에서 얻은 가치를
    루트 노드까지 올라가며 각 노드의 visits와 value를 업데이트합니다.
    종료 노드의 경우 보상에 따라 다른 방식으로 가치를 업데이트합니다.
    
    Args:
        node: 역전파를 시작할 노드 (보통 시뮬레이션이 종료된 노드)
        value: 역전파할 가치 (시뮬레이션에서 얻은 보상)
    """
    # import pdb; pdb.set_trace()
    while node:
        node.visits += 1  # 방문 횟수 증가
        if node.is_terminal:
            # 종료 노드의 경우: 보상이 0이면 -1을, 1이면 시뮬레이션 가치를 사용
            if node.reward == 0:
                node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                logging.info(f"    Depth {node.depth}: Terminal (reward=0) → value: {node.value:.4f} (visits: {node.visits})")
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logging.info(f"    Depth {node.depth}: Terminal (reward=1) → value: {node.value:.4f} (visits: {node.visits})")
        else:
            # 비종료 노드: 시뮬레이션 가치로 업데이트
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logging.info(f"    Depth {node.depth}: Non-terminal → value: {node.value:.4f} (visits: {node.visits})")

        node = node.parent  # 부모 노드로 이동

def generate_prompt(node):
    """
    노드의 현재 상태를 기반으로 LLM에 전달할 프롬프트를 생성합니다.
    
    노드에서 루트까지의 전체 궤적(Thought, Action, Observation 시퀀스)을
    수집하여 질문과 함께 하나의 프롬프트로 구성합니다.
    이 프롬프트는 LLM이 다음 액션을 생성하거나 현재 상태를 평가하는 데 사용됩니다.
    
    Args:
        node: 프롬프트를 생성할 노드
    
    Returns:
        str: 질문과 궤적을 포함한 완전한 프롬프트
    """
    # import pdb; pdb.set_trace()
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # 루트 노드의 observation은 제외
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))  # 루트부터 현재 노드까지의 순서로