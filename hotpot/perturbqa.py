import os
import re
import sys
from pathlib import Path
from base import Task
from hotpot import *
from models import gpt as _gpt_base
import lats

# 전역 gpt 함수 - lats.py에서 partial로 설정된 함수를 사용하기 위해
# lats.py에서 설정한 전역 gpt를 사용하도록 함
# lats_search가 실행되면 lats.gpt가 설정되므로 이를 사용
gpt = _gpt_base

# Import PerturbQA-specific prompts
try:
    from perturbqa_prompts import (
        cot_prompt_perturbqa,
        cot_prompt_perturbqa_short,
        cot_prompt_perturbqa_feedback,
        cot_prompt_perturbqa_feedback_short
    )
except ImportError:
    # Fallback to hotpot prompts if perturbqa_prompts not available
    # These are defined in hotpot.py via "from hotpot import *"
    cot_prompt_perturbqa = cot_prompt  # type: ignore
    cot_prompt_perturbqa_short = cot_prompt_short  # type: ignore
    cot_prompt_perturbqa_feedback = cot_prompt_feedback  # type: ignore
    cot_prompt_perturbqa_feedback_short = cot_prompt_feedback_short  # type: ignore
import logging
from transformers import GPT2Tokenizer
import random
import json
import pandas as pd
from typing import Dict, List, Tuple

# Add MoA_finetune to path to import data loading utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'MoA_finetune'))
try:
    from utils.progressive_reasoning.data_loader import (
        load_sorted_genes_data,
        build_perturbqa_dataframe
    )
    from utils.progressive_reasoning.nodes import (
        get_formatted_history,
        get_cell_info_cached,
        process_tool_info_to_descriptions,
        get_go_similarity_cached,
        get_kg_loader,
    )
    from utils.prompt import return_prompt as prompt_utils_return_prompt
    from utils.kg_loader import KGLoader
    
    # Override get_kg_loader to use the correct path
    # Set the kg_dir to the absolute path where ensembl.json is located
    _kg_loader_instance = None
    _kg_dir_path = "/home/work/khm/MoA_finetune/data/kg"
    
    def get_kg_loader_override():
        """Override get_kg_loader to use the correct kg_dir path"""
        global _kg_loader_instance
        if _kg_loader_instance is None:
            _kg_loader_instance = KGLoader(kg_dir=_kg_dir_path, use_uniprot=True)
        return _kg_loader_instance
    
    # Replace the original get_kg_loader with our override
    import utils.progressive_reasoning.nodes as nodes_module
    nodes_module.get_kg_loader = get_kg_loader_override
    
except ImportError:
    print("Warning: Could not import perturbQA data loading utilities. Please ensure MoA_finetune is in the correct location.")
    load_sorted_genes_data = None
    build_perturbqa_dataframe = None
    get_formatted_history = None
    get_cell_info_cached = None
    process_tool_info_to_descriptions = None
    get_go_similarity_cached = None
    prompt_utils_return_prompt = None

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def get_token_length(text):
    return len(tokenizer.encode(text))

max_token_length = 4000

class PerturbQATask(Task):
    """
    Input (x)   : a text instruction (perturbQA question)
    Output (y)  : a text generation
    Reward (r)  : correctness score
    """
    def __init__(self, sorted_genes_dir=None):
        """
        sorted_genes_dir: path to sorted_genes directory containing perturbQA data
        """
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}
        
        # Load perturbQA data
        if sorted_genes_dir is None:
            # Try to get from environment variable or use default
            sorted_genes_dir = os.getenv('PERTURBQA_DATA_DIR', None)
            if sorted_genes_dir is None:
                raise ValueError("sorted_genes_dir must be provided or set PERTURBQA_DATA_DIR environment variable")
        
        # Extract cell line from path
        # Path structure: .../sorted_genes/.../model_name/cell_line/method/param
        # Example: data/perturbqa/sorted_genes/unsloth/DeepSeek-R1-Distill-Llama-8B/hepg2/methodA/temp_0.6
        # Cell line is the directory name before method (second to last directory)
        sorted_genes_path = Path(sorted_genes_dir).resolve()
        path_parts = sorted_genes_path.parts
        
        # Find cell line: it's typically the directory before the last two (method and param)
        # Or we can look for it after model name patterns
        cell_line = "unknown"
        if len(path_parts) >= 3:
            # Try to find cell line by looking at the structure
            # Usually: .../model_name/cell_line/method/param
            # So cell_line is at index -3 (third from last)
            cell_line = path_parts[-3]
        elif len(path_parts) >= 2:
            # Fallback: use second to last
            cell_line = path_parts[-2]
        
        # Store cell line for use in _build_full_prompt
        self.cell_line = cell_line
        
        if load_sorted_genes_data is None or build_perturbqa_dataframe is None:
            raise ImportError("Could not import perturbQA data loading utilities")
        
        # Load data
        pert_data = load_sorted_genes_data(sorted_genes_dir)
        perturbqa_data, perturbqa_data_indexed = build_perturbqa_dataframe(pert_data)
        
        # Convert to question-answer format for LATS
        # Each perturbation gene becomes a question
        self.data = []
        for pert_gene, records in pert_data.items():
            # Build a full prompt similar to scientist_reasoning_node
            question = self._build_full_prompt(pert_gene, records)
            # DEBUG: 프롬프트 생성 확인
            # 확인할 값: question (생성된 전체 프롬프트), pert_gene, records
            # Answer is the list of affected genes (sorted by score)
            affected_genes = [r.get('gene', '') for r in records if r.get('gene')]
            answer = ', '.join(affected_genes[:10])  # Top 10 genes as answer
            self.data.append((question, answer, pert_gene, records))
        
        self.pert_data = pert_data
        self.perturbqa_data = perturbqa_data
        self.perturbqa_data_indexed = perturbqa_data_indexed

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx][0]  # Return question
    
    def get_pert_gene(self, idx: int) -> str:
        """Get the perturbation gene for a given index"""
        return self.data[idx][2]
    
    def get_records(self, idx: int) -> List[Dict]:
        """Get the records for a given index"""
        return self.data[idx][3]
    
    def test_output(self, idx: int, output: str):
        # lats.py에서 설정한 전역 gpt 함수를 사용
        # lats_search가 실행되면 lats.gpt가 partial로 감싸져서 설정됨
        # lats 모듈의 전역 gpt 사용 (lats_search에서 설정된 model, temperature 등이 자동 적용됨)
        global_gpt = getattr(lats, 'gpt', _gpt_base)  # lats 모듈의 전역 gpt 사용, 없으면 기본 gpt 사용
        
        output = output.split('Action:\n')[-1]
        prompt = score_prompt + output
        # max_tokens를 충분히 크게 설정하여 평가 응답이 잘리지 않도록 함
        # 전역 gpt 함수 사용 (lats.py에서 설정한 model, temperature 등이 자동 적용됨)
        score_outputs = global_gpt(prompt, n=5, max_tokens=500)
        scores = []
        for score_output in score_outputs:
            pattern = r".*correctness score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def generate_self_reflection(z, question):
        # lats.py에서 설정한 전역 gpt 함수를 사용
        # lats_search가 실행되면 lats.gpt가 partial로 감싸져서 설정됨
        # lats 모듈의 전역 gpt 사용 (lats_search에서 설정된 model, temperature 등이 자동 적용됨)
        global_gpt = getattr(lats, 'gpt', _gpt_base)  # lats 모듈의 전역 gpt 사용, 없으면 기본 gpt 사용
        
        reflection_mapping = []
        trajectories = ""

        sampled_items = random.sample(z, min(3, len(z)))
        failed_trajectories = "\n".join([f"{question}\n{traj}\n" for traj in z])
        failed_trajectories = [f"Question: {traj}" for traj in failed_trajectories.split("Question: ")[1:]]
        
        for traj in failed_trajectories:
            trajectories += traj
            
            reflect_prompt = reflection_prompt.format(trajectory=traj)
            
            # max_tokens를 충분히 크게 설정하여 반성 응답이 잘리지 않도록 함
            # 전역 gpt 함수 사용 (lats.py에서 설정한 model, temperature 등이 자동 적용됨)
            reflection = global_gpt(reflect_prompt, max_tokens=1000)
            
            trajectories += "Reflection: " + reflection[0] + "\n"
            
            reflection_mapping.append({
                'question': question,
                'trajectory': traj,
                'reflection': reflection[0]
            })

        return reflection_mapping

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[]):
        """
        PerturbQA 전용 CoT 프롬프트 래핑
        
        HotPotQA용 프롬프트 대신 PerturbQA 전용 프롬프트를 사용하여
        더 상세하고 생물학적으로 정확한 Thought와 Action 생성을 유도합니다.
        """
        question = x
        input = x + y
        trajectories = ""
        if reflection_mapping_list:
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                trajectories += traj_with_reflection
            
            # PerturbQA 전용 feedback 프롬프트 사용
            prompt = cot_prompt_perturbqa_feedback.format(trajectories=trajectories, input=input)
            if get_token_length(prompt) > max_token_length:
                print("Too long, using short version")
                trajectories = ""
                for reflection_mapping in reflection_mapping_list[:3]:
                    traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                    trajectories += traj_with_reflection
                prompt = cot_prompt_perturbqa_feedback_short.format(trajectories=trajectories, input=input)
            
            return prompt
        else:
            # PerturbQA 전용 CoT 프롬프트 사용
            prompt = cot_prompt_perturbqa.format(input=input)
            if get_token_length(prompt) > max_token_length:
                prompt = cot_prompt_perturbqa_short.format(input=input)
            return prompt
    
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt + "\n" + x + "\n\n"
        for i, y in enumerate(ys, 1):
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best trajectory is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        
        # Extract the last Action for each trajectory
        last_actions = []
        for y in ys:
            lines = y.split('\n')[::-1]
            for line in lines:
                if "Action" in line:
                    last_actions.append(line.split('Action')[-1].strip(': '))
                    break

        assert len(last_actions) == 2, 'Expected to find 2 Actions'

        prompt = compare_prompt + f'Action 1:{last_actions[0]}\n\nAction 2:{last_actions[1]}\n'
        return prompt

    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more correct trajectory is 1' in compare_output:
            return 0
        elif 'more correct trajectory is 2' in compare_output:
            return 1
        elif "two trajectories are similarly correct" in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str, z: list = [], reflections: list = []) -> str:
        question = x.split('\n')[0]
        if len(z) != 0:
            failed_trajectories = ""
            
            # Combine the trajectories with their corresponding reflections
            for traj, ref in zip(z, reflections):
                failed_trajectories += f"{question}\n{traj}\nThis trajectory is incorrect as {ref['reflection']}\nThus the correctness score is 1\n"
            
            inp = x + y + "\nThis trajectory is "
            
            prompt = value_prompt_reasoning_feedback.format(s="", trajectories=failed_trajectories, input=inp)
            
            if get_token_length(prompt) > max_token_length:
                prompt = value_prompt_reasoning_feedback_short.format(s="", trajectories=failed_trajectories, input=inp)
        else:
            inp = y + "\nThis trajectory is "
            prompt = value_prompt_reasoning.format(s="", input=inp)
            
        return prompt

    
    @staticmethod
    def value_outputs_unwrap(evaluate_prompt: str):
        evaluate_prompt = evaluate_prompt[0]
        if '10' in evaluate_prompt:
            return 1.0
        elif '9' in evaluate_prompt:
            return 0.9
        elif '8' in evaluate_prompt:
            return 0.8
        elif '7' in evaluate_prompt:
            return 0.7
        elif '6' in evaluate_prompt:
            return 0.6
        elif '5' in evaluate_prompt:
            return 0.5
        elif '4' in evaluate_prompt:
            return 0.4
        elif '3' in evaluate_prompt:
            return 0.3
        elif '2' in evaluate_prompt:
            return 0.2
        elif '1' in evaluate_prompt:
            return 0.1
        else:
            return -1

    def _build_full_prompt(self, perturb_gene: str, records: List[Dict]) -> str:
        """
        Construct a rich prompt inspired by scientist_reasoning_node.full_prompt.
        Falls back to a simple question if prompt utilities or tools are unavailable.
        """
        try:
            if prompt_utils_return_prompt is None or process_tool_info_to_descriptions is None:
                raise ImportError("Prompt utilities unavailable")

            # Select a target gene candidate from records (first entry as proxy)
            top_record = records[0] if records else {}
            target_gene = top_record.get("gene", "UNKNOWN")
            score = float(top_record.get("score", 0.0))
            
            # Get cell line from instance variable (extracted from path in __init__)
            # Fallback to record if not available (for backward compatibility)
            if hasattr(self, 'cell_line') and self.cell_line != "unknown":
                cell_line = self.cell_line
            else:
                cell_line = top_record.get("cell_line", top_record.get("cell", "unknown"))

            history_str = ""

            # Tool data - handle missing files gracefully
            try:
                pert_descriptions = process_tool_info_to_descriptions(perturb_gene, max_items=20)
            except (FileNotFoundError, IOError, OSError) as e:
                logging.warning(f"Could not load tool descriptions for {perturb_gene}: {e}")
                pert_descriptions = []
            except Exception as e:
                logging.warning(f"Unexpected error loading tool descriptions for {perturb_gene}: {e}")
                pert_descriptions = []
            
            try:
                target_descriptions = process_tool_info_to_descriptions(target_gene, max_items=20)
            except (FileNotFoundError, IOError, OSError) as e:
                logging.warning(f"Could not load tool descriptions for {target_gene}: {e}")
                target_descriptions = []
            except Exception as e:
                logging.warning(f"Unexpected error loading tool descriptions for {target_gene}: {e}")
                target_descriptions = []
            
            try:
                go_similarity = (
                    get_go_similarity_cached(perturb_gene, target_gene)
                    if get_go_similarity_cached
                    else 0.0
                )
            except (FileNotFoundError, IOError, OSError) as e:
                logging.warning(f"Could not load GO similarity for {perturb_gene}-{target_gene}: {e}")
                go_similarity = 0.0
            except Exception as e:
                logging.warning(f"Unexpected error loading GO similarity: {e}")
                go_similarity = 0.0

            # Get cell line information (cached to avoid repeated API calls)
            cell_info = ""
            try:
                if get_cell_info_cached:
                    cell_info = get_cell_info_cached(cell_line)
                    if cell_info:
                        cell_info = f"\n{cell_info}\n"
            except Exception as e:
                logging.warning(f"Could not load cell line information for {cell_line}: {e}")
                cell_info = ""

            pert_entries = "\n".join([f"- {desc}" for desc in pert_descriptions]) or "- (no data)"
            target_entries = "\n".join([f"- {desc}" for desc in target_descriptions]) or "- (no data)"
            print("cell_line", cell_line)

            prompt_context = (
                f"{history_str}\n\n"
                f"Cell line: {cell_line}{cell_info}\n"
                f"Domain knowledge for knockdown gene {perturb_gene}:\n"
                f"{pert_entries}\n\n"
                f"Domain knowledge for target gene {target_gene}:\n"
                f"{target_entries}\n\n"
                f"Interaction Metrics:\n"
                f"- STRINGdb score: {score:.4f}\n"
                f"- GO semantic similarity: {go_similarity:.4f}\n\n"
            )

            system_query = prompt_utils_return_prompt("gene_perturb_system_prompt")
            user_prompt_template = prompt_utils_return_prompt("gene_perturb_user_prompt")
            user_query = user_prompt_template.format(cell_line=cell_line, pert=perturb_gene, gene=target_gene)

            user_content = f"{prompt_context}\n{user_query}"

            full_prompt = (
                f"<|SYSTEM|>\n{system_query}\n<|END_SYSTEM|>\n"
                f"<|USER|>\n{user_content}\n<|END_USER|>\n<|ASSISTANT|>\n"
            )
            return full_prompt
        except Exception as e:
            logging.warning(f"Falling back to simple question prompt for {perturb_gene}: {e}")
            return f"What genes are affected when {perturb_gene} is perturbed?"

