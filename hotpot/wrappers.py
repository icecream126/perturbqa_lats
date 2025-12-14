import json
import os
import gym
import numpy as np
import re
import string
from collections import Counter

    
DATA_DIR = "data"
HOTPOTQA_SPLIT_FILE = {
  "train": "hotpot_train_v1.1_simplified.json",
  "dev": "hotpot_dev_v1_simplified.json",
  "test": "hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
  "train": "train.jsonl",
  "dev": "paper_dev.jsonl",
}


class HistoryWrapper(gym.ObservationWrapper):
  def __init__(self, env, obs_format, prompt=None):
    super().__init__(env)
    assert obs_format in ["obs", "history"]
    if obs_format == "history":
      assert hasattr(self.env, "traj")
    self.obs_format = obs_format
    self.prompt = prompt if prompt is not None else ""

  def observation(self, obs):
    if self.obs_format == "obs":
      return obs
    elif self.obs_format == "history":
      observation = self.env.traj["observations"][0] + "\n"
      for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
        observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
      return self.prompt + observation
    

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  
  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall
  
class HotPotQAWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    data_file = f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE[split]}"
    self.data = json.load(open(data_file))
    self.data = [(d['question'], d['answer']) for d in self.data]
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Question: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "hotpot_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      score = (pred == gt)
      return int(score)
    return 0
  
  def get_metrics(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      em = (pred == gt)
      f1 = f1_score(pred, gt)[0]
      return {'reward': em, 'em': em, 'f1': f1}
    return {'reward': 0, 'em': 0, 'f1': 0}

  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update(self.get_metrics(info))
    return obs, reward, done, info
  
  def __len__(self):
    return len(self.data)

class PerturbQAWrapper(gym.Wrapper):
  def __init__(self, env, sorted_genes_dir=None):
    super().__init__(env)
    
    # Import perturbQA data loading
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'MoA_finetune'))
    try:
      from utils.progressive_reasoning.data_loader import load_sorted_genes_data
    except ImportError:
      raise ImportError("Could not import perturbQA data loading utilities")
    
    if sorted_genes_dir is None:
      sorted_genes_dir = os.getenv('PERTURBQA_DATA_DIR', None)
      if sorted_genes_dir is None:
        raise ValueError("sorted_genes_dir must be provided or set PERTURBQA_DATA_DIR environment variable")
    
    # Load perturbQA data
    pert_data = load_sorted_genes_data(sorted_genes_dir)
    
    # Convert to question-answer format
    self.data = []
    for pert_gene, records in pert_data.items():
      question = f"What genes are affected when {pert_gene} is perturbed?"
      affected_genes = [r.get('gene', '') for r in records if r.get('gene')]
      answer = ', '.join(affected_genes[:10])  # Top 10 genes as answer
      self.data.append((question, answer, pert_gene, records))
    
    self.data_idx = 0
    self.pert_data = pert_data

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Question: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0],
      "pert_gene": self.data[self.data_idx][2],
      "perturbqa_split": "train"
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(info['answer'])
      gt = normalize_answer(self.data[self.data_idx][1])
      # For perturbQA, we check if predicted genes match ground truth genes
      pred_genes = set([g.strip() for g in pred.split(',')])
      gt_genes = set([g.strip() for g in gt.split(',')])
      # Calculate F1-like score based on overlap
      if len(gt_genes) == 0:
        return 0
      intersection = pred_genes & gt_genes
      if len(intersection) == 0:
        return 0
      precision = len(intersection) / len(pred_genes) if len(pred_genes) > 0 else 0
      recall = len(intersection) / len(gt_genes)
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
      # Return 1 if F1 > 0.5, else 0
      return int(f1 > 0.5)
    return 0
  
  def get_metrics(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(info['answer'])
      gt = normalize_answer(self.data[self.data_idx][1])
      pred_genes = set([g.strip() for g in pred.split(',')])
      gt_genes = set([g.strip() for g in gt.split(',')])
      intersection = pred_genes & gt_genes
      if len(gt_genes) == 0:
        return {'reward': 0, 'em': 0, 'f1': 0}
      precision = len(intersection) / len(pred_genes) if len(pred_genes) > 0 else 0
      recall = len(intersection) / len(gt_genes)
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
      em = int(pred_genes == gt_genes)
      reward = int(f1 > 0.5)
      return {'reward': reward, 'em': em, 'f1': f1}
    return {'reward': 0, 'em': 0, 'f1': 0}

  def step(self, action):
    # PerturbQA는 Search, Analyze, Finish 액션을 사용
    # wikienv는 search, lookup, finish, think만 지원하므로 Analyze를 처리해야 함
    action = action.strip()
    
    # Analyze 액션 처리 (wikienv에서 지원하지 않음)
    if action.startswith("analyze[") and action.endswith("]"):
      # Analyze 액션은 정보 분석을 의미하므로, 관찰만 반환하고 done=False 유지
      param = action[len("analyze["):-1]
      obs = f"Analyzing {param}: Based on the provided domain knowledge, this relationship involves complex biological mechanisms that require careful consideration of the available evidence."
      done = False
      reward = 0
      info = self._get_info()
      return obs, reward, done, info
    
    # Finish 액션이 아닌 경우 done=False로 강제 설정
    # (wikienv가 finish 액션에 대해 done=True를 반환하지만, 
    #  PerturbQA에서는 Finish 액션만 종료를 의미하므로 다른 액션은 계속 진행 가능)
    is_finish_action = action.startswith("finish[") and action.endswith("]")
    
    obs, _, done, info = self.env.step(action)
    
    # Finish 액션이 아닌 경우 done=False로 강제 설정
    if not is_finish_action:
      done = False
    
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update(self.get_metrics(info))
    return obs, reward, done, info
  
  def __len__(self):
    return len(self.data)

class FeverWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    
    data_path = f"./data/{FEVER_SPLIT_FILE[split]}"
    with open(data_path, "r") as json_file:
      json_list = list(json_file)

    data = []
    for json_str in json_list:
      json_str = json.loads(json_str)
      label = json_str["label"]
      claim = json_str["claim"]
      data.append((claim, label))

    self.data = data
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Claim: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "fever_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      label = normalize_answer(self.data[self.data_idx][1])
      pred = normalize_answer(info['answer'])
      if label == pred:
        return 1
    return 0

  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update({'em': reward, 'reward': reward, 'f1': reward})
    return obs, reward, done, info
    
  def __len__(self):
    return len(self.data)
  
  
class LoggingWrapper(gym.Wrapper):
  def __init__(self, env, folder="trajs", file_id=None):
    super().__init__(env)
    self.trajs = []
    self.traj = {"observations": [], "actions": []}
    self.folder = folder
    self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
    self.file_path = f"{self.folder}/{self.file_id}.json"
    os.makedirs("trajs", exist_ok=True)

  def __len__(self):
    return len(self.env.data)
  

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=idx)
    observation = output[0] if return_info else output
    self.traj = {"observations": [observation], "actions": []}
    return output

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.traj["observations"].append(obs)
    self.traj["actions"].append(action)
    if done:
      self.traj.update(info)
    return obs, reward, done, info

  def update_record(self):
    if len(self.traj) > 0:
      self.trajs.append(self.traj)
      self.traj = {"observations": [], "actions": []}
  
  def write(self):
    self.update_record()
    with open(self.file_path, "w") as f:
      json.dump(self.trajs, f)
      print(f"Saved trajs to trajs/{self.file_id}.json")
    
  def close(self):
    self.write()
