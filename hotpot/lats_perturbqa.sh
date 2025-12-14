#!/bin/bash
# Example script for running LATS with perturbQA dataset

# Set environment variables for perturbQA
export DATASET_TYPE=perturbqa
export PERTURBQA_DATA_DIR=/home/work/khm/MoA_finetune/data/perturbqa/sorted_genes/unsloth/DeepSeek-R1-Distill-Llama-8B/hepg2/methodA/temp_0.6

# Local model name (first argument overrides default)
LOCAL_MODEL_NAME="${1:-unsloth/DeepSeek-R1-Distill-Llama-8B}"
export LOCAL_MODEL_NAME

python run.py \
    --backend local \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/lats_perturbqa_local.log \
    --algorithm lats \
    --dataset_type perturbqa \
    --perturbqa_data_dir ${PERTURBQA_DATA_DIR} \
    --local_model_name "${LOCAL_MODEL_NAME}" \
    ${@:2}


"""
python run.py \
    --backend local \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/lats_perturbqa_local.log \
    --algorithm lats \
    --dataset_type perturbqa \
    --perturbqa_data_dir /home/work/khm/MoA_finetune/data/perturbqa/sorted_genes/unsloth/DeepSeek-R1-Distill-Llama-8B/hepg2/methodA/temp_0.6 \
    --local_model_name unsloth/DeepSeek-R1-Distill-Llama-8B 
"""