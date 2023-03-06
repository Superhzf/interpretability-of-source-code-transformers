#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=10-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load ml-gpu

cd /work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Defect-detection/code/CodeGPT/Finetuning

ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python run_extraction.py \
    --output_dir=./saved_models_python/original \
    --model_type=gpt2 \
    --output_file=train_activations.json \
    --config_name=microsoft/CodeGPT-small-py \
    --model_name_or_path=microsoft/CodeGPT-small-py \
    --tokenizer_name=microsoft/CodeGPT-small-py \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/train.jsonl \
    --eval_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/valid.jsonl \
    --test_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --evaluate_during_training \
    --max_grad_norm 1.0 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1| tee saved_models_python/original/train_python_original.log
