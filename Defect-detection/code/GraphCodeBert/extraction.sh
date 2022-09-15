#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1

module load ml-gpu
cd /work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/code/GraphCodeBert

CUDA_LAUNCH_BLOCKING=1 ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python run.py \
    --output_file=train_activations.json \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_extract \
    --train_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/train.jsonl \
    --eval_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/valid.jsonl \
    --test_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/test.jsonl \
    --extract_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/example.jsonl \
    --eval_batch_size=32 \
    --train_batch_size=32 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only  \
    --seed 123456  2>&1 | tee train_extraction.log

