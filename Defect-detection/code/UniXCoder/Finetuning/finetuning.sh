#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=1-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load ml-gpu

cd /work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Defect-detection/code/UniXCoder/Finetuning

ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/pip3 install -U transformers

ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --output_file=valid_activations.json \
    --config_name=microsoft/unixcoder-base \
    --model_name_or_path=microsoft/unixcoder-base \
    --tokenizer_name=microsoft/unixcoder-base \
    --do_extract \
    --train_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/train.jsonl \
    --eval_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/valid.jsonl \
    --test_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/test.jsonl \
    --extract_data_file=/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/example.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1| tee valid_extraction.log
    
