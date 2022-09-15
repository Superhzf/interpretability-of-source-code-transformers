#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=10-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-pcie:1
module load ml-gpu

cd /work/LAS/jannesar-lab/arushi/Redundancy/Code_data/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/code

ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --output_file=./train_activations.json \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_extract \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --extract_data_file=../dataset/example.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1| tee train_extraction.log



def
