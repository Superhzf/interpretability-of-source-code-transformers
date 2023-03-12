#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=1-02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load ml-gpu

#cd /work/LAS/jannesar-lab/arushi/Redundancy/Code_data/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/code

TRAIN_DATA="/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Clone-Detection/dataset/train.txt"
EVAL_DATA="/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Clone-Detection/dataset/valid.txt"
TEST_DATA="/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Clone-Detection/dataset/test.txt"
MODEL="CAUKiel/JavaBERT"
OUTPUT_DIR="./saved_models/JavaBERT"

DATA=$1

if [ "$DATA" = "train" ]; then

          EXTRACT_DATA="$TRAIN_DATA"

elif [ "$DATA" = "dev" ]; then

          EXTRACT_DATA="$EVAL_DATA"

elif [ "$DATA" = "test" ]; then

          EXTRACT_DATA="$TEST_DATA"
fi


cd /work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Clone-Detection/code/CodeBERT

ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python run.py \
    --output_dir=./saved_models/JavaBERT \
    --model_type=bert \
    --output_file=./saved_models/JavaBERT/train_activations.json \
    --config_name=$MODEL \
    --model_name_or_path=$MODEL \
    --tokenizer_name=bert-base-cased \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=$TRAIN_DATA \
    --eval_data_file=$EVAL_DATA \
    --test_data_file=$TEST_DATA \
    --extract_data_file=$EXTRACT_DATA \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --evaluate_during_training \
    --max_grad_norm 1.0 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1| tee ./saved_models/JavaBERT/train.log



