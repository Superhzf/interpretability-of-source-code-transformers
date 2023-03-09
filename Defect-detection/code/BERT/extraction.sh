#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=10-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

TRAIN_DATA="/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/train.jsonl"
EVAL_DATA="/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/valid.jsonl"
TEST_DATA="/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/test.jsonl"
MODEL="roberta-base"
OUTPUT_DIR="./saved_models/RoBERTa/"

#When extracting activations run sbatch extraction.sh train, extraction.sh dev or extraction.sh test. Add --do_extract argument while running puthon file below.
DATA=$2
if [DATA=="train"]
then
	EXTRACT_DATA=TRAIN_DATA
elif [DATA=="dev"]
then
	EXTRACT_DATA=EVAL_DATA
elif [DATA=="test"]
then
	EXTRACT_DATA=TEST_DATA
fi
module load ml-gpu

cd /work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Defect-detection/code/BERT

ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python run_extraction.py \
    --output_dir=.$OUTPUT_DIR \
    --model_type=roberta \
    --output_file=$OUTPUT_DIR/{$DATA}_activations.json \
    --config_name=$MODEL \
    --model_name_or_path=$MODEL \
    --tokenizer_name=$MODEL \
    --do_extract \
    --train_data_file=$TRAIN_DATA \
    --eval_data_file=$EVAL_DATA \
    --test_data_file=$TEST_DATA \
    --extract_data_file=$EXTRACT_DATA \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1| tee {$OUTPUT_DIR}/{$DATA}_extract.log





