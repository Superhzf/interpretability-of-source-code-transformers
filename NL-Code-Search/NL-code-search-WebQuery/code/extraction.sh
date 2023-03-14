#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=3-02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load ml-gpu


cd /work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/NL-Code-Search/NL-code-search-WebQuery

python code/run_classifier.py \
			--model_type roberta \
			--do_train \
			--do_eval \
			--eval_all_checkpoints \
			--train_file train_codesearchnet_7.json \
			--dev_file dev_codesearchnet.json \
			--max_seq_length 200 \
			--per_gpu_train_batch_size 16 \
			--per_gpu_eval_batch_size 16 \
			--learning_rate 1e-5 \
			--num_train_epochs 3 \
			--gradient_accumulation_steps 1 \
			--warmup_steps 1000 \
			--evaluate_during_training \
			--data_dir ./data/ \
			--output_dir ./model_codesearchnet \
			--encoder_name_or_path microsoft/codebert-base| tee ./code/train.log
