#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=6-02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load ml-gpu


MODEL_SETTINGS=$1
if [ "$MODEL_SETTINGS" = "CodeBERT" ]; then

          MODEL="microsoft/codebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/CodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "GraphCodeBERT" ]; then

          MODEL="microsoft/graphcodebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/GraphCodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeBERTa" ]; then

          MODEL="huggingface/CodeBERTa-small-v1"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/CodeBERTa"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "UniXCoder" ]; then

          MODEL="microsoft/unixcoder-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/UniXCoder"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "RoBERTa" ]; then

          MODEL="roberta-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/RoBERTa"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "BERT" ]; then

          MODEL="bert-base-uncased"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./code/saved_models/BERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "JavaBERT" ]; then

          MODEL="CAUKiel/JavaBERT"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./code/saved_models/JavaBERT"
          TOKENIZER="bert-base_cased"

elif [ "$MODEL_SETTINGS" = "GPT2" ]; then

          MODEL="gpt2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/GPT2"
          TOKENIZER=$MODEL #need to add padding token 
elif [ "$MODEL_SETTINGS" = "CodeGPT-java" ]; then

          MODEL="microsoft/CodeGPT-small-java"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/java-original"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-python" ]; then

          MODEL="microsoft/CodeGPT-small-py"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/python-original"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-java-adapted" ]; then

          MODEL="microsoft/CodeGPT-small-java-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/java-adapted"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-python-adapted" ]; then

          MODEL="microsoft/CodeGPT-small-java-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/python-adapted"
          TOKENIZER=$MODEL

fi



cd /work/LAS/cjquinn-lab/zefuh/selectivity/Interpretability/interpretability-of-source-code-transformers/NL-Code-Search/NL-code-search-WebQuery
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python code/run_classifier.py \
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
                        --output_dir ./code/saved_models/CodeBERT/model_codesearchnet \
                        --encoder_name_or_path microsoft/codebert-base \
			--tokenizer_name microsoft/codebert-base | tee code/saved_models/CodeBERT/train.log

ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python code/run_classifier.py \
			--model_type roberta \
			--do_train \
			--do_eval \
			--eval_all_checkpoints \
			--train_file cosqa_train.json \
			--dev_file cosqa_dev.json \
			--max_seq_length 200 \
			--per_gpu_train_batch_size 16 \
			--per_gpu_eval_batch_size 16 \
			--learning_rate 1e-5 \
			--num_train_epochs 3 \
			--gradient_accumulation_steps 1 \
			--warmup_steps 5000 \
			--evaluate_during_training \
			--data_dir ./data/ \
			--output_dir ./code/saved_models/CodeBERT/model_cosqa_continue_training \
			--encoder_name_or_path ./code/saved_models/CodeBERT/model_codesearchnet
                        --tokenizer_name microsoft/codebert-base | tee ./code/saved_models/CodeBERT/continue_train.log




