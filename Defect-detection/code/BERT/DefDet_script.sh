#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=1-02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

TRAIN_DATA="/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/train.jsonl"
EVAL_DATA="/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/valid.jsonl"
TEST_DATA="/work/LAS/jannesar-lab/arushi/Redundancy/Defect-detection/dataset/test.jsonl"


DATA=$1

if [ "$DATA" = "train" ]; then

          EXTRACT_DATA="$TRAIN_DATA"

elif [ "$DATA" = "dev" ]; then

          EXTRACT_DATA="$EVAL_DATA"

elif [ "$DATA" = "test" ]; then

          EXTRACT_DATA="$TEST_DATA"
fi

TASK=$2

if [ "$TASK" = "finetune" ]; then

          DO="do_train --do_eval --do_test"
elif [ "$TASK" = "extract" ]; then

          DO="do_extract"

fi

MODEL_SETTINGS=$3
if [ "$MODEL_SETTINGS" = "CodeBERT" ]; then

          MODEL="microsoft/codebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/CodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "GraphCodeBERT" ]; then

          MODEL="microsoft/graphcodebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/GraphCodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeBERTa" ]; then

          MODEL="huggingface/CodeBERTa-small-v1"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/CodeBERTa"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "UniXCoder" ]; then

          MODEL="microsoft/unixcoder-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/UniXCoder"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "RoBERTa" ]; then

          MODEL="roberta-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/RoBERTa"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "BERT" ]; then

          MODEL="bert-base-uncased"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./saved_models/BERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "JavaBERT" ]; then

          MODEL="CAUKiel/JavaBERT"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./saved_models/JavaBERT"
          TOKENIZER="bert-base_cased"

elif [ "$MODEL_SETTINGS" = "GPT2" ]; then

          MODEL="gpt2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/GPT2"
          TOKENIZER=$MODEL #need to add padding token 
elif [ "$MODEL_SETTINGS" = "CodeGPT-java" ]; then

          MODEL="microsoft/CodeGPT-small-java"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/java-original"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-python" ]; then

          MODEL="microsoft/CodeGPT-small-py"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/python-original"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-java-adapted" ]; then

          MODEL="microsoft/CodeGPT-small-java-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/java-adapted"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-python-adapted" ]; then

          MODEL="microsoft/CodeGPT-small-java-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/python-adapted"
          TOKENIZER=$MODEL

fi

module load ml-gpu

cd /work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Defect-detection/code/BERT

ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python run_extraction.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$MODEL_TYPE \
    --output_file=$OUTPUT_DIR/${DATA}_activations.json \
    --config_name=$MODEL \
    --model_name_or_path=$MODEL \
    --tokenizer_name=$TOKENIZER \
    --train_data_file=$TRAIN_DATA \
    --eval_data_file=$EVAL_DATA \
    --test_data_file=$TEST_DATA \
    --$DO \
    --extract_data_file=$EXTRACT_DATA \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --evaluate_during_training \
    --max_grad_norm 1.0 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1| tee $OUTPUT_DIR/${DATA}_${TASK}.log



