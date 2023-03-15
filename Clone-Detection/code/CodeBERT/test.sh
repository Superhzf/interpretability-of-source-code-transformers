#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=56G
#SBATCH --time=00:10:00
EVAL_DATA="/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Clone-Detection/dataset/valid.txt"
TEST_DATA="/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/Clone-Detection/dataset/test.txt"
MODEL="microsoft/codebert-base"
OUTPUT_DIR="./saved_models/CodeBERT"

DATA=$1

if [ "$DATA" = "train" ]; then

          EXTRACT_DATA="$TRAIN_DATA"

elif [ "$DATA" = "dev" ]; then
	  
	  EXTRACT_DATA="$EVAL_DATA"

elif [ "$DATA" = "test" ]; then

          EXTRACT_DATA="$TEST_DATA"
fi


echo {$OUTPUT_DIR}/{$DATA}_test.log

echo "{$EXTRACT_DATA} $DATA"
