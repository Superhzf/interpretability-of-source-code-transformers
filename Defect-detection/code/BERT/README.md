
To run finetuning or extraction sbatch extraction1.sh <dataset-train,dev,test> <task-extract,finetune> <mode> e.g. sbatch extraction1.sh train extract GraphCodeBERT. All model settings are provided in the script.

mkdir saved\_models
cd saved\_models
mkdir CodeBERT GraphCodeBERT ....(Directory for every model)


###Details of models and tokenizers
1. CodeBERT
          MODEL="microsoft/codebert-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/CodeBERT"
          TOKENIZER=$MODEL

2. GraphCodeBERT

          MODEL="microsoft/graphcodebert-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/GraphCodeBERT"
          TOKENIZER=$MODEL

3. CodeBERTa"

          MODEL="huggingface/CodeBERTa-small-v1"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/CodeBERTa"
          TOKENIZER=$MODEL

4. UniXCoder

          MODEL="microsoft/unixcoder-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/UniXCoder"
          TOKENIZER=$MODEL

5. RoBERTa

          MODEL="roberta-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/RoBERTa"
          TOKENIZER=$MODEL

6. BERT

          MODEL="bert-base-uncased"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./saved_models/BERT"
          TOKENIZER=$MODEL

7. JavaBERT

          MODEL="CAUKiel/JavaBERT"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./saved_models/JavaBERT"
          TOKENIZER="bert-base_cased"

8. GPT2

          MODEL="gpt2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/GPT2"
          TOKENIZER=$MODEL #need to add padding token

9. CodeGPT-java

          MODEL="microsoft/CodeGPT-small-java"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/java-original"
          TOKENIZER=$MODEL

10. CodeGPT-python

          MODEL="microsoft/CodeGPT-small-py"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/python-original"
          TOKENIZER=$MODEL

11. CodeGPT-java-adapted

          MODEL="microsoft/CodeGPT-small-java-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/java-adapted"
          TOKENIZER=$MODEL

12. CodeGPT-python-adapted

          MODEL="microsoft/CodeGPT-small-java-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/python-adapted"
          TOKENIZER=$MODEL



