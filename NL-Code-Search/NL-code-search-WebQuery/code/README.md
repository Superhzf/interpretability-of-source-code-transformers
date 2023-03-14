
Follow instructions to get data here : https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery

mkdir saved\_models
cd saved\_models
mkdir CodeBERT GraphCodeBERT UniXCoder RoBERTa CodeBERTa BERT

cd ../../../code
sbatch finetuning2.sh

