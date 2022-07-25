# Interpretability of source code transformers

## Create ml-gpu environment finetuning_env using the following steps:
```
srun --time=00:10:00 --nodes=1 --cpus-per-task=4 --partition=gpu --gres=gpu:1 --pty /usr/bin/bash  
module load ml-gpu  
mkdir finetuning_env  
ml-gpu python -m venv <path to environment>/finetuning_env  
cd finetuning_env 
```
## Install transformers from source 
```
ml-gpu <path to environment>/finetuning_env/bin/pip3pip install git+https://github.com/huggingface/transformers  
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install --force --extra-index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio 
```   
(For CUDA 11.6)


## Steps to get extractions from Codebert Defect-detection finetuned model
 

1. Run run_extraction.py -- update paths(model and data) in jobscript as needed.
```
sbatch jobscript
```


