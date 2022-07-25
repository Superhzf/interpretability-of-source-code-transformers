# Interpretability of source code transformers

## Create ml-gpu environment aux_env using the following steps:
```
srun --time=00:10:00 --nodes=1 --cpus-per-task=4 --partition=gpu --gres=gpu:1 --pty /usr/bin/bash  
module load ml-gpu  
mkdir aux_env  
ml-gpu python -m venv <path to environment>/aux_env  
cd aux_env 
```
## Install transformers==2.0.0 to run the redundancy experiments on extracted activations
```
ml-gpu <path to environment>/aux_env/bin/pip3 install transformers==2.0.0
```   

## Update paths in config_codebert to run experiments

