# Interpretability of source code transformers

## Create ml-gpu environment NeuroX_env using the following steps:
```
srun --time=00:10:00 --nodes=1 --cpus-per-task=4 --partition=gpu --gres=gpu:1 --pty /usr/bin/bash  
module load ml-gpu  
mkdir NeuroX_env  
ml-gpu python -m venv <path to environment>/NeuroX_env  
cd NeuroX_env
```
## Install Neurox from source (pip install version is not updated with Selectivity/Control task code)
```
git clone https://github.com/fdalvi/NeuroX.git  
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install -e .  
```   

## Steps to generate POS dataset from text file containing python code  
We use the python tokenizer to tokenize the code and obtain POS labels.   

1. This will generate codetest.in and codetest.label but neurox skips some lines, not sure why.  
```
ml-gpu /work/LAS/jannesar-lab/arushi/Environments/NeuroX_env/bin/python POS_dataset_creation.py
```
TODO: Figure out why neurox skips lines in skip_logs  

2. Extract activations, identify and remove skipped lines form dataset for now -- codetest2.in and codetest2.label  
```
ml-gpu /work/LAS/jannesar-lab/arushi/Environments/NeuroX_env/bin/python get_skip_lines.py > skip_logs
```
TODO: Remove additional stuff from skip logs  

3. From skip_logs, get the lines to skip from datset and create codetest2.in and codetest2.label  
```
ml-gpu /work/LAS/jannesar-lab/arushi/Environments/NeuroX_env/bin/python skip_lines.py
```

