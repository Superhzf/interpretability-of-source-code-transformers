# Interpretability of source code transformers

## Create ml-gpu environment using the following steps
```
srun --time=00:10:00 --nodes=1 --cpus-per-task=4 --partition=gpu --gres=gpu:1 --pty /usr/bin/bash  
module load ml-gpu  
mkdir NeuroX_env  
ml-gpu python -m venv <path to environment>/NeuroX_env  
cd NeuroX_env
```
## Install PyTorch

The default Neurox installation is not compatible with the most advanced GPU on
pronto. Therefore, we have to install PyTorch first. The required PyTorch version
is 1.8.1
```
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Install Neurox from source (pip install version is not updated with Control task code)
```
git clone https://github.com/fdalvi/NeuroX.git  
cd NeuroX
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install -e .  
```
## Run experiments
Update script with correct paths to your environment.  
Set --extract=False if activation files have already been generated.  
Location of activation.json files:
/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/POS\ Code/Experiments  
```
cd interpretability-of-source-code-transformers/POS\ Code/Experiments
sbatch script
```
