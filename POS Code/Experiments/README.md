# Interpretability of source code transformers

##Create ml-gpu environment using the following steps
srun --time=3:00:00 --nodes=1 --cpus-per-task=4 --partition=gpu --gres=gpu:1 --pty /usr/bin/bash
module load ml-gpu
mkdir NeuroX_env
ml-gpu python -m venv /work/LAS/jannesar-lab/arushi/Environments/NeuroX_env
cd NeuroX_env

##Install Neurox from source (pip install version is not updated with Control task code)
git clone https://github.com/fdalvi/NeuroX.git
ml-gpu /work/LAS/jannesar-lab/arushi/Environments/NeuroX_env/bin/pip3 install -e .

##Run run_neurox1.py
Update script with correct paths to your environment
sbatch script

