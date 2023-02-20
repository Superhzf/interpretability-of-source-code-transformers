#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000:1

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/selectivity/interpretability-of-source-code-transformers/POS\ Code/Experiments
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python run_neurox1.py --extract=True> log
