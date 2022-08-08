#!/bin/bash
#BATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100-pcie:1

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/selectivity/interpretability-of-source-code-transformers/POS\ Code/test
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python test.py
