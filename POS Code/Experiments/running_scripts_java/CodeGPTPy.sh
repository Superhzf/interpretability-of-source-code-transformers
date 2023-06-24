#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=72G
#SBATCH --time=24:00:00
#SBATCH --partition=speedy

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/selectivity/interpretability-of-source-code-transformers/POS\ Code/Experiments
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python run_neurox1.py --extract=False --this_model pretrained_codeGPTPy --language java > ./running_scripts_java/CodeGPTPy.log
 