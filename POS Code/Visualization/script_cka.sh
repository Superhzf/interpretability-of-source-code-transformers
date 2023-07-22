#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=speedy

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/selectivity/interpretability-of-source-code-transformers/POS\ Code/Visualization
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python cka.py --language java > log_cka