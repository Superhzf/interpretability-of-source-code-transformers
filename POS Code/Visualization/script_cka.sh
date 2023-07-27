#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=76G
#SBATCH --time=10:00:00
#SBATCH --partition=speedy

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/selectivity/interpretability-of-source-code-transformers/POS\ Code/Visualization
# ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python cka.py --task java > log_cka
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python cka.py --task CloneDetection > log_cka_clonedet