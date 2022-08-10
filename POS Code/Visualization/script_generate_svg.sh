#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-pcie:1

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/selectivity/interpretability-of-source-code-transformers/POS\ Code/Visualization
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python svg_stack-main/svg_stack.py result/bert_45_0_2945.svg result/space.svg result/bert_48_0_2945.svg result/space.svg result/bert_74_0_2945.svg result/space.svg result/bert_826_0_2945.svg> result/bert.svg
