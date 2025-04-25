#!/bin/bash
#SBATCH --job-name=IFTQwen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --time=12:00:00

/home/shu4/.conda/envs/cs336_alignment_test/bin/python /home/shu4/koa_scratch/s2025-assignment3-alignment/scripts/train.py