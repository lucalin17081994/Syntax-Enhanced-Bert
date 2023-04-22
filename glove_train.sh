#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=07:00:00

ls
module load 2021
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
pip install --upgrade scikit-learn
pip install --upgrade pandas
pip install wandb






python Glove.py
