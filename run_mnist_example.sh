#!/bin/csh
#SBATCH --time=5:00:00
#SBATCH --mem=128GB
#SBATCH --output=privacy_mnist_example_%A.out
#SBATCH --job-name=privacy_mnist_example_%A
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=noam.bensason@mail.huji.ac.il
#SBATCH --gres=gpu:1,vmem:10G
#SBATCH --killable

cd /cs/labs/tomhope/noam_bs97/advanced_privacy_project
module load cuda/11.8
source /cs/labs/tomhope/noam_bs97/anaconda3/etc/profile.d/conda.csh
conda activate privacy_env

export CUDA_HOME="/usr/local/nvidia/cuda/11.8"

python mnist_example.py --device=cpu --disable-dp --n=20 --lr=.1 -b=240
