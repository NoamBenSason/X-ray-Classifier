#!/bin/csh
#SBATCH --time=5:00:00
#SBATCH --mem=128GB
#SBATCH --output=privacy_mnist_example_%A.out
#SBATCH --job-name=privacy_mnist_example_%A
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=noam.bensason@mail.huji.ac.il
#SBATCH --gres=gpu:1,vmem:32G

cd /cs/labs/tomhope/noam_bs97/advanced_privacy_project
module load cuda/11.8
source /cs/labs/tomhope/noam_bs97/miniconda3/etc/profile.d/conda.csh
conda activate privacy_env

python dp_chest_xray.py --device=cuda -n=10 --lr=0.001 --sigma=1.3 -c=1.5 -b=16
