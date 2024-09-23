#!/bin/csh
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH -c4
#SBATCH --output=wandb_privacy_dp_chest_xray_%A.out
#SBATCH --job-name=wandb_privacy_dp_chest_xray_%A
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=noam.bensason@mail.huji.ac.il
#SBATCH --gres=gpu:1,vmem:24G
#SBATCH --killable


cd /cs/labs/tomhope/noam_bs97/advanced_privacy_project
module load cuda/11.8
source /cs/labs/tomhope/noam_bs97/miniconda3/etc/profile.d/conda.csh
conda activate privacy_env

python wandb_dp_chest_xray.py --model_name vit