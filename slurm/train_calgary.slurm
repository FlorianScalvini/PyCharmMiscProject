#!/bin/bash
#SBATCH --job-name=Calgary
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=100:00:00
#SBATCH --output=./output/calgary_%A_%a.out
#SBATCH --error=./error/calgary_%A_%a.err
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --array=0-0

set -euo pipefail

module purge
module load arch/h100
module load pytorch-gpu

PROJECT_DIR=/lustre/fswork/projects/rech/crp/ulq73oz/IABM2025/PyCharmMiscProject/
cd "$PROJECT_DIR"

lambda_reg_values=(10)
lambda_jac_values=(1000)

lambda_reg=${lambda_reg_values[$SLURM_ARRAY_TASK_ID]}
lambda_jac=${lambda_jac_values[$SLURM_ARRAY_TASK_ID]}

set -x
srun python -u src/train.py \
    --dataset "$PROJECT_DIR/data/calgary.yaml" \
    --lambda_reg "$lambda_reg" \
    --lambda_jac "$lambda_jac" \
    --lambda_seg 0 \
    --lambda_sim 1 \
    --max_epochs 5000 \
    --devices 4 \
    --num_nodes 1 \
    --num_workers 4 \
    --num_sanity_val_steps 0 \
    --val_check_interval 100
