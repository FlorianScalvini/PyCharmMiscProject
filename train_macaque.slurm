#!/bin/bash
#SBATCH --job-name=joint_seg
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=./output/joint_seg_%A_%a.out
#SBATCH --error=./error/joint_seg_%A_%a.err
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-0

set -euo pipefail

module purge
module load arch/h100
module load pytorch-gpu

PROJECT_DIR=/lustre/fswork/projects/rech/crp/ulq73oz/IABM2025/PyCharmMiscProject
cd "$PROJECT_DIR"

lambda_reg_values=(0.5)
lambda_jac_values=(10)

lambda_reg=${lambda_reg_values[$SLURM_ARRAY_TASK_ID]}
lambda_jac=${lambda_jac_values[$SLURM_ARRAY_TASK_ID]}

set -x
srun python -u src/train.py \
    --dataset "$PROJECT_DIR/data/adni.yaml" \
    --lambda_reg "$lambda_reg" \
    --lambda_jac "$lambda_jac" \
    --lambda_seg 1 \
    --lambda_sim 0.0 \
    --max_epochs 5000 \
    --num_workers "$SLURM_CPUS_PER_TASK" \
    --num_sanity_val_steps 0 \
    --val_check_interval 100
