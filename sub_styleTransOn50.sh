#!/bin/bash
#SBATCH --gres=gpu:4        # Request GPUs
#SBATCH --time=02-00:00:00  # Job time allocation
#SBATCH --mem=40G          # Memory
#SBATCH -c 16               # Number of cores
#SBATCH -J tr_on_50     # Job name
#SBATCH -o log_fit_50.out      # Output file

# Load environment
module load anaconda
# source activate ml
export OMP_NUM_THREADS=1

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME

# Print environment info
which python
python --version
conda info --envs
conda list
pip list
conda env export > environment.yml

# Remember some metadata
echo -e "Run start: "`date` >> ./metadata.txt
echo -e "   Job ID: "$SLURM_JOB_ID >> ./metadata.txt
echo -e "   Job Name: "$SLURM_JOB_NAME >> ./metadata.txt
echo -e "   ASD-AFM-dev commit:\n   "`git --git-dir ./ASD-AFM-dev/.git log -1` >> ./metadata.txt
echo -e "   Comment: Training PosNet on water Au111 dataset." >> ./metadata.txt

# Divide number of cores by number of GPUs to get number of workers per GPU
num_gpus=$(echo "$SLURM_JOB_GPUS" | sed -e $'s/,/\\\n/g' | wc -l)
num_workers=$((SLURM_CPUS_PER_TASK/num_gpus))
echo "GPUs: $SLURM_JOB_GPUS, num_workers: $num_workers"

# Run fit script
rm -r ~/.cache # Sometimes it gets stuck if there are existing builds of cuda extensions
python -u 4_fit_posnet_augrate.py \
    --train True \
    --test True \
    --predict True \
    --epochs 1000 \
    --num_workers $num_workers \
    --batch_size 4 \
    --lr 1e-3 \
    --avg_best_epochs 10 \
    --pred_batches 20 \
    --data_dir /scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-Au111/ \
    --urls/train "Water-K-{1..10}_train_{0..31}.tar" \
    --urls/val "Water-K-{1..10}_val_{0..7}.tar" \
    --urls/test "Water-K-{1..10}_test_{0..7}.tar" \
    --peak_std 0.20 \
    --zmin -2.5 \
    --z_lims -2.9 0.5 \
    --style_trans True \
    --aug_rate 0.5 \
    --run_dir StyleTransOn50 \
