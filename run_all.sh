#!/bin/bash
#SBATCH --job-name=mmbert
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --exclude=c001,h01,n04,n06,n01
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --output=/scratch/bvandur1/oweller2/nlu-evals/logs/%x_%j_%a.log
#SBATCH --error=/scratch/bvandur1/oweller2/nlu-evals/logs/%x_%j_%a.log

# evaluate each folder in the model_dir
cd /scratch/bvandur1/oweller2/nlu-evals
source env/bin/activate

# debug prints
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT: $SLURM_ARRAY_TASK_COUNT"
echo "SLURM_ARRAY_TASK_MAX: $SLURM_ARRAY_TASK_MAX"
echo "SLURM_ARRAY_TASK_MIN: $SLURM_ARRAY_TASK_MIN"

# hostname and other stats as well as time
echo "Hostname: $(hostname)"
echo "Time: $(date)"

entire_path=$1
folder_only=$(basename $(dirname $entire_path))
model_name=$(basename $entire_path)
echo "model_name: $model_name"
echo "folder_only: $folder_only"
echo "entire_path: $entire_path"

# Run all subsets sequentially
for (( group = 0; group <= 5; group += 1 ))
do
    bash scripts/entry.sh $entire_path results/$folder_only-$model_name $group
done

# Get result
python tools/gather_glue.py results/$folder_only--$model_name
python tools/gather_xtreme.py results/$folder_only--$model_name

echo "End Time: $(date)"