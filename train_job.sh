#!/bin/bash
# Overnight / weekend training on a DAS GPU node. Submit from the project root with the same
# number of hours twice (once for SLURM, once for this script):
#     sbatch --time=12:00:00 train_job.sh 12      # a night
#     sbatch --time=60:00:00 train_job.sh 60      # a weekend
# Runs momo_s0, momo_s1, momo_s2 one after another. Finished runs are skipped and unfinished
# ones resume, so just resubmit until all three are done.
#
# Check the constraint name against DAS-5 (`sinfo -o "%N %f"` lists node features).
#SBATCH --job-name=crater
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX2080Ti
#SBATCH --output=slurm-%j.out

HOURS=${1:?usage: sbatch --time=HH:00:00 train_job.sh HOURS}
deadline=$(( $(date +%s) + HOURS * 3600 - 600 ))   # stop 10 minutes early so the last epoch is saved

for seed in 0 1 2; do
    minutes=$(( (deadline - $(date +%s)) / 60 ))
    (( minutes < 5 )) && { echo "out of time; resubmit to continue"; break; }
    uv run python -m modeling.train --name "momo_s$seed" --seed "$seed" --max-minutes "$minutes"
done