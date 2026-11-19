#!/bin/bash
# Overnight / weekend training on a DAS GPU node. Submit from the project root with the same
# number of hours twice (once for SLURM, once for this script):
#     sbatch --begin=20:00 --time=12:00:00 train_job.sh 12                  # a night (20:00-08:00)
#     sbatch --begin=2026-11-20T20:00 --time=60:00:00 train_job.sh 60      # a weekend (Fri 20:00-Mon 08:00)
# DAS-5 allows long jobs only at night and in weekends; during working hours (08:00-20:00) the
# maximum is 15 minutes, hence --begin. The RTX 2080 Ti nodes are in partition proq.
# Runs momo_s0, momo_s1, momo_s2 one after another. Finished runs are skipped and unfinished
# ones resume, so just resubmit until all three are done.
#SBATCH --job-name=crater
#SBATCH --partition=proq
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX2080Ti
#SBATCH --output=slurm-%j.out

HOURS=${1:?usage: sbatch --time=HH:00:00 train_job.sh HOURS}
module load cuda13.0/toolkit        # puts the NVIDIA driver library (libcuda) on the path for torch
deadline=$(( $(date +%s) + HOURS * 3600 - 600 ))   # stop 10 minutes early so the last epoch is saved

for seed in 0 1 2; do
    minutes=$(( (deadline - $(date +%s)) / 60 ))
    (( minutes < 5 )) && { echo "out of time; resubmit to continue"; break; }
    # the venv's python directly: `uv run` may try to reach the internet, which compute nodes lack
    .venv/bin/python -m modeling.train --name "momo_s$seed" --seed "$seed" --max-minutes "$minutes"
done