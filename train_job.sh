#!/bin/bash
# Overnight / weekend training on a DAS GPU node. Submit from the project root:
#     sbatch --time=HH:00:00 train_job.sh HOURS [PREFIX] [extra train.py settings...]
#     sbatch --begin=20:00 --time=12:00:00 train_job.sh 12                          # momo_s0..2, defaults
#     sbatch --time=08:00:00 train_job.sh 8 reg --drop-path 0.1 --label-smoothing 0.1  # reg_s0..2
# DAS-5 allows long jobs only at night and in weekends; during working hours (08:00-20:00) the
# maximum is 15 minutes, hence --begin when submitting during the day. HOURS must match --time.
# Runs PREFIX_s0, PREFIX_s1, PREFIX_s2 one after another. Finished runs are skipped and unfinished
# ones resume, so just resubmit the same line until all three are done.
#SBATCH --job-name=crater
#SBATCH --partition=proq
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX2080Ti
#SBATCH --output=slurm-%j.out

HOURS=${1:?usage: sbatch --time=HH:00:00 train_job.sh HOURS [PREFIX] [settings...]}
PREFIX=${2:-momo}
shift $(( $# < 2 ? $# : 2 ))                   # whatever remains is passed on to train.py
module load cuda13.0/toolkit                   # puts the NVIDIA driver library (libcuda) on the path for torch
export PYTHONUNBUFFERED=1                      # write log lines immediately, so `tail -f` shows live progress
deadline=$(( $(date +%s) + HOURS * 3600 - 600 ))   # stop 10 minutes early so the last epoch is saved

for seed in 0 1 2; do
    minutes=$(( (deadline - $(date +%s)) / 60 ))
    (( minutes < 5 )) && { echo "out of time; resubmit to continue"; break; }
    # the venv's python directly: `uv run` may try to reach the internet, which compute nodes lack
    .venv/bin/python -m modeling.train --name "${PREFIX}_s$seed" --seed "$seed" --max-minutes "$minutes" "$@"
done