#!/bin/bash -l
#SBATCH -J curv_schedule
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 20:00:00
# ---------------------------------------------------------------------------
# Curvature-matched step schedules: linear vs cosine vs power vs an arc-length-matched
# grid derived from the measured profile, at 2-16 steps. Produces
# curvature_schedule.json in $WORK_DIR (Table 4). Inference only.
#
# The #SBATCH lines above are generic placeholders (partition, GPU request). Adjust them
# for your cluster (the reported runs used one 16 GB V100), or run the last line
# directly on any machine with a GPU.
#
#   sbatch slurm/run_curv_schedule.sh
#   SC_PROFILE=$WORK_DIR/curvature_nfe_v2.json sbatch slurm/run_curv_schedule.sh
#
# Without SC_PROFILE it reads curvature_nfe.json, the FIRST curvature run -- see
# open point 5 in README.md. If neither file exists it calibrates.
# ---------------------------------------------------------------------------
set -e
ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$ENV_SH"

export SC_REPS=${SC_REPS:-5}
export SC_NFE=${SC_NFE:-2,4,8,16}
export SC_BASES=${SC_BASES:-latent,direct}
export SC_CAL=${SC_CAL:-64}           # clips used to calibrate the arc-length profile
export EVAL_N=${EVAL_N:-512}
$PY curvature_schedule.py
echo "=== CURV_SCHEDULE DONE ==="
