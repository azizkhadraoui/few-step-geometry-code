#!/bin/bash -l
#SBATCH -J threespace
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 4:00:00
# ---------------------------------------------------------------------------
# Trajectory geometry measured at three points of the pipeline, Z -> X -> Q, so the
# distortion splits into a decoder part and a kinematic part. Produces
# three_space_geometry.json and fig_three_space.pdf in $WORK_DIR. Inference only.
#
# The #SBATCH lines above are generic placeholders (partition, GPU request). Adjust them
# for your cluster (the reported runs used one 16 GB V100), or run the last line
# directly on any machine with a GPU.
#
#   sbatch slurm/run_three_space.sh
# ---------------------------------------------------------------------------
set -e
ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$ENV_SH"

export TS_REPS=${TS_REPS:-5}
export TS_NFE=${TS_NFE:-1,2,4,8,16}
export TS_BASES=${TS_BASES:-latent,direct}
export EVAL_N=${EVAL_N:-512}
$PY three_space_geometry.py
echo "=== THREE SPACE DONE ==="
