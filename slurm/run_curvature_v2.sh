#!/bin/bash -l
#SBATCH -J curv_v2
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 10:00:00
# ---------------------------------------------------------------------------
# Curvature and few-step degradation, replicated over seeds, with the decoder-gain
# and turning-angle predictors. Produces curvature_nfe_v2.json (Tables 2-3) and
# fig_curvature_dist.pdf in $WORK_DIR. Inference only.
#
# The #SBATCH lines above are generic placeholders (partition, GPU request). Adjust them
# for your cluster (the reported runs used one 16 GB V100), or run the last line
# directly on any machine with a GPU.
#
#   sbatch slurm/run_curvature_v2.sh
# ---------------------------------------------------------------------------
set -e
ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$ENV_SH"

export CV_REPS=${CV_REPS:-5}          # sampling seeds per configuration
export CV_NFE=${CV_NFE:-1,2,4,8,16}   # step budgets to degrade at
export CV_BASES=${CV_BASES:-latent,direct}
export EVAL_N=${EVAL_N:-512}          # clips per replication
$PY curvature_nfe_v2.py
echo "=== CURVATURE V2 DONE ==="
