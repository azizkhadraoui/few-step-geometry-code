#!/bin/bash -l
#SBATCH -J disentangle
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 3:00:00
# ---------------------------------------------------------------------------
# Is the output-map gain a predictor, or only an amplifier? Repeats the correlation
# against integration-space degradation, where no amplification is possible, and
# reports degradation in both MPJPE and the RMS norm. Produces
# gain_disentangle.json in $WORK_DIR. Inference only.
#
# The #SBATCH lines above are generic placeholders (partition, GPU request). Adjust them
# for your cluster (the reported runs used one 16 GB V100), or run the last line
# directly on any machine with a GPU.
#
#   sbatch slurm/run_disentangle.sh
# ---------------------------------------------------------------------------
set -e
ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$ENV_SH"

export DS_REPS=${DS_REPS:-5}
export DS_NFE=${DS_NFE:-1,2,4,8,16}
export DS_BASES=${DS_BASES:-latent,direct}
export EVAL_N=${EVAL_N:-512}
$PY gain_disentangle.py
echo "=== DISENTANGLE DONE ==="
