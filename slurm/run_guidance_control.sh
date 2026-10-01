#!/bin/bash -l
#SBATCH -J gctrl
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 4:00:00
# ---------------------------------------------------------------------------
# Guidance control for the reflow comparison: evaluates the teacher at guidance 1.0 on
# the same clips and seeds, separating the guidance scale from the straightening.
# Produces guidance_control.json in $WORK_DIR (Table 5, Fig 5). Inference only.
# Requires the reflow student checkpoint from run_reflow.sh.
#
# The #SBATCH lines above are generic placeholders (partition, GPU request). Adjust them
# for your cluster (the reported runs used one 16 GB V100), or run the last line
# directly on any machine with a GPU.
#
#   sbatch slurm/run_guidance_control.sh
# ---------------------------------------------------------------------------
set -e
ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$ENV_SH"

export GC_REPS=${GC_REPS:-3}
export GC_NFE=${GC_NFE:-1,2,4,8,16,50}
export GC_GUIDES=${GC_GUIDES:-1.0,1.5,2.5}
export EVAL_N=${EVAL_N:-512}
$PY guidance_control.py
echo "=== GUIDANCE CONTROL DONE ==="
