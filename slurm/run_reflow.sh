#!/bin/bash -l
#SBATCH -J reflow
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 20:00:00
# ---------------------------------------------------------------------------
# Reflow / trajectory rectification. THE ONLY JOB HERE THAT TRAINS. Three phases:
#   1. generate REFLOW_PAIRS teacher pairs   (~1 h at 20k)
#   2. train the student on them             (~7 h at 40k steps)
#   3. NFE sweep assessment                  (~1.5 h)
# Resumable: pair shards are cached under $WORK_DIR/reflow_pairs and the student
# checkpoints every REFLOW_SAVE_EVERY steps, so resubmitting the same line continues.
# Writes ONLY reflow_*.pt -- it never touches latent_*.pt or direct_*.pt.
# Produces reflow_assessment.json in $WORK_DIR.
#
# The #SBATCH lines above are generic placeholders (partition, GPU request). Adjust them
# for your cluster (the reported runs used one 16 GB V100), or run the last line
# directly on any machine with a GPU.
#
#   sbatch slurm/run_reflow.sh
#   REFLOW_PAIRS=20000 REFLOW_STEPS=40000 sbatch slurm/run_reflow.sh
# ---------------------------------------------------------------------------
set -e
ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$ENV_SH"

export WANDB_RUN=${WANDB_RUN:-reflow_rectification}
export REFLOW_PAIRS=${REFLOW_PAIRS:-20000}   # teacher noise->sample pairs to generate
export REFLOW_STEPS=${REFLOW_STEPS:-40000}   # student training steps
export REFLOW_LR=${REFLOW_LR:-5e-5}
export REFLOW_NFE=${REFLOW_NFE:-1,2,4,8,16,50}
export EVAL_N=${EVAL_N:-1024}
$PY reflow_experiment.py
echo "=== REFLOW DONE ==="
