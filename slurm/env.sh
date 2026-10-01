# ---------------------------------------------------------------------------
# Shared environment for every job in slurm/. Sourced by each run_*.sh.
#
# Nothing in this repository hardcodes a site path: set the four variables below
# (here, in your shell, or on the sbatch command line) and everything follows.
#
#   HML3D_ROOT   HumanML3D `humanml` directory (must contain train.txt,
#                new_joint_vecs/, texts/)
#   RVQ_CKPT     frozen RVQ-VAE checkpoint, e.g. .../rvq_vae_best.pt
#   EVAL_ROOT    directory holding the frozen motion evaluator
#                (movement_encoder.pt + motion_encoder.pt, or finest.tar)
#   WORK_DIR     writable output directory: checkpoints, JSON results, figures
#
# Anything left unset is searched for under SEARCH_ROOTS (default: $WORK, $HOME).
# ---------------------------------------------------------------------------

# Repository root. Submit from the repo root, or pass REPO=/path/to/repo.
export REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}

# Python interpreter (a conda env, a venv, or plain `python`).
export PY=${PY:-python}

# ---- paths: edit these, or export them before submitting -------------------
export HML3D_ROOT=${HML3D_ROOT:-}
export RVQ_CKPT=${RVQ_CKPT:-}
export EVAL_ROOT=${EVAL_ROOT:-}
export WORK_DIR=${WORK_DIR:-$REPO/runs}
export SEARCH_ROOTS=${SEARCH_ROOTS:-}

# ---- fixed by the repo layout ----------------------------------------------
export MAIN_SCRIPT=$REPO/lfm_clfm_cdfm_experiment.py

# ---- Weights & Biases: off by default; set USE_WANDB=1 to enable ------------
export USE_WANDB=${USE_WANDB:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-motion-clfm}
export WANDB_ENTITY=${WANDB_ENTITY:-}

mkdir -p "$WORK_DIR"
[ -f "$MAIN_SCRIPT" ] || { echo "MAIN_SCRIPT not found: $MAIN_SCRIPT (submit from the repo root, or set REPO=)"; exit 1; }
cd "$REPO"
echo "REPO=$REPO  WORK_DIR=$WORK_DIR"
echo "HML3D_ROOT=${HML3D_ROOT:-<search>}  RVQ_CKPT=${RVQ_CKPT:-<search>}  EVAL_ROOT=${EVAL_ROOT:-<search>}"
