# Which Geometry Predicts Few-Step Degradation in Latent Flow Matching?

**Mohamed Aziz Khadraoui**<sup>1</sup> · **Abdelkader Baggag**<sup>2</sup><br>
<sup>1</sup>Higher School of Communication of Tunis (SUP'COM), Tunisia · <sup>2</sup>Qatar Computing Research Institute, Hamad Bin Khalifa University

[BeNTo Workshop @ NeurIPS 2026](https://bento-neurips.github.io/) — Beyond Next-Token Prediction: Diffusion & Flow Models for Next-Generation Decoding

[Project page](https://azizkhadraoui.github.io/few-step-geometry/) · [Paper](https://openreview.net/pdf?id=K0bixbUbER) · [OpenReview](https://openreview.net/forum?id=K0bixbUbER) · [Citation](#citation)

Code release for the paper: trajectory geometry and few-step degradation in latent vs. direct flow matching.

Measurement code for the geometric analysis of text-to-motion flow models on HumanML3D. This README
maps every reported quantity to the script and function that produces it, states the exact
definition used, and lists the points where the implementation and the paper's text do not yet
correspond (§5), so that nothing in the output is read as more settled than it is.

Two generators are compared throughout:

| | integration space | model |
|---|---|---|
| **LFM** (latent flow matching) | `Z`, the frozen RVQ-VAE latent, 49×256 | transformer flow model in latent space |
| **CDFM** (constrained direct flow matching) | `X`, the 196×263 motion representation | DiT on the raw representation |

Everything below is **inference only** except `reflow_experiment.py`, which trains one student model.

---

## 1. Layout

```
lfm_clfm_cdfm_experiment.py    core module: data, models, sampler, evaluator, metrics
wandb_opt.py                   optional W&B logging (no-ops unless USE_WANDB=1)

curvature_nfe_v2.py            curvature statistics, correlations, stratified FID
three_space_geometry.py        arc/chord and turning angle in Z, X and Q
gain_disentangle.py            output-map gain: predictor or amplifier?
curvature_schedule.py          curvature-matched step schedules
guidance_control.py            teacher guidance sweep, reflow comparison control
reflow_experiment.py           trains the reflow student, then sweeps NFE
make_figs.py                   the four ablation figures

slurm/env.sh                   paths and knobs shared by every job — edit this one file
slurm/run_*.sh                 one launcher per script
figures/                       figures from the reported runs
```

### Paper ↔ code

| script | reported as |
|---|---|
| `curvature_nfe_v2.py` | curvature statistics, correlations, stratified FID (Tables 2–3, Figs. 2–4) |
| `three_space_geometry.py` | arc/chord and turning angle in `Z`, `X`, `Q` (three-space table) |
| `gain_disentangle.py` | output-map gain variants and the amplification control |
| `curvature_schedule.py` | step-schedule comparison (Table 4) |
| `guidance_control.py` | teacher guidance sweep and reflow comparison (Table 5, Fig. 5) |
| `reflow_experiment.py` | trains the reflow student used by the comparison above |
| `make_figs.py` | the four ablation figures (main replicated table, guidance sweep, in-process projection, sampler ablation) |

### How the scripts share the core module

`lfm_clfm_cdfm_experiment.py` holds everything: the HumanML3D dataset, the frozen RVQ-VAE, both
flow models, the sampler, the frozen motion evaluator, and the metrics. Each measurement script
imports it with `ABLATION_IMPORT=1`, which makes the module raise the sentinel `M_ABLATION_STOP`
once the models, dataset and evaluator are constructed but *before* it runs its own evaluation. The
importing script catches the sentinel and reuses the loaded objects:

```python
os.environ["ABLATION_IMPORT"] = "1"
spec = importlib.util.spec_from_file_location("expmod", MAIN)
M = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP":
        raise
```

**Consequence worth knowing:** only definitions appearing *before* that raise (top of Cell 13, line
720) are visible to a measurement script. Anything defined after it — the qualitative figures, the
GIF rendering — is not.

---

## 2. Setup

### Dependencies

```bash
pip install -r requirements.txt
```

PyTorch with CUDA, plus `transformers` for the `sentence-t5-base` caption encoder. `wandb` is
optional and off by default; every number is printed to stdout and written to JSON without it.

### What you need on disk

| | what | how it is found |
|---|---|---|
| **HumanML3D** | the `humanml` directory containing `train.txt`, `new_joint_vecs/`, `texts/` | `HML3D_ROOT` |
| **RVQ-VAE** | frozen motion autoencoder checkpoint, `rvq_vae_best.pt` | `RVQ_CKPT` |
| **Evaluator** | frozen motion evaluator: `movement_encoder.pt` + `motion_encoder.pt`, or `finest.tar` | `EVAL_ROOT` |
| **Flow models** | `latent_best.pt`, `direct_best.pt` under `$WORK_DIR/clfm/ckpt/` | trained by the core module |

Anything left unset is searched for under `SEARCH_ROOTS` (`os.pathsep`-separated; defaults to
`$WORK` then `$HOME`). No site path is compiled into any file.

### Environment variables

| variable | meaning | default |
|---|---|---|
| `HML3D_ROOT` | HumanML3D `humanml` directory | searched |
| `RVQ_CKPT` | frozen RVQ-VAE checkpoint | searched |
| `EVAL_ROOT` | frozen motion evaluator directory | searched |
| `SEARCH_ROOTS` | roots to search when a path is unset | `$WORK`, `$HOME` |
| `WORK_DIR` | writable output root — checkpoints, JSON, figures | `./runs` |
| `MAIN_SCRIPT` | path to the core module | beside the script |
| `EVAL_N` | clips per replication | 512 (1024 for reflow) |
| `USE_WANDB` | `1` enables W&B logging | `0` |
| `VARIANT` | what a core-module job trains: `latent` \| `direct` \| `latent_pen` \| `direct_pen` \| `all` \| `eval` | `all` |

### Training the base models

The measurement scripts need `latent_best.pt` and `direct_best.pt`. They come from the core module,
one base per job:

```bash
export HML3D_ROOT=... RVQ_CKPT=... WORK_DIR=...
VARIANT=latent  python lfm_clfm_cdfm_experiment.py     # 300k steps
VARIANT=direct  python lfm_clfm_cdfm_experiment.py
VARIANT=eval    python lfm_clfm_cdfm_experiment.py     # builds the projection table + figures
SMOKE_TEST=1 VARIANT=all python lfm_clfm_cdfm_experiment.py   # 200-step end-to-end check
```

Training is resumable: checkpoints land every 2000 steps and a resubmitted job continues.

---

## 3. Running the measurements

Edit the paths at the top of [`slurm/env.sh`](slurm/env.sh) once — every launcher sources it — then
submit from the repository root:

```bash
sbatch slurm/run_curvature_v2.sh
sbatch slurm/run_three_space.sh
sbatch slurm/run_disentangle.sh
sbatch slurm/run_curv_schedule.sh
sbatch slurm/run_reflow.sh              # the only job that trains
sbatch slurm/run_guidance_control.sh    # after run_reflow.sh
```

The `#SBATCH` headers (partition `gpu`, `--gres gpu:1`) are generic placeholders; the reported runs
used a single 16 GB V100. Adjust them for your cluster, or ignore the launchers entirely and run the
scripts directly with the environment set:

```bash
CV_REPS=5 CV_NFE=1,2,4,8,16 EVAL_N=512 python curvature_nfe_v2.py
```

### What each script does

| script | measures | writes to `$WORK_DIR` | wall clock (V100) |
|---|---|---|---|
| `curvature_nfe_v2.py` | curvature statistics, their Spearman correlation with few-step degradation, FID stratified by curvature quartile | `curvature_nfe_v2.json`, `fig_curvature_dist.pdf` | ~10 h |
| `three_space_geometry.py` | arc/chord and max turning angle in `Z`, `X`, `Q`; splits distortion into a decoder factor `X/Z` and a kinematic factor `Q/X` | `three_space_geometry.json`, `fig_three_space.pdf` | ~4 h |
| `gain_disentangle.py` | whether the output-map gain predicts degradation measured in the *integration* space, where no amplification is possible; degradation in both MPJPE and the RMS norm | `gain_disentangle.json` | ~3 h |
| `curvature_schedule.py` | linear vs. cosine vs. power vs. arc-length-matched step grids at 2–16 steps | `curvature_schedule.json` | ~20 h |
| `reflow_experiment.py` | generates teacher pairs, trains the student, sweeps NFE × {teacher, student} × {no projection, post-decode projection} | `reflow_assessment.json`, `reflow_*.pt` | ~10 h |
| `guidance_control.py` | teacher at guidance 1.0/1.5/2.5 vs. student at 1.0 — separates the guidance scale from the straightening | `guidance_control.json` | ~4 h |

Per-script knobs (`CV_*`, `TS_*`, `DS_*`, `SC_*`, `GC_*`, `REFLOW_*`) are listed with their defaults
at the top of the matching launcher.

`reflow_experiment.py` writes **only** `reflow_*.pt` and never touches `latent_*.pt` or
`direct_*.pt`. It is resumable: pair shards are cached under `$WORK_DIR/reflow_pairs` and the
student checkpoints every `REFLOW_SAVE_EVERY` steps, so resubmitting the same line continues.

One caution the reflow code enforces and prints: the student's training pairs were generated with
classifier-free guidance already applied, so its velocity field has the guided field baked in and it
**must** be sampled at guidance 1.0. Sampling it at 2.5 applies guidance twice, FID explodes, and it
looks as though reflow failed. That asymmetry is exactly what `guidance_control.py` controls for.

---

## 4. What is measured

### The three spaces

```
Z  --D_phi-->  X  --K-->  Q
```

- **`Z`** — the state the ODE integrates. For LFM this is the standardized continuous encoder
  output, 49×256. **For CDFM there is no separate `Z`**: the sampler integrates 196×263 directly, so
  `Z` and `X` are the same tensor. The two rows coinciding for CDFM is a definitional identity, not
  a measurement.
- **`X`** — the 196×263 motion representation. `rvq.decoder(z * z_std + z_mean)` for LFM, the state
  itself for CDFM. The un-standardization is part of the map.
- **`Q`** — Cartesian joints, 196×22×3, from `_gj` ([`lfm_clfm_cdfm_experiment.py:390`](lfm_clfm_cdfm_experiment.py#L390)),
  which is `recover_from_ric(mn * std_t + mean_t)`. So `K` as implemented includes the
  un-normalization by the dataset statistics as well as the forward-kinematic recovery.

Norms: Euclidean on the flattened state for `Z` and `X`; frame- and joint-averaged RMS over valid
frames for `Q` (`d_q`, [`three_space_geometry.py:69`](three_space_geometry.py#L69)). Explicitly,
for `Q`,

```
||Q||²_rms = (1/|T|) Σ_{τ∈T} (1/J) Σ_j ||q_{τ,j}||²
```

with `T` the frames before padding.

### The statistics

- **Arc-over-chord `R`** — summed step displacement over endpoint distance, in the norm of the space
  concerned, on the 50-step reference trajectory (`arc_chord`). Equals 1 for a straight path.
- **Maximum turning angle** — the largest angle between consecutive step displacements,
  `max_k arccos(<D_k, D_{k+1}> / (|D_k| |D_{k+1}|))`, cosine clamped to ±(1 − 1e−6), measured from
  the *extension* of the incoming segment so a straight path gives 0. Resolution-dependent — it
  tends to 0 as *n* → ∞ for any smooth path, so it is comparable only at a fixed grid; always
  *n* = 50 here.
- **Output-map gain** — finite-difference Jacobian-vector products at the terminal state with
  `P = 4` random unit probes and `eps = 1e−2`:
  `g = (1/P) * sum_p |Psi(Y1 + eps*v_p) − Psi(Y1)|_rms / eps`. Variants at other evaluation points
  (trajectory mean, trajectory max, segment midpoint) are in `gain_disentangle.py`.
- **Velocity variation** — mean deviation of the velocity field along the trajectory from its
  trajectory average, normalized by that average: the straightness measure from the rectified-flow
  literature.
- **Degradation** — per clip, MPJPE between the sample at *n* steps and the same clip's 50-step
  sample **at identical initial noise**, enforced by passing the same `seed` to both calls. A
  per-sample quantity, precisely because FID is not.
- **FID and R-precision** — `fid_calc` and `rprec`
  ([`lfm_clfm_cdfm_experiment.py:501`](lfm_clfm_cdfm_experiment.py#L501),
  [`:505`](lfm_clfm_cdfm_experiment.py#L505)), on features from `memb`
  ([`:499`](lfm_clfm_cdfm_experiment.py#L499)), the frozen motion evaluator. R@3 is `rprec(...)[3]`.

### Protocol: seeding, clip selection, averaging

- **Clip set.** Fixed across every experiment: `np.random.default_rng(0)`, first 512 of a
  permutation of the test indices, sorted. The same clips in the same order everywhere.
- **Sampling seed.** Within a batch starting at index `s` and replication `r`, the seed is
  `s + 100000*r`. A given clip therefore receives the same noise at every step budget within a
  replication — which is what makes the degradation measure paired — and different noise across
  replications.
- **Two levels of averaging.** Curvature statistics and correlations are computed *per clip*, then
  reduced to one number per replication (a mean, or a Spearman coefficient across clips); the
  confidence interval is taken across replications. The intervals therefore describe seed-to-seed
  variability, not clip-to-clip. Clip-to-clip spread is reported separately as a standard deviation,
  and it is the quantity of interest in the regularity result.
- **Replication counts.** Five seeds for curvature, correlations, schedules and the three-space
  study; three for the reflow and guidance comparisons. Intervals are *t*-based with
  `t_{0.975, R−1}` (`ci95`).

---

## 5. Open points

These are **unresolved in this code** and are listed so that nothing in the output is read as
settled.

1. **The `Q` turning angle is probably wrong.** In `Q` the joint tensor is flattened before the
   angle is computed, so the result is an angle between 66-dimensional displacement vectors and
   individual joints' turns partially cancel. This is the likely cause of the non-monotonicity
   (0.358 → 0.406 → 0.347 across the three spaces for LFM) while arc-over-chord rises monotonically.
   Either omit the `Q` turning angle, or define it per joint and aggregate explicitly. The
   non-monotonicity should not be reported as a finding.
2. **`K` as implemented includes un-normalization.** `_gj` multiplies by the dataset standard
   deviation and adds the mean before the kinematic recovery. That affine step is part of what is
   being called the kinematic map. It is a diagonal rescaling, so it cannot bend a path, but it does
   change the metric, and the setup should say that `K` denotes the composition.
3. **The gain is not the Lipschitz constant.** The probe estimates a *mean* singular value at a
   single point — the probes are isotropic, so it is nearer a normalized Frobenius norm than the
   operator norm — whereas the bound's constant is a supremum over the region between the exact and
   the approximate endpoints. The segment-evaluated variant matches the bound's structure for CDFM
   but not for LFM, and that difference is unexplained.
4. **Degradation is in MPJPE, the analysis is in the RMS norm.** They are equivalent up to a
   constant and order clips near-identically (rank correlation 0.993–0.997), so no conclusion
   changes, but the two sides of the inequality are currently measured in different norms.
   `gain_disentangle.py` emits both, so the correspondence can be shown rather than asserted.
5. **The schedule experiment reads a stale profile.** `curvature_schedule.py` loads the decoded
   arc-length profile from `curvature_nfe.json`, the output of the *first* curvature run, not
   `curvature_nfe_v2.json`. The profiles are close and the conclusion is a null result either way,
   but it should be regenerated before it is reported:
   `SC_PROFILE=$WORK_DIR/curvature_nfe_v2.json sbatch slurm/run_curv_schedule.sh`.
6. **Evaluation is on 512 clips, not the full set.** Chosen for cost. The main replicated table
   elsewhere in the project uses 1024, and a full-set control at 2644 was run separately. FID is
   biased upward at smaller sample sizes, so absolute values are not comparable across those
   protocols and should be labelled distinctly.

Also worth stating in any write-up: `Z` and `X` use an unnormalized Euclidean norm while `Q` uses a
frame-averaged one, so the three are not on a common scale. Arc-over-chord is a ratio and therefore
dimensionless, so comparing it across spaces is still valid; any statement about *absolute*
distances is not. The turning angle is likewise scale-free.

---

## 6. Outputs

Each script writes one JSON file to `$WORK_DIR`, keyed by base (`latent`, `direct`). Curvature
results, for example:

```
curvature_nfe_v2.json
└── latent
    ├── reps, n_eval
    ├── curvature      { predictor: {mean, ci} }
    ├── correlations   { predictor: { n: {mean, ci} } }     Spearman vs. degradation
    ├── fid            { n: {mean, ci} }
    ├── stratified     { curvature quartile: { n: FID } }
    └── per_clip_seed0 { predictor: [ per-clip values, seed 0 ] }
```

### Figures

`figures/` holds the figures from the reported runs. Where each one comes from:

| figure | produced by |
|---|---|
| `fig_curvature_dist.pdf` | `curvature_nfe_v2.py`, at the end of its run |
| `fig_main_replicated.pdf`, `fig_guidance_sweep.pdf`, `fig_inproc_ablation.pdf`, `fig_sampler_ablation.pdf` | `make_figs.py` (`FIG_DIR=figures python make_figs.py`), from values inlined in the script |
| `fig_nfe_curves.pdf`, `fig_predictors.pdf`, `fig_stratified.pdf` | rendered from the saved JSON (`curvature_nfe_v2.json`, `guidance_control.json`) by a plotting script that is not part of this snapshot |

`three_space_geometry.py` likewise emits `fig_three_space.pdf` at the end of its run.

---

## 7. Provenance

The code is the same as the version that produced the reported numbers. Only the following changed
for release, none of it affecting any measurement:

- Cluster-specific paths and notebook scaffolding were removed. Dataset, checkpoint and evaluator
  locations now come from environment variables, with `SEARCH_ROOTS` as the fallback.
- The SLURM launchers' repeated path exports moved into `slurm/env.sh`; the `#SBATCH` resource lines
  are generic placeholders; job logs go to `slurm-%x-%j.out` in the submit directory.
- `wandb` became optional (`wandb_opt.py`), off unless `USE_WANDB=1`. `IPython` likewise — it was a
  hard import used only by the notebook figure cells.
- Line numbers in `lfm_clfm_cdfm_experiment.py` are unchanged, so the line references in this README
  (`_gj` at 390, `memb` at 499, `fid_calc` at 501, `rprec` at 505) resolve.

Not included here: the earlier exploratory scripts (composition, manifold projection, round-trip,
in-process projection ablations, external evaluation) that the paper does not report.

---

## Citation

```bibtex
@inproceedings{khadraoui2026geometry,
  title     = {Which Geometry Predicts Few-Step Degradation in Latent Flow Matching?},
  author    = {Mohamed Aziz Khadraoui and Abdelkader Baggag},
  booktitle = {NeurIPS 2026 Workshop on Beyond Next-Token Prediction:
               Diffusion \& Flow Models for Next-Generation Decoding (BeNTo)},
  year      = {2026}
}
```

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
