# In-Sequence (Per-Timestep) Thrust Classification

Scope note: this doc covers
[scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py](../scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py),
the per-timestep sibling of the whole-trajectory
[mambaTimeSeriesClassificationGMATThrusts.py](../scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py)
(see [extended_study_plan.md](extended_study_plan.md) for that script's roadmap). Instead of one
label per ~30-minute trajectory, this script classifies **every timestep**: is thrust occurring
at this minute, and if so, is it Chemical, Electric, or Impulsive?

## Goal and core question

The motivating question was architectural: is it better to train one joint 4-class per-timestep
model (0=No Thrust, 1=Chemical, 2=Electric, 3=Impulsive), or a **two-model cascade** — a binary
"is thrust occurring" detector (stage 1) feeding a 3-class type classifier (stage 2, Chemical/
Electric/Impulsive)?

**Answer built into the script rather than assumed**: both paths run in the same invocation,
sharing the same data split and reporting code, so the comparison is a number in the log, not a
guess. `--mode {all,joint,cascade,stage1,stage2}` controls which paths run.

The motivating argument for the cascade: per-timestep positive rates differ wildly by class —
Chemical bursts last 1 of 30 minutes (config: `chemThrustMin: 1`), Electric bursts last 10 of 30
(`elecThrustMin: 10`), and Impulsive is a one-time event whose effect is treated as permanent (see
"Impulsive label semantics" below). A single joint softmax has to learn all these base rates at
once, which tends to hurt the rarer classes; splitting into "any thrust?" (pools positives
together, less imbalanced) then "which type, given thrust is happening?" (a cleaner 3-way
problem) targets that directly. The risk is error propagation (a stage-1 miss means stage-2 never
gets a vote) and double the models to maintain — hence building both and comparing empirically.


## Data pipeline

`prepareInSequenceThrustClassificationDatasets` loads all 4 classes
(`statesArray{Chemical,Electric,ImpBurn,NoThrust}.npz`) from `{data.yaml classification path}/
{orbit}/{propMin}min-{systems}/` and builds **one shared set of DataLoaders** yielding 4-tuples per
batch: `x[B,T,F], y_joint[B,T], y_stage1[B,T], y_stage2[B,T]`. Sharing one loader across all three
label views (rather than building three separate datasets) guarantees the joint model, stage 1,
and stage 2 all see identical val/test splits and batch ordering — required for the cascade-vs-
joint comparison to be apples-to-apples.

Key pieces:
- **`_getThrustingTime`**: defensive per-class loader described above.
- **`_forwardFillFromFirstEvent`**: `np.maximum.accumulate(thrustingTime, axis=1)`, applied to
  Impulsive only. **Impulsive label semantics**: a delta-v burn permanently alters the orbit, so
  "thrust occurred" is treated as true from the burn instant through the end of the propagation
  window, not just the single instant — this was an explicit design decision (see "Design
  decisions" below). The forward-fill is idempotent regardless of whether the upstream fix ends up
  saving just the instant or the already-persisted window.
- **`_deriveStage1Stage2`**: derives `y_stage1 = (y_joint > 0)` and `y_stage2` (Chemical/Electric/
  Impulsive remapped to 0/1/2, with every non-thrusting position masked to `pad_idx=-100`) from
  `y_joint`. Shared by both the DataLoader-building path and the classic-ML/PCA+MLP row-based
  paths in Phase C.
- **`_icGroupSplit`**: one IC-index permutation applied identically to every class block, so the
  same underlying initial condition (IC index `i`, shared across all 4 class `.npz` files since
  each GMAT-Thrust-Data generator run reseeds the same RNG) always lands in the same split
  regardless of which thrust-type file it appears in — prevents leaking near-duplicate pre-thrust
  dynamics across train/val/test. Assumes equal IC counts per class (asserted).
- `--OE`/`--noise`/`--norm`/`--energy` transforms mirror the whole-trajectory script's semantics
  exactly (see `_applyTransforms`), including the slightly quirky `--energy`-alone-vs-with-`--OE`
  channel-count behavior (1 channel vs. 8 = 7 OE + energy) ported faithfully from the original.

## Two approaches, compared side by side

- **Joint**: one model, `num_classes=4`, straight `CrossEntropyLoss` with inverse-frequency class
  weights (`_infer_class_weights`).
- **Cascade**: stage 1 (`num_classes=2`) and stage 2 (`num_classes=3`, trained only on thrusting
  frames via `pad_idx` masking) are separate models. `combineCascadePredictions` combines them at
  inference (`final = 0 if stage1==0 else stage2+1`) and reports the combined result through the
  same `_reportFromPredictions` used by the joint model, plus a diagnostic breakdown
  (stage-1-only accuracy vs. stage-2 accuracy conditioned on correct stage-1 detection) so a
  cascade shortfall is attributable to bad detection vs. bad typing.

`_reportFromPredictions` is the single formatting/metrics path every model family (neural,
classic-ML) reports through — this is what makes cross-approach comparison trustworthy rather than
approximate.

## Backbones

Every backbone implements `forward(x) -> logits [B,T,num_classes]` — full per-timestep output, no
pooling. This shared contract (`build_model(backbone, num_classes, input_size, hidden_size,
num_layers)`) is what lets one `train_model`/`validateInSequenceClassifier`/`runCascadeEvaluation`
serve every architecture without per-model branching.

| Backbone | Flag | Notes |
|---|---|---|
| LSTM | `--no-lstm` to disable | BiLSTM → LSTM → per-timestep linear head |
| Mamba | `--no-mamba` to disable | `Mamba(config)` → per-timestep linear head; `Mamba.forward` already returns the full `[B,T,D]` sequence with no internal pooling, so no backbone changes were needed vs. the whole-trajectory `MambaClassifier` |
| Transformer | `--transformer` | CLS-token pooling dropped (not needed for per-timestep output); positional embedding + per-position linear head |
| CNN (InceptionTime) | `--cnn` | Same `InceptionModule` stack as the whole-trajectory script (already same-length-preserving); global-average-pool replaced with a 1x1 conv head applied at every timestep |
| Hybrid | `--hybrid` | See below — not a per-timestep nn.Module |

Both run always by default (`--no-lstm`/`--no-mamba` opt out); Transformer/CNN/Hybrid are opt-in.

### Hybrid: whole-trajectory MiniRocket + Mamba (important design lesson)

`--hybrid` pairs a **whole-trajectory** MiniRocket stage-1 detector with the standard per-timestep
Mamba stage 2. This went through a design revision worth recording:

The first implementation windowed MiniRocket (a trailing ~10-step context per timestep, like the
GBDT baselines below). This was wrong. **MiniRocket's PPV (proportion-of-positive-values) pooling
is calibrated to and gets its power from the full series it's fit on** — windowing chopped each
30-step trajectory into ~20 heavily-overlapping, non-independent sub-series, too short for the
kernels to pick up real signal (a 10-minute Electric burst barely fits in a 10-step window; a
1-minute Chemical burst only shows up in 1-2 of ~20 windows). This is specific to MiniRocket's
architecture, not a general cascade principle — the neural backbones are recurrent/local and
genuinely benefit from per-timestep supervision.

The fix: `trainMiniRocketStage1Detector` fits on the **entire** trajectory (`train_stage1.any
(axis=1)` → one binary "does this trajectory contain thrust anywhere" label per row), matching how
`--minirocket` already works in the whole-trajectory script.

This changes stage 1's semantics from a per-timestep detector to a **trajectory-level gate**, which
has a real consequence for the combined per-timestep report: stage 2 has no "background" class (it
was trained with `pad_idx` masking specifically so it never sees non-thrusting frames), so within a
trajectory stage 1 flags positive, there is no further per-timestep suppression — stage 2 alone
determines which minutes look idle vs. thrusting. The chosen combination is **broadcast**: stage
1's trajectory-level 0/1 decision is repeated across all T timesteps
(`np.repeat(pred_stage1_traj[:, None], T, axis=1)`) before combining with stage 2's real per-
timestep predictions via the same `combineCascadePredictions` used elsewhere. The alternative
considered and rejected (for now) was retraining stage 2 as a 4-class model with its own background
class to self-suppress within positive trajectories — rejected because it partially reintroduces
the imbalance problem the cascade was meant to avoid. `runCascadeEvaluation`'s hybrid branch prints
MiniRocket's own trajectory-level accuracy *and* the broadcast-combined per-timestep report
separately, so both granularities are visible.

## Classic ML / GBDT + PCA+MLP baselines (Hankel-windowed rows)

Unlike MiniRocket, tree/linear models have no global-pooling requirement — a windowed context
feature vector per timestep is a natural row for them. `buildHankelWindowRowsPerTimestep(states,
labels, hankel_L=5)` builds one row per `(IC, timestep)` with a trailing `hankel_L`-step context
(frames before `t=hankel_L-1` are dropped — a small, documented data loss, ~13% of frames at
`hankel_L=5, T=30`). This is the whole-trajectory script's `pca_mode="hankel"` windowing minus its
final mean-pool over time.

- **LightGBM** (on by default, `--no-classic` to disable), **XGBoost** (`--xgboost`), **CatBoost**
  (`--catboost`), **Random Forest** (`--rf`), **Extra Trees** (`--extratrees`): same hyperparameters
  as the whole-trajectory script, fit on Hankel rows via `runClassicMLModes`, which dispatches
  joint/cascade/stage1-solo/stage2-solo uniformly across all five families.
- **PCA+MLP** (`--pca N`, `--mlp`): `StandardScaler`+`PCA` fit on train Hankel rows only (95%
  variance retained by default), feeding a small `MLP` trained via `trainMLPRowWise` (a row-wise
  analog of `train_model`).
- **Standalone MiniRocket** (`--minirocket`): whole-trajectory 4-class classifier — a direct
  per-timestep-comparable analog of the whole-trajectory script's `--minirocket`, broadcast across
  timesteps like `--hybrid`'s stage 1. Joint-mode only (prints a no-op note for cascade/stage1/
  stage2 modes rather than doing nothing silently).

## Bugs found and fixed this session (worth knowing if extending this script)

1. **NaN-loss / gradient-poisoning on fully-masked batches** (`train_model`): with the still-broken
   upstream data, positive frames are so sparse that a batch can be entirely `pad_idx` for stage 2,
   making `CrossEntropyLoss(ignore_index=...)` return `nan` (nothing to average over). Calling
   `.backward()` on that would have permanently poisoned every parameter with `nan`. Fixed by
   skipping the optimizer step (and excluding the batch from the loss average) whenever the batch
   loss is `nan`, in both the training and validation loops.
2. **XGBoost `objective="multi:softprob"` returns a probability matrix, not hard labels, when
   `num_class=2`** — confirmed empirically (`.predict()` gives shape `(M,2)` instead of `(M,)`).
   This never surfaced in the whole-trajectory script (always 4-class) but broke stage 1's binary
   cascade here (a reshape crash: `cannot reshape array of size 46800 into shape (26,900)`). Fixed
   by switching to `objective="multi:softmax"`, which returns correct 1D labels for any class count.
3. **CatBoost `.predict()` always returns shape `(M,1)`**, regardless of class count — wouldn't
   crash, but would silently corrupt accuracy via numpy broadcasting when compared against a `(M,)`
   ground-truth array. Fixed with a shared `_predictClassicLabels` helper that defensively
   flattens every classic-ML prediction (`.reshape(-1)`), a no-op for already-1D families.

## CLI reference

```
--systems N              random systems to load (default 1500)
--propMin N               propagation minutes (default 30)
--orbit STR                training orbit (default vleo)
--test STR / --testSys N   OOD test orbit / system count (defaults to --orbit / --systems)
--OE / --noise / --norm / --energy   same semantics as the whole-trajectory script
--velNoise F                velocity noise std (default 1e-3)
--train_ratio F              (default 0.7; val/test split from the remainder)
--one-pass                    1 epoch, for smoke tests
--save                         redirect stdout/stderr to a log file under gmat/data/seqClassification/...
--mode {all,joint,cascade,stage1,stage2}   which of joint/cascade to run (default all)

--no-lstm / --no-mamba        disable the always-on backbones
--transformer / --cnn         opt-in per-timestep backbones
--hybrid                      opt-in mixed cascade: whole-trajectory MiniRocket stage 1 + Mamba stage 2

--no-classic                  disable LightGBM (on by default)
--xgboost / --catboost / --rf / --extratrees   opt-in GBDT/RF baselines (Hankel-windowed rows)
--pca N / --mlp               PCA+MLP baseline (Hankel-windowed rows)
--minirocket                   standalone whole-trajectory MiniRocket 4-class baseline (--mode joint/all only)
```

## Status (phased delivery)

Built as 4 phases, each independently runnable and smoke-tested with `--one-pass`:

- **Phase A** (done): data pipeline, LSTM/Mamba backbones, joint-vs-cascade comparison,
  `train_model`/`validateInSequenceClassifier`/`combineCascadePredictions` infra.
- **Phase B** (done): Transformer/CNN backbones, later revised Hybrid backbone (MiniRocket+Mamba).
- **Phase C** (done): classic ML/GBDT + PCA+MLP + standalone MiniRocket baselines.
- **Phase D** (not started): SHAP analysis and Mamba superweight/super-activation analysis.
  Superweight is expected to be low-risk (operates on Mamba's internal weights, independent of the
  per-timestep head shape). SHAP is expected to be the highest-risk item — `qutils.ml.shap.
  run_shap_analysis` expects a standard `(xb, yb)` loader with one label per sample and a
  `(B, n_classes)`-producing model, so it won't accept `[B,T,C]` output as-is; the planned approach
  is a `SingleTimestepPredictionWrapper` that reframes "explain timestep t" as "explain a small
  whole-window classification ending at t" via the same Hankel-window framing used for the classic-
  ML baselines, keeping `run_shap_analysis` itself unchanged.

## Known limitations / open items

- All current results are against the still-broken upstream dataset (see "Blocking data issue")
  and should be re-run once fixed.
- `--saveNets` (checkpoint saving) from the whole-trajectory script was not ported — out of scope
  for the phases built so far.
- The whole-trajectory OOD test-set path (`--test`/`--testSys` pointing at a second dataset) is
  implemented and reaches the right file path, but wasn't fully exercised end-to-end since only
  one orbit/duration combination (`vleo/30min-1500`) currently exists on disk.
- No shell-sweep script (analogous to `generateThrustClass.sh`/`runClassGen.sh` for the
  whole-trajectory script) exists yet for this script — each invocation is run manually.
