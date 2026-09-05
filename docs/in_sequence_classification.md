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
Chemical bursts last 6 of 30 minutes (config: `chemThrustMin: 6`), Electric bursts last 24 of 30
(`elecThrustMin: 24`) — these scale proportionally with `numMinsToProp`, so ~20% chemical and
~60-80% electric duty cycle at any window — and Impulsive is a one-time event whose effect is
treated as permanent (see
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
- `--energyRate` adds per-timestep orbital-energy rate of change (`np.diff` along time, `prepend`
  giving `t=0` a rate of exactly 0) as an additional feature — physics-informed signal for *when*
  thrust is occurring, since energy should be ~constant under two-body coasting and shows a
  residual spike during thrust. It generalizes the existing `--energy`-alone quirk rather than
  changing it: the "energy-family" channel set (energy and/or rate, whichever flags are set) either
  *replaces* states (no `--OE`, matching `--energy`-alone's existing behavior) or is *appended*
  after OE (`--OE` set) — `--energy` alone keeps its exact current 1-channel-replace behavior.
  `--energyRate` alone (no `--energy`) still computes energy internally, just doesn't include the
  raw value as a channel. Inherits a pre-existing, out-of-scope caveat: `norming_energy` is a
  scalar recomputed fresh per `_applyTransforms` call, so `--norm --energy --test <different orbit>`
  normalizes train/test by different scalars.

## Physics-informed loss (`--physics-loss-weight`)

`--physics-loss-weight W` (default `0.0`, off) adds an auxiliary **Chemical-vs-Electric** loss term
— no new model heads, no new parameters — that nudges the model's *existing* Chemical-vs-Electric
logit margin toward a physics-derived pseudo-target. Chemical thrust accelerations are typically on
the order of LEO's J2 zonal-harmonic gravity perturbation; Electric thrust is on the order of the
much smaller combined J3-J6 perturbation, so which analytic scale the measured residual sits closer
to is itself evidence of the type. Applies to `--mode joint`/`stage2` (`stage1` has no per-type
logits, and classic-ML/PCA+MLP/MiniRocket/hybrid don't go through `train_model`).

Measurement supports this term's premise: Chemical's measured residual is ~3.4×10⁻⁵ km/s², within
~3× of a_J2 (1.2×10⁻⁵) and ~12× above Electric's (~2.9×10⁻⁶) — a real, separable magnitude gap.

> **A companion `detect` (Thrust-vs-NoThrust) term was built and then removed.** It targeted the
> NoThrust/Electric boundary via "is `residual_accel` above the J3-J6 floor". `analyzeThrustSeparability.py`
> measured the quantities it assumed and refuted them: coasting frames already sit at ~2.7×10⁻⁶ km/s²
> versus a J3-J6 floor of ~3.8×10⁻⁸, so the pseudo-target was true on essentially every frame —
> it taught the model *"everything is thrust."* Recalibrating the threshold doesn't rescue it,
> because the underlying per-frame feature has **ROC AUC ≈ 0.52** for Electric-vs-NoThrust (chance):
> any correctly-calibrated threshold flips the target to 0 on true Electric frames, supervising the
> model to *miss* Electric. The quantity that does separate them is the **signed, orbit-integrated**
> energy change (AUC 0.652 over one full orbit), which is trajectory-level and doesn't fit a
> per-timestep loss. See "Separability analysis" below.

The magnitude coincidence is LEO-specific: J2/J3-J6 accelerations fall off steeply with altitude
(J2 ~1/r^4, J3-J6 even faster) while thruster-produced accelerations don't shrink with orbit
altitude, so at GEO both thrust types vastly exceed every zonal harmonic and the cue disappears.
To stay orbit-regime-agnostic without any regime-specific branching, the mechanism normalizes
J2/J3-J6 accelerations by local two-body gravity (`mu/r^2`) into dimensionless ratios `h2`/`h36`
that are large in LEO and collapse toward zero in GEO by construction, and uses that as a **gate**
on the term's weight rather than as a hard target:

- **`qutils.orbital`** gains `twoBodyAccel`, `j2AccelMag` (Curtis closed form) and
  `zonalAccelMag`/`j3to6AccelMag` (general order-0 zonal closed form via the standard Legendre
  recursion, verified to reduce exactly to the J2 formula at degree 2), with JGM2-consistent
  coefficients read from the actual `.cof` file the training data's GMAT force model uses.
- **`_computePhysicsResidualTensors`** (per-timestep, from the raw pre-`--OE`/pre-`--norm` ECI
  snapshot, same snapshot-before-transform pattern as `--energy`) computes `h2 = a_J2/g0`,
  `h36 = a_J3-J6/g0`, and an empirical thrust-accel-scale estimate `residual_accel = |dE/dt|/|v|`
  (the same finite-difference energy residual `--energyRate` uses, but divided by the real 60s
  sample interval to get a physical km/s^2 estimate).
- **`_load_and_label`** builds the pseudo-target/gate pair from those tensors, purely from physics,
  independent of the true label: it compares `residual_accel` against `a_J2`/`a_J3-J6` in log-space
  (does the residual look closer to the J2 scale, "Chemical-like", or the J3-J6 scale,
  "Electric-like"?), gated by `h2+h36` and zeroed outside Chemical/Electric-labeled frames
  (NoThrust/Impulsive get no contribution). A `[physics-loss] chem/elec gate ...` diagnostic prints
  the mean/max gate so the regime-attenuation behavior is checkable from logs (confirmed: ~1.6e-3
  mean at `leo/30min-1500` vs. ~3.9e-5 at `geo/30min-1500`, a ~40× drop with zero regime-specific
  code).
- **`physicsConsistencyLoss(logits, mode, y_phys_target, y_phys_gate)`** reuses the model's
  *existing* logit margin as a binary prediction — `joint` → `logits[Chemical]-logits[Electric]`;
  `stage2` → `logits[0]-logits[1]`. Trained via `BCEWithLogitsLoss`, weighted per-timestep by the
  gate. Added to the existing CE/focal loss in both the train and val loops of `train_model`,
  scaled by `--physics-loss-weight` outside the weighted-mean ratio (folding it into the gate
  instead would make it cancel out as a constant scale factor and silently no-op the CLI flag).
- The physics tensors are threaded through the DataLoader as two extra per-timestep float tensors
  (`y_phys_target`, `y_phys_gate`), widening every batch from a 4-tuple to a 6-tuple — every
  loader-consumer in the file (`_infer_class_weights`, `train_model`,
  `validateInSequenceClassifier`, `_predictPerTimestepNeural`) unpacks
  six values, ignoring the last two where unused. `--physics-loss-weight 0.0` keeps these as
  zero-filled dummy tensors with zero added compute cost — bit-identical behavior to before the
  flag existed.

## Temporal gap-closing (`--smooth-max-gap`)

A purely post-hoc knob — no retraining, no loss changes, evaluation-time only. A per-timestep
classifier can detect a long burst patchily, flickering between Thrust/NoThrust within what is
really one continuous event, even when its overall sense of "something is happening here" is right.
`--smooth-max-gap N` (default `0`, off) fills short NoThrust gaps *between* two thrusting
predictions before event-level reporting/plotting, targeting exactly that patchiness.

The naive versions of "smoothing" — a majority-vote window, or erasing short predicted runs
(minimum-run-length filtering) — are unsafe: they erase short *positive* runs, and a lone true
positive is indistinguishable from a lone false positive by shape alone, so they trade real
detections for flicker removal. The safe version only ever **closes gaps** and never erases a
predicted-positive run, however short:

- **`smoothPerTimestepGrid(pred_grid, max_gap, no_thrust_label=0)`** reuses the existing
  `_findSegments` helper `_eventLevelReport` already uses (no new segment-finding logic): finds
  `NoThrust`-label runs per row, and closes (fills in) a run only if it's `<= max_gap` long **and**
  touches neither end of the row. That edge check transparently handles the classic-ML/PCA+MLP
  Hankel-window padded prefix too (see `buildHankelWindowRowsPerTimestep`) — a run touching the
  row's start is never closed, whether it's a genuine leading NoThrust stretch or a hardcoded pad,
  no `hankel_L`-specific logic needed. Filled frames' *type* (Chemical/Electric/Impulsive) is
  forward-filled from the frame immediately before the gap, matching this file's existing
  Impulsive forward-fill precedent (`_getThrustingTime`/`_forwardFillFromFirstEvent`) rather than
  inventing a new imputation rule.
- **`_reportEventLevelWithSmoothing(...)`** is the shared report+plot hook every call site below
  goes through: it always runs the existing `_eventLevelReport` + `plotSequencePrediction` on the
  **raw** grid first (unchanged from before this flag existed), then — only if `smooth_max_gap >
  0` — a second, clearly labeled (`"--- After temporal gap-closing ---"`) pass on the gap-closed
  grid, saved to a `_smoothed`-suffixed plot path. The raw report is never replaced, only
  supplemented, so the payoff (or lack thereof) is always directly visible side by side. It
  deliberately does **not** feed into `_reportFromPredictions`'s flat per-timestep accuracy numbers
  anywhere — those stay on raw predictions throughout, since gap-closing is an event-shape
  correction, not a general accuracy claim, and flat per-timestep accuracy is dominated by the
  NoThrust majority class regardless.
- **One shared hook covers every cascade variant for every family**: `combineCascadePredictions`
  is the single function backing neural, classic-ML, PCA+MLP, and hybrid cascades, so wiring
  `smooth_max_gap` in there once covers all of them. Three more small, mechanical edits cover the
  standalone joint-mode paths (classic-ML `runClassicFamily`, PCA+MLP `runPCAMLPFamily`, and the
  neural joint block in `main()`). Everything after the smoothed report inside
  `combineCascadePredictions` (the stage-1/end-to-end flat metrics, `stage2_conditional_acc`,
  `stage1_only_acc`) stays on the original unsmoothed `final_pred`, unchanged.
- **Deliberately excluded**: standalone `--minirocket` — its prediction is a single
  whole-trajectory decision broadcast across every timestep, so every row is already constant;
  gap-closing on a constant row is an inherent no-op, not worth an unused parameter. (Hybrid's
  cascade stage 1 is *also* a broadcast, but it still goes through `combineCascadePredictions` like
  every other cascade — harmless no-op there too, no special-casing needed.)

### Measured effect: it depends entirely on the backbone

`scripts/two_body/sweepSmoothGap.py` trains one model and sweeps gap values on that single fixed
prediction grid (no retraining per value — `train_model` has no fixed seed, so re-running per value
would compare *different models*). It also prints the interior-gap length histogram, so a flat
sweep table is distinguishable from a bug. On `leo/30min-1500`, 100 epochs, standardized:

| Backbone | Interior gaps found | Effect | Best gap |
|---|---|---|---|
| LSTM (default / +oversample / +oversample+effective) | **0** | none | n/a |
| CNN (InceptionTime) | 8 (all length 1) | negligible | n/a |
| **Transformer** | **373** (max length 3) | Impulsive event recall **+0.9pp**, Chemical/Electric bit-identical | **3** |

The pattern is architectural, not a training-config accident: LSTM and CNN both carry
local/recurrent context between adjacent timesteps, which structurally suppresses frame-to-frame
flicker regardless of how well-trained the model is. Transformer's attention has no such smoothness
bias, so it's the one place gap-closing earns its keep here. **If you're using LSTM or CNN, leave
this at 0.**

## Separability analysis (`analyzeThrustSeparability.py`)

`scripts/two_body/analyzeThrustSeparability.py` measures how separable the thrust classes are **in
the data itself** — no model, no training. It exists because three architecturally unrelated
backbones (LSTM, CNN, Transformer) all converged on Electric event-level precision of 22.5 / 24.0 /
23.7% at 94–100% recall. Models that different agreeing that closely is evidence of a ceiling
imposed by the data, not by model capacity.

It computes the same `residual_accel = |dE/dt|/|v|` the physics loss uses and scores how well that
single physical quantity separates thrusting from coasting frames (ROC AUC), across every available
propagation window. Results on `leo`, 1500 systems, no measurement noise (i.e. **best case**):

| Window | orbits | Electric AUC (per-frame) | Chemical AUC (per-frame) | Electric AUC (integrated ΔE) |
|---|---|---|---|---|
| 10 min | 0.11 | 0.554 | 1.000 | 0.514 |
| 30 min | 0.33 | 0.522 | 1.000 | 0.527 |
| 100 min | 1.08 | 0.508 | 1.000 | **0.652** |

**Chemical is trivially separable; Electric is not** — measured *this way*. Per-frame,
Electric-vs-NoThrust is a coin flip, which fully explains the precision ceiling. But the cause
turned out to be the measurement itself, not the data.

### Root cause: the Keplerian energy is the wrong invariant

The NoThrust median *signed* `dE/dt` is exactly `+0.000e+00` — the background is a **zero-mean,
orbit-periodic oscillation** of amplitude ~2.7×10⁻⁶ km/s², and electric thrust adds a
**constant-sign** bias of only ~+2.0×10⁻⁷ on top of it (the thruster fires along-track, GMAT's
`ElectricThruster` default), a 1:13 ratio. Chemical sits at ~3.4×10⁻⁵, ~12× *above* the background,
hence AUC 1.000.

That oscillation is **J2, not drag**, and the distinction is decisive because J2 is removable:

- 49.6% of NoThrust frames have `dE/dt > 0`. Drag can never add energy, so the background is
  conservative.
- It tracks **latitude** (+1.3×10⁻⁶ at the equator → −4.9×10⁻⁶ at high latitude, the P₂(sin φ)
  shape) with **no coherent altitude trend** — the inverse of a drag signature.
- Its amplitude matches the analytic J2 potential exchange (2.26×10⁻⁶ predicted vs 2.91×10⁻⁶
  measured). An order-of-magnitude drag estimate is 2.9×10⁻⁹ — **1000× too small to matter, and
  ~70× smaller than the electric thrust itself.** Drag was never the obstacle.

`E = v²/2 − μ/r` is the *Keplerian* energy, which **is not conserved under J2**: energy shuttles
between it and the J2 potential term every orbit. Its time derivative therefore measures that
exchange, not the perturbing forces. Adding the J2 potential term back
(`qutils.orbital.orbitalEnergyJ2`) collapses the background scatter **55×** (5.5×10⁻⁶ → 9.9×10⁻⁸),
lifting the thrust signal above it:

| Window | orbits | per-frame AUC (Kep → J2-corr) | integrated AUC (Kep → J2-corr) |
|---|---|---|---|
| 10 min | 0.11 | 0.520 → **0.826** | 0.514 → 0.842 |
| 30 min | 0.33 | 0.517 → **0.835** | 0.527 → **0.940** |
| 100 min | 1.08 | 0.517 → 0.825 | 0.652 → **1.000** |

**Electric detection was never data-limited — it was a wrong-invariant bug in the feature.**
Per-frame AUC is ~0.83 at *every* window, so window length was not the binding constraint (an
earlier revision of this document concluded it was; that conclusion was an artifact of measuring
with the Keplerian energy and is retracted). Longer windows still help the *integrated* view — 100
min is perfectly separable — but the 30-min data is already workable at 0.94.

This also corrupted the `chem_elec` pseudo-target: thresholding a residual that is mostly J2
exchange put **76% of Electric frames on the wrong side** (only 23.4% correctly targeted 0). With
the J2-inclusive energy that becomes 98.7%. `_computePhysicsResidualTensors` therefore uses
`orbitalEnergyJ2` **unconditionally** — it is internal, has no prior published results to preserve,
and there is no reason to keep a knowingly mis-supervising default. The `--energy`/`--energyRate`
*feature* channels are switched by the opt-in `--j2-energy` flag instead, purely so previously
logged results stay reproducible; turn it on for any low-thrust work.

The original domain premise behind `--physics-loss-weight` holds up on magnitudes (electric thrust
2.0×10⁻⁷ is within ~5× of the J3-J6 scale 3.8×10⁻⁸; chemical 3.4×10⁻⁵ within ~3× of J2 1.2×10⁻⁵) —
but the thing Electric must be separated *from* is the J2 oscillation, ~300× larger than the
electric thrust itself, not the J3-J6 scale. That distinction is why the `detect` term was removed.

## Two approaches, compared side by side

- **Joint**: one model, `num_classes=4`, straight `CrossEntropyLoss` with inverse-frequency class
  weights (`_infer_class_weights`).
- **Class imbalance, two independent knobs**: `--loss-scheme`/`--cb-beta`/`--focal-gamma` reweight
  the *loss* (`_infer_class_weights`, `FocalLoss`). `--oversample` instead reweights *sampling*:
  the neural `train_loader` draws whole training trajectories with replacement via a
  `WeightedRandomSampler`, biased toward trajectories containing rarer per-timestep classes
  (`_computeOversamplingWeights` — weight = max over classes present in a trajectory of
  `1/count_c`). Trajectories, not individual timesteps, are the resampling unit, since shuffling
  timesteps within a sequence would break the temporal context LSTM/Mamba need. Val/test are never
  resampled, and classic-ML/PCA+MLP/MiniRocket/hybrid-stage-1 train on raw arrays outside
  `train_loader`, so `--oversample` only affects the per-timestep neural backbones. The two knobs
  can be combined.
- **Cascade**: stage 1 (`num_classes=2`) and stage 2 (`num_classes=3`, trained only on thrusting
  frames via `pad_idx` masking) are separate models. `combineCascadePredictions` combines them at
  inference (`final = 0 if stage1==0 else stage2+1`) and reports the combined result through the
  same `_reportFromPredictions` used by the joint model, plus a diagnostic breakdown
  (stage-1-only accuracy vs. stage-2 accuracy conditioned on correct stage-1 detection) so a
  cascade shortfall is attributable to bad detection vs. bad typing.

`_reportFromPredictions` is the single formatting/metrics path every model family (neural,
classic-ML) reports through — this is what makes cross-approach comparison trustworthy rather than
approximate.

**Event-level (segment) metrics, a second, complementary report.** Per-timestep accuracy can look
fine while a model still "flickers" between wrong classes across a trajectory (e.g. predicting
Chemical at one timestep and Electric at another within a trajectory that's truly all-one-type) —
per-timestep metrics average this away. `_eventLevelReport` reports segment-level precision/recall
per class instead: a true/predicted event is a maximal contiguous run of one class along the time
axis, and detection is "any overlap" (point-adjust, Xu et al. 2018) — a true event is recalled if
any timestep in its span is predicted correctly; a predicted event is a false positive only if it
has zero overlap with any true event of that class. Two caveats worth remembering when reading the
table: Impulsive's forward-filled label (burn instant → end of window) makes its numbers
structurally easier to satisfy than Chemical/Electric's bounded bursts, so it isn't directly
comparable across classes; and "any overlap" recall alone doesn't penalize flicker by itself — a
spurious class's own *precision* is what catches that.

Wired into every **joint** and **cascade** evaluation across every model family (LSTM/Mamba/
Transformer/CNN, classic-ML/GBDT, PCA+MLP, standalone MiniRocket, hybrid) via
`combineCascadePredictions` (cascade) plus one hook per family's standalone joint path. Two params
handle family-specific quirks: `valid_from` (classic-ML/PCA+MLP grids hardcode the first
`hankel_L-1` timesteps to background — see `predictClassicPerTimestep`/`predictPCAMLPPerTimestep`
— so true events entirely inside that prefix are excluded from the recall denominator rather than
scored as automatic misses) and `granularity_note` (MiniRocket/hybrid-stage1's whole-trajectory
broadcast means every predicted "event" spans the entire row, so recall there collapses to
whole-trajectory detection, not genuine localization — a note is printed alongside those tables to
flag it). **Not** wired into `--mode stage1`/`stage2` standalone solo modes for any family — those
diagnostic paths keep per-timestep-only reporting.

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

### Hybrid: whole-trajectory MiniRocket + CNN (important design lesson)

`--hybrid` pairs a **whole-trajectory** MiniRocket stage-1 detector with a per-timestep CNN
(InceptionTime) stage 2 (stage 2's backbone is hardcoded to `"cnn"` in `build_model(...)` for this
combination, independent of `--cnn`/`--no-mamba`). This went through a design revision worth
recording:

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
--energyRate                  additional per-timestep energy-rate-of-change feature
--j2-energy                   compute --energy/--energyRate from the J2-INCLUSIVE specific energy
                               (orbitalEnergyJ2) instead of Keplerian; recommended for low-thrust
                               work — see "Root cause" under Separability analysis above
--physics-loss-weight F       auxiliary Chemical-vs-Electric physics-consistency loss term
                               (--mode joint/stage2 only); see "Physics-informed loss" above
--smooth-max-gap N            post-hoc: close short NoThrust gaps between thrusting predictions
                               before event-level reporting/plotting; see "Temporal gap-closing" above
--velNoise F                velocity noise std (default 1e-3)
--train_ratio F              (default 0.7; val/test split from the remainder)
--one-pass                    1 epoch, for smoke tests
--save                         redirect stdout/stderr to a log file under gmat/data/seqClassification/...
--mode {all,joint,cascade,stage1,stage2}   which of joint/cascade to run (default all)

--no-lstm / --no-mamba        disable the always-on backbones
--transformer / --cnn         opt-in per-timestep backbones
--hybrid                      opt-in mixed cascade: whole-trajectory MiniRocket stage 1 + CNN stage 2

--no-classic                  disable LightGBM (on by default)
--xgboost / --catboost / --rf / --extratrees   opt-in GBDT/RF baselines (Hankel-windowed rows)
--pca N / --mlp               PCA+MLP baseline (Hankel-windowed rows)
--minirocket                   standalone whole-trajectory MiniRocket 4-class baseline (--mode joint/all only)
```

## Status (phased delivery)

Built as 4 phases, each independently runnable and smoke-tested with `--one-pass`:

- **Phase A** (done): data pipeline, LSTM/Mamba backbones, joint-vs-cascade comparison,
  `train_model`/`validateInSequenceClassifier`/`combineCascadePredictions` infra.
- **Phase B** (done): Transformer/CNN backbones, later revised Hybrid backbone (MiniRocket+CNN).
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
