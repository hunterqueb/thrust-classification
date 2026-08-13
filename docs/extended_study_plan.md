# Extended Study Plan: Coordinate Frames, Classifiers, Data Regimes

Scope note: this plan covers extensions to
[scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py](../scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py)
and its dependency `qutils.ml.classifer.prepareThrustClassificationDatasets`. Nothing below exists yet
in `qutils` unless noted otherwise — `qutils/orbital.py` currently only provides `ECI2OE`/`OE2ECI` and
energy helpers, so both frame additions below are new implementation work, not flag flips.

## 1. Coordinate frames for this study: ROE and Delaunay/Poincaré

### 1.1 Quasi-nonsingular relative orbital elements (ROE)

**Definition**: deviation of the actual (thrusted) orbit from a reference/chief orbit, expressed as
`[δa, δλ, δe_x, δe_y, δi_x, δi_y]` (relative semi-major axis, mean longitude, eccentricity vector
components, inclination vector components).

**Why it fits this dataset specifically**: `prepareThrustClassificationDatasets` already loads a
same-initial-condition no-thrust trajectory (`statesArrayNoThrust`) alongside each thrust class. That
no-thrust twin *is* the natural chief/reference orbit — no extra propagation needed, just reuse of
data already generated.

Steps:
- [ ] Add `OE2ROE(oe_chief, oe_deputy)` (or `ECI2ROE` taking both state vectors) to `qutils/orbital.py`,
      built on top of the existing `ECI2OE` classical-element output.
- [ ] In `prepareThrustClassificationDatasets`, compute chief elements from `statesArrayNoThrust` at
      each timestep/system, and deputy elements from each thrust class's own trajectory, then take the
      quasi-nonsingular difference. Mirror the existing `useOE` branch structure (see the block starting
      at `qutils/ml/classifer.py:105`) rather than inventing a new data path.
- [ ] Add `--ROE` flag to `mambaTimeSeriesClassificationGMATThrusts.py`, parallel to `--OE`, with
      `feat_names = ['da','dlambda','dex','dey','dix','diy']` for the SHAP feature-name wiring
      (`main()` around line 619).
- [ ] Sanity check before trusting results: ROE should be ~0 (up to noise) for correctly-paired
      no-thrust vs. no-thrust comparisons, and should show class-separable drift for the three thrust
      classes. Add this as an assert/plot in a scratch notebook before wiring into training — cheap way
      to catch a sign/frame error early.
- [ ] Watch the linearization-validity assumption: quasi-nonsingular ROE is a small-deviation
      approximation. Confirm impulsive-burn trajectories don't push `δe`/`δi` far enough from the chief
      to invalidate the linear mapping — if they do, either fall back to nonlinear ROE or flag
      Impulsive as a known-harder case for this frame.
- [ ] Consider adding ROE *rates* (finite-difference in time) as auxiliary channels — burn-type
      signatures should show up as derivative discontinuities (impulsive/chemical) vs. smooth nonzero
      slopes (electric), same rationale as the existing `--energy` feature.

### 1.2 Delaunay / Poincaré (action-angle) elements

**Definition**: Delaunay elements are canonical actions/angles `L = √(μa), G = L√(1−e²), H = G·cos(i)`
paired with angles `l = M, g = ω, h = Ω`. Poincaré elements are the nonsingular reformulation
(`p₁ = √(2(L−G))·cos(g+l)`, etc.) that removes the low-`e`/low-`i` singularities Delaunay inherits from
classical elements.

Steps:
- [ ] Add `OE2Delaunay(oe)` to `qutils/orbital.py`, a direct closed-form transform off the existing
      `ECI2OE` output (`a, e, i, Ω, ω, M`) — no new orbit determination needed, just the action-angle
      formulas above.
- [ ] Add `Delaunay2Poincare(delaunay)` for the nonsingular variant — needed for VLEO/LEO where
      near-circular, near-equatorial orbits will otherwise hit the same `g`/`h` singularity issues
      already flagged for classical OE.
- [ ] Add `--delaunay` / `--poincare` flags analogous to `--OE`, with the same normalization treatment
      currently applied to `a` under `useNorm` (`qutils/ml/classifer.py:116`) — `L` scales with `√a`, so
      confirm whether normalization should apply to `L` directly or to the underlying `a` before
      conversion.
- [ ] Run as an explicit ablation against `--energy`: `L = √(μa)` is a monotonic function of specific
      orbital energy, so the Delaunay/Poincaré frame likely subsumes the existing energy feature.
      Worth checking whether `--energy` adds anything once `L` is already in the input, or whether it's
      redundant.
- [ ] Since Poincaré is the non-singular variant, prefer it as the default for VLEO/LEO runs; keep raw
      Delaunay available for GEO/HEO runs (less singular there) as a cheaper-to-interpret alternative for
      SHAP analysis, since Poincaré's mixed sin/cos action-angle terms are harder to attribute physically.

### Shared implementation notes
- Both frames should go through the same `train_ratio`/`val_ratio`/`test_ratio` split logic and
  `output_np` path already in `prepareThrustClassificationDatasets` — no changes needed downstream of
  feature computation.
- Extend `--shap` feature-name handling (`feat_names` blocks at lines ~567 and ~619) for both new flags
  so `run_shap_analysis` output stays labeled correctly instead of silently reusing OE/ECI names.
- Update `strAdd` log-filename tagging (`qutils/... strAdd` block, lines 240–270) so `--ROE` and
  `--delaunay`/`--poincare` runs produce distinguishable log/artifact filenames, consistent with how
  `--OE`/`--energy`/`--norm` are currently tagged.

## 2. Additional classifiers

Priority order, cheapest/highest-signal first:

- [x] **Fix `TransformerClassifier`** (`mambaTimeSeriesClassificationGMATThrusts.py:102`) before drawing
      conclusions from it — it currently runs a full encoder-decoder `nn.Transformer` with `x` fed as
      both `src` and `tgt`, which is atypical for classification. Switch to encoder-only + CLS-token or
      mean-pool head so it's a fair baseline.
      Done: now `nn.TransformerEncoder` with a learnable CLS token + positional embedding; `nhead` falls
      back automatically if `hidden_size` isn't divisible by 8. Gated behind existing `--transformer` flag.
- [x] **XGBoost / CatBoost** alongside the existing LightGBM path (`use_classic` block, line 428) — same
      flattened `(N, T·D)` input already prepared, cheap to add as a GBDT bake-off.
      Done: `--xgboost` / `--catboost` flags added. Shared `printClassicModelSize`/`validate_classic_classifier`
      helpers added to `qutils/ml/classic/classifier.py` so LightGBM/XGBoost/CatBoost/RF/ExtraTrees all
      reuse one validator instead of duplicating it. CatBoost needs `bootstrap_type="Bernoulli"` to accept
      `subsample`. XGBoost is wired in but untested on this machine — its native lib needs `libomp`
      (`brew install libomp` on macOS).
- [x] **Random Forest / Extra Trees** as a simpler classic-ML floor.
      Done: `--rf` / `--extratrees` flags added, same flattened-feature path and shared validator.
- [x] **1D-CNN / InceptionTime** as a non-SSM, non-Transformer deep baseline — commonly competitive at
      this sequence length and cheaper to train than the Transformer.
      Done: `--cnn` flag adds an `InceptionTimeClassifier` (stacked multi-kernel inception modules,
      residual connections every 3 modules, GroupNorm instead of BatchNorm so training is robust to a
      size-1 trailing batch, global average pool + linear head).
- [ ] **TSFresh / catch22 feature extraction** feeding into a classic classifier — gives an
      interpretable-features alternative to MiniRocket's random-kernel features, useful for a cleaner
      SHAP story.
- [ ] **Extend `--shap` to LightGBM and MiniRocket** (currently only wired for LSTM/Mamba in `main()`)
      so feature attribution is comparable across the full classifier roster, not just the two deep
      sequence models.

## 3. Data regime extensions

- [ ] **Systematic combined-regime sweep**: `orbitType`/`--test` are plain path segments, so
      `combined/leo-meo-geo`, `combined/vleo-heo-meo-geo`, etc. (already present under
      `GMAT-Thrust-Data/data/classification/combined/`) work with existing flags today. Run a full
      train-on-combined vs. train-on-single-regime matrix, evaluated cross-regime, to characterize
      generalization — this doesn't require new code, just systematic invocation.
- [ ] **`--propMin` sweep** (10/30/100 min variants already exist) crossed against classifier choice —
      quantify where Mamba's advantage over LSTM grows with sequence length, rather than relying on
      single-length anecdotal comparisons.
- [ ] **Class balance audit**: confirm the 4 classes are generated in equal proportion; if not,
      class-weighted loss (currently `class_weights=None` in `trainMLP`, unweighted elsewhere) may
      matter more than architecture or frame choice.
- [ ] **Cross-frame x cross-regime interaction**: once ROE/Delaunay/Poincaré are wired in, re-run the
      combined-regime sweep per frame — ROE's chief-orbit linearization assumption in particular may
      behave differently across LEO/VLEO (fast, drag-perturbed) vs. GEO/HEO (slow, third-body-perturbed)
      regimes, so don't assume a frame ranking from one regime transfers to another.

## 4. Suggested sequencing

1. Implement and sanity-check ROE and Delaunay/Poincaré conversions in isolation (no-thrust-vs-no-thrust
   ROE ≈ 0 check; Delaunay/Poincaré round-trip against `OE2ECI`) before wiring into training.
2. Wire both new frames into `prepareThrustClassificationDatasets` + new CLI flags; re-run existing
   LSTM/Mamba baselines on the new frames to get first-pass accuracy/SHAP comparisons against
   ECI/OE/equinoctial results already collected.
3. Wire in the already-built Mamba variants (`mambaAtt`, `mamba_kda_model`) and the fixed
   encoder-only Transformer so the classifier roster is on equal footing across all frames.
4. Run the combined-regime and `propMin` sweeps across the full frame x classifier grid.
5. Add the remaining classic-ML baselines (XGBoost/CatBoost/RandomForest/InceptionTime/TSFresh) as
   supplementary comparison points, and extend SHAP coverage to LightGBM/MiniRocket.
