# Extended Study Plan: Coordinate Frames, Classifiers, Data Regimes

Scope note: this plan covers extensions to
[scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py](../scripts/two_body/mambaTimeSeriesClassificationGMATThrusts.py)
and its dependency `qutils.ml.classifer.prepareThrustClassificationDatasets`. Nothing below exists yet
in `qutils` unless noted otherwise — `qutils/orbital.py` currently only provides `ECI2OE`/`OE2ECI` and
energy helpers, so both frame additions below are new implementation work, not flag flips.

## 1. Coordinate frames for this study: ROE, Delaunay/Poincaré, and ground-station-realistic sensor frames

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

### 1.3 Ground-station-realistic sensor frames (AER + range-rate, RA/Dec)

**Motivation**: ROE and Delaunay/Poincaré (1.1/1.2) are still *state*-frames — they assume perfect
inertial position/velocity knowledge. This track instead represents what a real ground tracking
system would actually output, split by what's physically realistic per orbit regime rather than one
frame for everything:

- **AER + range-rate** (`Az, El, Range, dAz, dEl, dRange` — 6 channels) for `leo`/`vleo`/`meo`. Radar
  power falls off as `1/R⁴`, so ground radar realistically only reaches these regimes — this is the
  frame a radar-tracking ground station would actually produce.
- **RA/Dec, angles-only** (`RA, Dec, dRA, dDec` — 4 channels, **no range/range-rate**) for `geo`/`heo`.
  Real optical SSA sensors (GEODSS-class telescopes) report a bearing only for deep-space objects — a
  single optical observation genuinely doesn't carry range information at GEO distances. Forcing a
  range channel onto this regime would misrepresent what a real sensor gives you.

**Status: base implementation built and validated on a small test batch.** Lives in
`GMAT-Thrust-Data/python/generateSpacecraftThrustOptGroundStation.py` (a ground-station-enabled
sibling of `generateSpacecraftThrustOpt.py`), plus a standalone diagnostic plotter at
`GMAT-Thrust-Data/python/testGroundStationReconstruction.py`.

**Design fork — resolved**: went with the GMAT-native path (`GroundStation` + `CoordinateSystem` +
`CoordinateConverter`) rather than a from-scratch Python GMST implementation, so GMAT owns the
site-rotation math rather than a hand-rolled reimplementation.

**Shared infrastructure**:
- [x] Ground-station site definitions. Two concrete sites, hardcoded constants in the script (not yet
      config-driven): **Orlando, FL** (`28.5383°N, 81.3792°W`, ~sea level — Eastern Range/Cape Canaveral
      vicinity, radar-realistic, source for AER) and **New Mexico** (`33.9756°N, 107.1809°W`, ~3.23 km alt
      — Socorro/Magdalena Ridge vicinity, optical-realistic, source for RA/Dec).
      Still open: move `groundStationLat`/`Lon`/`Alt` into `configs/config.yaml` per the original plan,
      matching the `parse_config()` pattern used for `lowerAlt`/`chemThrust`/etc.
- [ ] Absolute-epoch handling: currently a **single fixed epoch shared by every system**
      (`20 Jul 2020 12:00:00 UTC`, matching the spacecraft's existing `Epoch` field) — the "preferred"
      per-system epoch randomization from the original plan is **not yet done**. Still open, and worth
      prioritizing before generating a real training batch, since right now every system in a batch sees
      the same site-relative geometry pattern.
- [x] `CoordinateConverter`-based conversion helpers, implemented directly in the generation script
      rather than as `qutils/orbital.py` additions (deviates from the original plan's proposed file
      location) — `eciStateToAER(state6, epoch)` and `eciStateToRADEC(state6, epoch)`, both built on a
      shared `_convert(epoch, state6, destCS)` wrapper around `CoordinateConverter.Convert`.
- [x] Real GMAT Python API gotchas hit and fixed while building this — worth preserving since they'll
      bite again on any future GMAT-API coordinate-system work:
      1. `CoordinateSystem.SetField("Axes", ...)` after a bare `gmat.Construct("CoordinateSystem", name)`
         silently leaves the `AxisSystem` sub-object unbuilt (`"Cannot initialize NULL axes"` at
         `gmat.Initialize()`). Fix: pass origin/axes directly to `gmat.Construct("CoordinateSystem", name,
         origin, axes)`.
      2. `CoordinateConverter.Convert` is overloaded 18 ways (`A1Mjd`/`Real`/`GmatTime` epoch types ×
         `Rvector`/raw-array state types × boolean-arg-count variants). `gmat.Rvector6` objects didn't
         cleanly match any overload in this SWIG build; plain Python lists/floats do
         (`Convert(Real, Real*, CoordinateSystem*, Real*, CoordinateSystem*)`).
      3. Fetching the source ECI coordinate system via
         `earthorb.GetRefObject(gmat.COORDINATE_SYSTEM, "EarthMJ2000Eq")` doesn't reliably return a
         properly-typed `CoordinateSystem*` for `Convert` to match against. Fix: construct it explicitly
         (`gmat.Construct("CoordinateSystem", "EciCS", "Earth", "MJ2000Eq")`), same as the two site frames.
      4. This `Convert` overload returns a `bool` success flag and mutates the passed-in output list in
         place — it does not return the converted state.

**AER + range-rate (leo/vleo/meo, from Orlando)**:
- [x] `Az = atan2(y,x)`, `El = asin(z/r)`, `Range = |ρ|` computed from GMAT's own `Topocentric`-axes
      conversion (not a hand-rolled SEZ rotation). `Range-rate = (ρ·ρ̇)/|ρ|` computed exactly from the
      converted velocity, not finite-differenced.
- [x] `dAz`/`dEl` computed via `np.gradient` in a post-loop pass, with `np.unwrap` applied first to avoid
      spurious spikes at the ±π wrap.
- [ ] Visibility/elevation-mask handling — **confirmed still needed, not a hypothetical**: the first test
      batch (3 systems, 30 min) showed the satellite below Orlando's horizon for the entire window
      (`El` between −86° and −30°). Expected given no masking exists yet, but it means this data isn't
      yet usable for training as-is — next concrete step for this frame.

**RA/Dec (geo/heo, from New Mexico)**:
- [x] `RA = atan2(y,x)`, `Dec = asin(z/r)` computed from GMAT's `MJ2000Eq`-axes-at-site conversion, no
      range channel. `dRA`/`dDec` via the same `np.unwrap` + `np.gradient` post-loop pass as AER.
- [ ] Window-length-vs-signal check (GEO period vs. propagation window) — not yet run against real
      GEO/HEO data; the validated test batch above was LEO/VLEO-regime only.
- [ ] HEO visibility/masking — not yet investigated.

**Validation — done, on a LEO/VLEO test batch (3 systems, 30 min)**:
- [x] Geometric-bound check: AER `Range` must satisfy `|r_sat| − R_earth ≤ Range ≤ |r_sat| + R_earth`
      for any ground site. Checked all 90 points — zero violations.
- [x] Independent cross-check: computed Orlando's ECI position from scratch (spherical Earth, algebraic
      GMST, no nutation) and compared the resulting chord distance to GMAT's own AER `Range` output —
      matched within 0.23% (18 km on ~7800 km), fully explained by the independent check's simplified
      Earth model. Also confirms the `Location2 = 360 − 81.3792` (0–360°E) longitude convention guess was
      correct — a wrong sign/range convention would have produced an error far larger than 0.23%.
- [x] RA/Dec parallax-bound check: topocentric vs. geocentric RA/Dec difference stayed under the
      theoretical max parallax angle for the observed satellite ranges, and scaled with proximity as
      expected.
- [x] Rate-channel sanity: `dAz`/`dEl`/`dRA`/`dDec` all smooth, no spurious spikes — the `unwrap` step is
      working.
- [x] Visual/diagnostic plotting script (`testGroundStationReconstruction.py`, standalone, no GMAT
      dependency) — reconstructs a real lat/lon ground track from the ECI truth (via an independent
      GMST implementation), and renders AER/RA-Dec as sensor-native views (Orlando sky plot, New Mexico
      RA/Dec trace) rather than attempting an under-determined or unverified-axis-convention inversion
      back to a literal ground track. Confirmed smooth, continuous, artifact-free tracks on the test
      batch, consistent with the numeric checks above.
- [ ] **Not yet validated**: the above all happened in the "satellite below horizon" regime (`El < 0`
      for the entire test batch) — the `El > 0` / above-horizon case hasn't been visually or numerically
      confirmed yet. Needs a larger batch or geometry more likely to produce an Orlando overhead pass.

**Channel-width mismatch**: AER (6 ch) and RA/Dec (4 ch) can't be concatenated into one tensor for the
existing `combined/` multi-regime datasets without padding or a mask channel. Keep AER-regime and
RA/Dec-regime runs **separate** for this sensor-realism comparison rather than forcing a unified
combined dataset — cross-regime generalization under mismatched channel counts is a separate design
problem, not part of this ablation.

**Noise model**: not yet implemented. Still planned — replace the existing Cartesian `--noise`/
`--velNoise` with frame-appropriate, physically scaled noise (arcsecond-level angular noise for RA/Dec,
arcminute-level angular + meter-level range + cm/s-level range-rate noise for AER).

**CLI wiring (generation side) — current state differs from the original plan**: the base implementation
computes **both** AER (from Orlando) and RA/Dec (from New Mexico) unconditionally for every `propType`
run, regardless of `orbit_type`, rather than auto-selecting one frame by regime. Simpler for an initial
validation pass; downstream code decides which of `aerArray*.npz`/`radecArray*.npz` is "the" realistic
one for a given regime. The original plan's `--orbit`-based auto-select (and `--forceFrame` override) is
still worth doing once this moves beyond small test batches, to avoid generating and storing an unused
frame's data at full `num_runs` scale.

### 1.4 Consuming AER/RA-Dec in the classification pipeline — done

`mambaTimeSeriesClassificationGMATThrusts.py` now has a `--frame {eci,aer,radec}` flag (default `eci`,
fully backward-compatible — no behavior change unless passed).

- [x] `loadGroundStationDataset()` added to the classification script (not `qutils`, since this frame
      support is still local/experimental) — loads `{aerArray,radecArray}{Chemical,Electric,ImpBurn,
      NoThrust}.npz` from the same per-orbit data directory `prepareThrustClassificationDatasets` uses
      for `statesArray*.npz`, and mirrors that function's exact labeling / IC-group train-val-test split
      / `DataLoader` construction. Every one of the 12 existing classifiers (LSTM/Mamba/Transformer/CNN/
      LightGBM/XGBoost/CatBoost/RF/ExtraTrees/1-NN/MiniRocket/MLP) already infers `input_size` generically
      from `train_data.shape[2]`, so all of them work against the new frames with no further changes.
- [x] Incompatible-flag guarding: `--OE`/`--energy`/`--noise` are ECI/OE-specific (Cartesian pos/vel
      noise doesn't map onto Az/El/Range or RA/Dec units) — disabled with a `[warning]` when `--frame`
      isn't `eci`. `--norm` is repurposed to mean per-channel z-score normalization (fit on the training
      split) for the new frames, with a `[note]` printed so this isn't a silent behavior difference.
- [x] `--mlp` (PCA+Hankel-pooled features) is scoped to `--frame eci` only for now — prints a skip
      message rather than silently doing something wrong; reimplementing PCA/Hankel pooling for the new
      frames wasn't part of this pass.
- [x] SHAP feature names (`feat_names`) and log/SHAP-directory naming (`strAdd`, `shap_dir_*`) updated to
      be frame-aware (`['Az','El','Range','dAz','dEl','dRange']` / `['RA','Dec','dRA','dDec']`).
- [x] Validated by actually running the script (not just syntax-checking) against a synthetic 4-class
      dataset built from the real `aerArrayNoThrust.npz`/`radecArrayNoThrust.npz` example files — confirmed
      correct data-location logging, correct `input_size` inference (6 for AER, 4 for RA/Dec, different
      Mamba param counts as expected), the warning/note messages firing correctly, the `--mlp` skip
      message, and a full train → validate → classification-report → confusion-matrix pipeline completing
      cleanly for both new frames.

### 1.5 Running AER/RA-Dec sweeps at scale — scripted, not yet run for real

- [x] `generateThrustClass_aer.sh` / `generateThrustClass_radec.sh` — mirror `generateThrustClass.sh`'s
      dataset sweep (vleo/leo/geo in-distribution + leo→vleo and combined/leo-meo-geo→vleo out-of-
      distribution, 10/30/100 min, `--systems 800 --testSys 800 --train_ratio 0.2`), with `--frame aer`/
      `--frame radec` on every invocation instead of the cart/OE split (AER/RA-Dec are each a single
      frame, so no OE sub-loop), and `--noise`/`--mlp` dropped since both are no-ops for these frames.
- [x] `generateLaTeXTablesColor_aer.sh` / `_radec.sh` — added since neither existed; use `--include-cart`
      instead of `--combine-features`, since `generateLatexTableCompact.py`'s `is_oe_log()` heuristic
      (looks for an "OE" token in the log stem) puts every AER/RA-Dec row in its "Cartesian" bucket, and
      `--include-cart` is what actually emits a populated table instead of the near-random-Cartesian
      summary paragraph the default path is designed for.
- [x] `runClassGen.sh` updated to insert the AER and RA/Dec phases (each: generate → tables → wipe)
      between the existing cart/OE phase and the energy phase. This wipe-between-phases (`removeClassLogs.sh`)
      is load-bearing, not cosmetic: `displayLogData.py --group-dir` recursively globs every `*.log`
      under a directory with **no filtering by frame** — `--group-name` only controls the output CSV
      filename, not which logs get included as input. Considered adding a `--log-pattern` filter to
      `displayLogData.py` to make phases independent of run order, but the repo already has this exact
      precedent (the existing cart/OE-vs-energy split relies on the same wipe-between-phases pattern), so
      kept it consistent rather than introducing a second mechanism.
- [ ] **Not yet run**: still blocked on the same prerequisite as the rest of section 1.3 — real
      800-system AER/RA-Dec data doesn't exist yet for any orbit/propMin combination, only the small
      validation batch. `bash -n` syntax-checked all five touched/new shell scripts; none have been
      executed end-to-end.

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
- [x] **`displayLogData.py`/LaTeX-table pipeline updated for the full 12-model roster** — tested end-to-end
      against a real log (not just reasoned about abstractly) and found two real bugs, both fixed:
      1. `RE_PARAMS` only matched digits, so `Total parameters: NaN` (printed by 6 of the 12 models —
         LightGBM/XGBoost/CatBoost/RF/ExtraTrees/1-NN, since a param count isn't meaningful for them) was
         silently coerced to `0` in the summary CSV, misreporting them as zero-parameter models in any
         accuracy-vs-size comparison. Fixed: `Summary.params` is now `float` (was `int`) so it can hold
         `NaN`, and the regex widened to match it.
      2. `normalize_models()` in both `generateLatexTable.py` and `generateLatexTableCompact.py` had a
         stale `"Decision Trees" → "DT"` mapping — the model's printed name changed to `"Decision Trees
         (LightGBM)"` once XGBoost/CatBoost were added for disambiguation, so the abbreviation silently
         stopped applying, and none of the 6 new classifiers had abbreviations at all. Fixed with
         `LightGBM`/`RF`/`ET`/`1-NN`/`CNN`, keeping the old `"Decision Trees"` entry as a legacy alias for
         pre-rename logs.

## 3. Data regime extensions

- [ ] **Systematic combined-regime sweep**: `orbitType`/`--test` are plain path segments, so
      `combined/leo-meo-geo`, `combined/vleo-heo-meo-geo`, etc. (already present under
      `GMAT-Thrust-Data/data/classification/combined/`) work with existing flags today. Run a full
      train-on-combined vs. train-on-single-regime matrix, evaluated cross-regime, to characterize
      generalization — thisn  yyn doesn't require new code, just systematic invocation.
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
5. ~~Add the remaining classic-ML baselines~~ — **mostly done**: XGBoost/CatBoost/RandomForest/
   InceptionTime all implemented and validated (section 2). TSFresh/catch22 still open. SHAP coverage
   still not extended to LightGBM/MiniRocket.
6. ~~Resolve the AER/RA-Dec generation-path fork~~ — **done**: went GMAT-native
   (`generateSpacecraftThrustOptGroundStation.py`), Orlando (AER) + New Mexico (RA/Dec) sites built and
   round-trip/cross-check validated on a small LEO/VLEO test batch (geometric bound check, independent
   GMST cross-check within 0.23%, RA/Dec parallax-bound check, and a standalone diagnostic plotting
   script — all passing). See section 1.3 for the detailed status and the GMAT API gotchas hit along
   the way.
7. Remaining before this frame is ready for a real training run: (a) visibility/elevation-mask handling
   — confirmed necessary, not hypothetical, since the validated test batch never had the satellite above
   Orlando's horizon; (b) per-system epoch randomization, currently a single fixed epoch shared by every
   system; (c) move the two site definitions into `configs/config.yaml` instead of hardcoded constants;
   (d) validate the `El > 0` / above-horizon case, only the below-horizon regime has been checked so far;
   (e) run the RA/Dec window-length-vs-signal check against real GEO/HEO data, only LEO/VLEO validated so
   far; (f) add the frame-appropriate sensor noise model; (g) wire the `--orbit`-based auto-select
   (currently generates both frames unconditionally) once moving beyond small test batches.
8. ~~Wire AER/RA-Dec into the classification pipeline and sweep scripts~~ — **done**: `--frame
   {eci,aer,radec}` added to `mambaTimeSeriesClassificationGMATThrusts.py` (section 1.4), and
   `generateThrustClass_aer.sh`/`_radec.sh` + matching LaTeX-table scripts + updated `runClassGen.sh`
   added (section 1.5). All validated on synthetic/small data and syntax-checked respectively. Still
   blocked on the same prerequisite as everything else in this track: real 800-system AER/RA-Dec data
   doesn't exist yet, so none of this has been run end-to-end for real.
