# In-Sequence Classification: Model and Simulation Recommendations

This document summarizes the main model gaps and simulation improvements for the per-timestep
thrust-classification study implemented in
[`scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py`](../scripts/two_body/mambaTimeSeriesSeqClassificationGMATThrusts.py)
and swept by [`runManuscriptSweep.sh`](../runManuscriptSweep.sh).

## Current model coverage

The manuscript sweep already spans most major sequence-model families:

| Family | Current model | Role |
|---|---|---|
| Recurrent | Bidirectional LSTM | Learned local and long-range temporal context |
| State space | Bidirectional Mamba | Selective state-space sequence model |
| Attention | Transformer encoder | Global, non-causal temporal context |
| Convolutional | InceptionTime | Multi-scale temporal convolutions |
| Random kernels | MiniRocket + ridge heads | Lightweight non-neural baseline at per-timestep resolution |
| Tree ensemble | LightGBM | Classic supervised baseline on engineered temporal rows |

This is already broad architecture coverage. The most useful additions are models that introduce a
new assumption about event structure, rather than another generic feature encoder.

## Important missing models

### 1. Dilated temporal convolutional network

A residual TCN is the strongest missing neural baseline. Unlike the current InceptionTime model,
a TCN uses exponentially increasing dilation to obtain a long receptive field while preserving
per-timestep output resolution. It directly tests whether weak, sustained electric-thrust events
benefit from a sequence-labeling architecture built around temporal persistence.

Start with a single-stage residual TCN. Add an MS-TCN refinement stack only if the existing models
show substantial event fragmentation or over-segmentation. MS-TCN combines frame classification
with a temporal smoothing objective specifically for sequence segmentation.

References:

- Bai, Kolter, and Koltun, [*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*](https://arxiv.org/abs/1803.01271).
- Farha and Gall, [*MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*](https://arxiv.org/abs/1903.01945).

### 2. Structured temporal decoder

A linear-chain conditional random field (CRF), hidden semi-Markov model, or duration-aware decoder
can be placed on top of an existing model's logits. This tests whether remaining errors come from
weak per-frame evidence or from temporally inconsistent decoding.

The binary cascade detector is the best first target because it has only two states: coast and
thrust. A duration-aware decoder can learn burn-length distributions from the training set while
still allowing one-sample impulsive events. Hard-coded minimum-duration filtering should be
avoided because it can erase valid impulses.

Reference: Lafferty, McCallum, and Pereira,
[*Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/LaffertyMcCallumPereira2001.pdf).

### 3. Physics-based maneuver detector

The manuscript needs a non-learned orbital-dynamics baseline. The simplest useful version is a
threshold or CUSUM detector applied to a coast-model innovation or acceleration residual. A more
complete version uses an EKF or UKF and an interacting multiple-model filter with coast and
maneuver hypotheses.

This comparison answers whether the learned models extract more information than a conventional
dynamics-based detector. It also gives a physically interpretable false-alarm threshold.

Reference: Lee and Alfriend,
[*Tracking maneuvering spacecraft with filter-through approaches using interacting multiple models*](https://www.sciencedirect.com/science/article/pii/S0094576515001903).

### 4. Optional 1D U-Net

A temporal U-Net can combine coarse sequence context with precise boundary localization through
downsampling and skip connections. It is worth adding only if onset and offset errors remain large
after testing a TCN or structured decoder.

## Recommended simulation improvements

### Separate burn duration from observation-window length

The current configurations scale chemical and electric burn durations with the propagation window.
For example, the 30-minute LEO configuration uses mean durations of 6 minutes for chemical thrust
and 24 minutes for electric thrust. Burns are normally placed entirely inside the observation
window. A classifier can therefore exploit duration and event-position shortcuts.

Generate longer parent trajectories and sample observation windows from them. The resulting set
should contain:

- Windows with no burn, one burn, and multiple burns.
- Burns that start before the window or continue after it.
- Independent distributions for observation length, burn duration, and coast interval.
- Overlapping chemical and electric duration distributions.

Split by parent trajectory before creating crops so related windows cannot appear in both training
and test sets.

### Simulate realistic observations

The current optional noise transform adds independent Gaussian noise directly to Cartesian
position and velocity. This is useful for a controlled sensitivity test, but real orbit estimates
contain correlated, time-dependent errors.

Use a staged observation study:

1. Add correlated position and velocity errors with a specified covariance.
2. Add missing observations, irregular cadence, and short data gaps.
3. Simulate range, range-rate, angle, or GNSS measurements.
4. Run orbit determination and classify the estimated state and its covariance or innovations.

Energy-rate and acceleration-residual features should be re-evaluated at each stage because
numerical differentiation can amplify observation noise.

### Vary thrust direction

The energy residual observes the component of unmodeled acceleration parallel to velocity:

\[
\frac{dE_{\mathrm{model}}}{dt}
= \mathbf{v}^{\mathsf T}\mathbf{a}_{\mathrm{unmodeled}}.
\]

A radial or cross-track burn can therefore have substantial acceleration while producing a small
energy-rate signal. Simulate along-track, radial, cross-track, and mixed-direction burns.

A complementary model input is a vector velocity innovation in the radial-transverse-normal frame:

\[
\mathbf{r}_{v,k}
= \mathbf{v}_{\mathrm{observed},k}
- \widehat{\mathbf{v}}_{k\mid k-1,\mathrm{coast}}.
\]

The coast prediction should be propagated over the actual observation interval. This retains the
direction of the unexplained motion and avoids treating a coarse finite difference as an
instantaneous acceleration.

### Introduce force-model mismatch

The truth simulation already includes high-degree gravity, drag, and solar-radiation pressure.
The next useful test is controlled mismatch between the truth model and the detector's assumed
model. Vary:

- Epoch and solar/geomagnetic conditions.
- Atmospheric-density realization and model family.
- Drag and SRP areas, coefficients, and spacecraft mass.
- Gravity truncation and coefficient uncertainty.
- Electric-thruster power availability, duty cycle, and thrust variability.

Report performance as the mismatch grows. This distinguishes thrust detection from memorization of
one environmental-force realization.

### Use saved force components for diagnostics

The variable-thrust generator already saves drag, SRP, J2, higher-order gravity, thrust, and mass
histories. These truth quantities should support diagnostics and controlled binning, but should not
be supplied as classifier inputs when they would be unavailable operationally.

Useful plots include electric-thrust recall and false-alarm rate against:

- Actual thrust acceleration.
- Background acceleration and force-model error.
- Observation uncertainty.
- Burn duration and thrust direction.
- Ratio of thrust acceleration to the residual background floor.

The generator should also verify that every commanded finite-burn label corresponds to nonzero
applied thrust.

### Separate maneuver detection from propulsion identification

Acceleration magnitude is not a unique propulsion identifier. Thruster force, spacecraft mass,
direction, and duty cycle can make chemical and electric acceleration distributions overlap.
Similarly, impulsive describes a maneuver approximation while chemical describes propulsion
technology. At a 60-second cadence, a short finite chemical burn can be observationally equivalent
to an impulse.

The simulation should include overlapping acceleration distributions and short finite burns. The
manuscript should report this identifiability limit and distinguish these tasks:

1. Detect whether an unmodeled maneuver occurred.
2. Estimate its start, end, and direction.
3. Classify the propulsion or maneuver type when the observations contain enough information.

## Recommended next experiment

Use the current best-performing model, one residual TCN, and one physics-based innovation detector.
Evaluate all three on a dataset with randomized window crops, correlated observation errors, and
varied thrust directions. Keep the current joint and cascade formulations.

Report frame-level macro precision, recall, and F1 together with event-level recall, false alarms
per hour, onset and offset error, intersection-over-union, and performance as a function of
thrust-to-background acceleration ratio. This experiment will reveal whether the remaining stage-1
difficulty is caused by insufficient physical evidence, temporal decoding, or model capacity.
