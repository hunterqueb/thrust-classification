# In-sequence thrust dataset loading, shared by mambaTimeSeriesSeqClassificationGMATThrusts.py
# (per-timestep labels) and mambaTimeSeriesClassificationGMATThrusts.py --seq-data (one label per
# trajectory), so both tasks see identical features, standardization and IC splits. Moved verbatim
# out of the in-sequence script; that script re-exports everything here via `from seqData import *`,
# so `import mambaTimeSeriesSeqClassificationGMATThrusts as M; M.<name>` keeps working.
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from qutils.ml.classifer import apply_noise

__all__ = [
    "RESIDUAL_LADDER_ALL_CHANNEL_NAMES",
    "RESIDUAL_LADDER_LEAN_INDICES",
    "residualLadderChannelNames",
    "residualLadderChannelCount",
    "CLASS_ORDER",
    "JOINT_LABELS",
    "JOINT_CLASS_NAMES",
    "STAGE1_CLASS_NAMES",
    "STAGE2_CLASS_NAMES",
    "_getThrustingTime",
    "_decomposeSinusoids",
    "_applyTransforms",
    "PHYS_LOSS_DT_SECONDS",
    "_computePhysicsResidualTensors",
    "RESIDUAL_LADDER_FLOOR",
    "_computeResidualLadder",
    "_load_and_label",
    "_deriveStage1Stage2",
    "_icGroupSplit",
    "_computeOversamplingWeights",
    "_standardizeSplits",
    "prepareInSequenceThrustClassificationDatasets",
]

# --residual-ladder channel layout, in the order _computeResidualLadder computes them. Used for the
# feature-count printout and for anyone reading a saved feature array back.
RESIDUAL_LADDER_ALL_CHANNEL_NAMES = (
    "log10(|dE_kep/dt| / (|v| g0))",      # rung 0: residual after removing two-body only
    "log10(|dE_J2/dt| / (|v| g0))",       # rung 1: ... after two-body + J2
    "log10(|dE_J2toJ6/dt| / (|v| g0))",   # rung 2: ... after two-body + J2..J6
    "log10(a_J2 / g0)",                   # reference rung height: J2
    "log10(a_J3toJ6 / g0)",               # reference rung height: J3-J6
)

# LEAN (default) keeps only the DEEPEST residual rung plus the two reference rung heights.
#
# Measured on leo/30min-1500, per-frame Electric-vs-NoThrust test AUC over an IC-split
# (1000 train / 500 test ICs), logistic regression / HistGradientBoosting:
#
#   rung 2 alone      0.8175 / 0.8127     three rungs (0,1,2)  0.8168 / 0.8193
#   rung 2 + refs     0.8187 / 0.8274     all five             0.8182 / 0.8301
#
# Stacking three residual rungs instead of one buys ~0.007 AUC, and rungs 1 and 2 are 0.944
# correlated -- they are near-duplicates, because the coasting floor barely moves between them
# (1.02x; the leftover is non-zonal, see the docs). What actually earns its channels is the pair
# of REFERENCE heights: rung 2 alone 0.8127 -> rung 2 + refs 0.8274 under the nonlinear model.
# So the lean set drops the two redundant rungs, not the comparison baselines, for 0.003 AUC.
#
# The multi-rung form is kept behind --residual-ladder-full: rung-to-rung drops are the diagnostic
# that told us the residual floor is not zonal, and that may not hold in another regime or against
# another force model. Caveat: the ablation above is per-frame and memoryless, while the backbones
# here are sequence models -- rung 0 is dominated by the J2 along-track projection, a smooth
# function of orbital phase that a sequence model could in principle use as a phase reference for
# the local floor. Untested; --residual-ladder-full vs. default is the A/B (see runResidLadderAB.sh).
RESIDUAL_LADDER_LEAN_INDICES = (2, 3, 4)


def residualLadderChannelNames(full=False):
    if full:
        return RESIDUAL_LADDER_ALL_CHANNEL_NAMES
    return tuple(RESIDUAL_LADDER_ALL_CHANNEL_NAMES[i] for i in RESIDUAL_LADDER_LEAN_INDICES)


def residualLadderChannelCount(full=False):
    return len(residualLadderChannelNames(full))


# ---------------------------------------------------------------------------
# Class/label conventions shared by data loading, training, and reporting
# ---------------------------------------------------------------------------
CLASS_ORDER = ["Chemical", "Electric", "ImpBurn", "NoThrust"]
JOINT_LABELS = {"NoThrust": 0, "Chemical": 1, "Electric": 2, "ImpBurn": 3}
JOINT_CLASS_NAMES = ["No Thrust", "Chemical", "Electric", "Impulsive"]
STAGE1_CLASS_NAMES = ["No Thrust", "Thrust"]
STAGE2_CLASS_NAMES = ["Chemical", "Electric", "Impulsive"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _getThrustingTime(npz_dict, class_name, N, T, warn_if_missing=True):
    if 'thrustingTime' in npz_dict:
        return npz_dict['thrustingTime']
    if warn_if_missing:
        print(f"[prepareInSequenceThrustClassificationDatasets] WARNING: '{class_name}' has no "
              f"'thrustingTime' key -- defaulting to all-background per-timestep labels. This is "
              f"expected until the upstream GMAT-Thrust-Data dataset fix lands; per-timestep "
              f"labels for {class_name} will be wrong until then.")
    return np.zeros((N, T, 1))


def _decomposeSinusoids(states, num_sinusoids):
    """states: [N,T,C] -> [N,T,C*num_sinusoids]. Every (IC, channel) time series is decomposed
    independently: an FFT along the time axis picks the `num_sinusoids` frequency bins with the
    largest magnitude (excluding the DC bin, which is just that channel's mean over the window, not
    an oscillation), and each selected bin is reconstructed as its own real sinusoid signal
    A*cos(2*pi*m*t/T + phase) of length T. The K reconstructed component signals REPLACE the single
    raw channel, so C channels become C*num_sinusoids -- e.g. 6 ECI channels with num_sinusoids=3
    yields 18, grouped as [chan0_comp0, chan0_comp1, chan0_comp2, chan1_comp0, ...].

    Components are ordered by descending magnitude per (IC, channel), so component 0 is always that
    trajectory's single most dominant oscillation for that channel, component 1 the next, etc.,
    regardless of which absolute frequency bin that turns out to be -- the network is fed a stable
    "1st/2nd/3rd dominant mode" ordering rather than fixed frequency bins that fit some trajectories
    (e.g. NoThrust, near-periodic) and not others (e.g. a burn mid-window)."""
    N, T, C = states.shape
    max_components = T // 2  # non-DC rfft bins: 1..T//2 (T//2+1 bins total incl. DC, and Nyquist if T even)
    if num_sinusoids > max_components:
        print(f"[_decomposeSinusoids] WARNING: requested {num_sinusoids} sinusoids but only "
              f"{max_components} non-DC frequency bins are available for T={T}; clamping.")
        num_sinusoids = max_components

    freqs = np.fft.rfft(states, axis=1)   # [N, T//2+1, C] complex
    mag = np.abs(freqs)
    mag[:, 0, :] = -1.0                   # exclude the DC bin from selection

    order = np.argsort(-mag, axis=1)[:, :num_sinusoids, :]         # [N,K,C] bin indices, descending magnitude
    amp = np.take_along_axis(mag, order, axis=1)                    # [N,K,C]
    phase = np.angle(np.take_along_axis(freqs, order, axis=1))      # [N,K,C]

    # A real signal's rfft coefficients need a factor of 2/T to reconstruct amplitude, except the
    # Nyquist bin (T even only), which has no conjugate partner and needs 1/T.
    nyquist_bin = T // 2 if T % 2 == 0 else -1
    scale = np.where(order == nyquist_bin, 1.0, 2.0) / T            # [N,K,C]

    t = np.arange(T).reshape(1, T, 1, 1)
    m = order.reshape(N, 1, num_sinusoids, C)
    a = (amp * scale).reshape(N, 1, num_sinusoids, C)
    p = phase.reshape(N, 1, num_sinusoids, C)

    components = a * np.cos(2 * np.pi * m * t / T + p)              # [N,T,K,C]
    return components.transpose(0, 1, 3, 2).reshape(N, T, C * num_sinusoids)


def _applyTransforms(states_by_class, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std, usePhysicsLoss=False, useJ2Energy=False, useResidualLadder=False, useResidualLadderFull=False):
    if useNoise:
        for c in CLASS_ORDER:
            states_by_class[c] = apply_noise(states_by_class[c], pos_noise_std, vel_noise_std)

    # orbitalEnergy() is a Cartesian quantity: it reads Y[:,0:3] as position and Y[:,3:6] as
    # velocity. --OE overwrites states_by_class with orbital elements below, so snapshot the raw
    # ECI states first and compute energy from those. Feeding orbital elements to it instead
    # evaluates 0.5*||(Omega,omega,M0)||^2 - mu/||(a,e,i)||, a quantity that correlates only ~0.25
    # with true orbital energy and destroys the whole premise of the feature (energy is ~constant
    # under two-body coasting, so departures from constant are the thrust residual).
    eci_by_class = None
    if useOE and (useEnergy or useEnergyRate):
        eci_by_class = {c: states_by_class[c].copy() for c in CLASS_ORDER}

    # Physics-loss inputs (--physics-loss-weight) and the residual ladder (--residual-ladder) both
    # need raw dimensional (km, km/s) ECI Cartesian state regardless of what --OE/--norm do to
    # states_by_class below -- unlike eci_by_class above, this snapshot is taken unconditionally
    # (not gated on useOE) so it stays correct even under --norm alone. It is taken AFTER
    # apply_noise, so under --noise the ladder measures the residual of the noisy trajectory the
    # model actually sees rather than a clean one the model has no access to.
    phys_eci_by_class = None
    if usePhysicsLoss or useResidualLadder:
        phys_eci_by_class = {c: states_by_class[c].copy() for c in CLASS_ORDER}

    oe_by_class = None
    if useOE:
        from qutils.orbital import ECI2OE
        oe_by_class = {}
        for c in CLASS_ORDER:
            s = states_by_class[c]
            n_ic, T = s.shape[0], s.shape[1]
            oe = np.zeros((n_ic, T, 7))
            for i in range(n_ic):
                for j in range(T):
                    oe[i, j, :] = ECI2OE(s[i, j, 0:3], s[i, j, 3:6])
            if useNorm:
                R = 6378.1363
                oe[:, :, 0] = oe[:, :, 0] / R
            oe_by_class[c] = oe
            states_by_class[c] = oe[:, :, 0:6]
    elif useNorm:
        from qutils.orbital import dim2NonDim6
        for c in CLASS_ORDER:
            s = states_by_class[c]
            for i in range(s.shape[0]):
                s[i, :, :] = dim2NonDim6(s[i, :, :])
            states_by_class[c] = s

    if useEnergy or useEnergyRate:
        from qutils.orbital import orbitalEnergy
        energy_by_class = {}
        rate_by_class = {}
        # norming_energy is a single scalar shared across all 4 classes (taken from the first
        # class processed), recomputed fresh on every _applyTransforms call -- when --test names a
        # different orbit/system count than --orbit, train and test energy (and energy rate) get
        # normalized by DIFFERENT scalars, a pre-existing train/test scale inconsistency under
        # --norm --energy --test <different orbit>, not introduced or fixed here.
        norming_energy = None
        for c in CLASS_ORDER:
            # eci_by_class is set only under --OE; otherwise states_by_class is still Cartesian
            # (dimensional, or non-dimensionalized by dim2NonDim6 under --norm) and is used as-is.
            s = states_by_class[c] if eci_by_class is None else eci_by_class[c]
            n_ic, T = s.shape[0], s.shape[1]
            energy = np.zeros((n_ic, T, 1))
            if useJ2Energy:
                # J2-inclusive specific energy: the Keplerian form below is NOT conserved under
                # J2, so its rate measures J2 potential exchange (~55x the amplitude of a
                # low-thrust electric signature) far more than it measures any thrust. See
                # orbitalEnergyJ2 and scripts/two_body/analyzeThrustSeparability.py. Vectorized
                # over the whole [N,T,6] block, unlike orbitalEnergy's per-row loop.
                from qutils.orbital import orbitalEnergyJ2
                energy[:, :, 0] = orbitalEnergyJ2(s)
            else:
                for i in range(n_ic):
                    energy[i, :, 0] = orbitalEnergy(s[i, :, :])
            if useNorm:
                if norming_energy is None:
                    norming_energy = energy[0, 0, 0]
                energy[:, :, 0] = energy[:, :, 0] / norming_energy
            if useEnergyRate:
                # diff is linear, so computing the rate after normalization above is equivalent to
                # normalizing a raw-energy rate -- order doesn't change the numeric result. prepend
                # gives t=0 a rate of exactly 0 (no prior point) in one call.
                rate_by_class[c] = np.diff(energy, axis=1, prepend=energy[:, :1, :])
            energy_by_class[c] = energy

        extra_channels = {}
        for c in CLASS_ORDER:
            parts = ([energy_by_class[c]] if useEnergy else []) + \
                    ([rate_by_class[c]] if useEnergyRate else [])
            extra_channels[c] = np.concatenate(parts, axis=2)

        # Energy/energy-rate are ADDITIONAL channels, never a replacement. The previous else-branch
        # assigned extra_channels directly, so --energy or --energyRate without --OE silently
        # dropped all six ECI channels and trained on the single energy scalar alone.
        base = oe_by_class if oe_by_class is not None else states_by_class
        for c in CLASS_ORDER:
            states_by_class[c] = np.concatenate((base[c], extra_channels[c]), axis=2)

    if useResidualLadder:
        # Hierarchical residual decomposition (--residual-ladder): 5 additional channels appended
        # to whatever --OE/--energy/--energyRate leave behind, never a replacement (same policy as
        # the energy channels above). Computed from the raw dimensional ECI snapshot, so these
        # channels are identical with and without --OE/--norm -- unlike --energy, which silently
        # changes meaning under --norm (see norming_energy above). See _computeResidualLadder.
        for c in CLASS_ORDER:
            ladder = _computeResidualLadder(phys_eci_by_class[c], full=useResidualLadderFull)
            states_by_class[c] = np.concatenate((states_by_class[c], ladder), axis=2)

    if numSinusoids > 0:
        # Decomposes whatever channels the rest of _applyTransforms leaves behind (raw ECI, OE,
        # and/or the energy/energy-rate channels above) -- runs last so it always sees the final
        # per-timestep feature set, not the pre-OE/pre-energy Cartesian states.
        for c in CLASS_ORDER:
            states_by_class[c] = _decomposeSinusoids(states_by_class[c], numSinusoids)

    return states_by_class, phys_eci_by_class


PHYS_LOSS_DT_SECONDS = 60.0  # GMAT generation scripts' fixed propagation step (see
                              # gmat/scripts/generateSpacecraftThrusts.py / generateSpacecraftEThrusts.py: dt = 60.0)


def _computePhysicsResidualTensors(eci_states, dt_seconds=PHYS_LOSS_DT_SECONDS):
    """eci_states: [N,T,6] raw ECI Cartesian (km, km/s), pre-OE/pre-norm (see phys_eci_by_class in
    _applyTransforms). Returns (h2, h36, g0, residual_accel), each [N,T]:
      g0             = mu/|r|^2          two-body gravitational accel magnitude (km/s^2)
      h2             = j2AccelMag(r)/g0  dimensionless J2-to-two-body ratio (orbit-invariant
                                          formula; VALUE naturally shrinks with altitude)
      h36            = j3to6AccelMag(r)/g0  dimensionless combined-J3-J6-to-two-body ratio
      residual_accel = |dE/dt| / |v|     thrust-accel-scale estimate: a finite-difference energy
                       residual divided by the real sample interval (not left as a raw per-index
                       delta, since this pathway compares against absolute physical accelerations)
                       and converted power->accel via /|v|.

    E here is orbitalEnergyJ2 -- the J2-INCLUSIVE specific energy -- NOT the Keplerian
    v^2/2 - mu/r that --energy/--energyRate use by default.

    mu is fixed to qutils.orbital.MU_EARTH_JGM2 throughout (energy, g0, h2, h36) for internal
    self-consistency of the log-ratio comparison in _load_and_label.
    """
    from qutils.orbital import MU_EARTH_JGM2, orbitalEnergyJ2, twoBodyAccel, j2AccelMag, j3to6AccelMag

    r = eci_states[..., 0:3]                                          # [N,T,3]
    v_mag = np.linalg.norm(eci_states[..., 3:6], axis=-1)             # [N,T]

    energy = orbitalEnergyJ2(eci_states, mu=MU_EARTH_JGM2)            # [N,T]
    energy_rate = np.diff(energy, axis=1, prepend=energy[:, :1]) / dt_seconds
    residual_accel = np.abs(energy_rate) / np.maximum(v_mag, 1e-9)    # [N,T], km/s^2

    g0 = twoBodyAccel(r, mu=MU_EARTH_JGM2)
    h2 = j2AccelMag(r, mu=MU_EARTH_JGM2) / g0
    h36 = j3to6AccelMag(r, mu=MU_EARTH_JGM2) / g0
    return h2, h36, g0, residual_accel


# Floor applied to every dimensionless ratio before the log. The quantities of interest sit around
# 1e-6..1e-3 (see the docstring), so 1e-12 is ~6 decades below anything meaningful: it exists only
# to keep an incidental near-zero energy-rate crossing from becoming a -inf/-30 spike that would
# dominate --standardize's mean/std for the whole channel.
RESIDUAL_LADDER_FLOOR = 1e-12


def _computeResidualLadder(eci_states, dt_seconds=PHYS_LOSS_DT_SECONDS, full=False):
    """Hierarchical residual decomposition. eci_states: [N,T,6] raw ECI Cartesian (km, km/s),
    pre-OE/pre-norm (see phys_eci_by_class in _applyTransforms). Returns [N,T,C] float32, where
    C is 3 for the default lean set (deepest rung + both reference heights) and 5 with full=True
    (every rung). All five are always computed -- the switch only selects which are returned, so
    the two forms are guaranteed bit-identical on the channels they share. See
    RESIDUAL_LADDER_LEAN_INDICES for the ablation behind that default.

    The idea: dE/dt under a given dynamics truncation measures exactly those accelerations the
    truncation leaves out. So evaluating the energy residual at successively richer truncations
    gives a ladder of noise floors, and a thrust reveals itself at the rung where it first stands
    above the floor:

      rung 0 (two-body):    floor is J2 potential exchange, ~2.7e-6 km/s^2 in LEO
      rung 1 (+J2):         floor drops ~55x to ~9.9e-8 km/s^2 -- J3-J6, tesserals, drag
      rung 2 (+J2..J6):     floor is whatever is left (tesseral/sectoral terms, drag, noise)

    That is a direct encoding of the domain fact this feature exists for: chemical thrust is
    O(J2) so it is visible at every rung, while electric thrust is O(J3-J6) so it only clears the
    floor at rungs 1-2. Rather than asserting that as a loss penalty (--physics-loss-weight,
    which is discarded at inference), it hands the model the measurements the assertion is about
    and lets it use them at every forward pass.

    All five channels are dimensionless (divided by g0 = mu/r^2) and log10-scaled:

      * dimensionless, so the same channel means the same thing in LEO and GEO -- the raw
        accelerations differ by ~3 decades between regimes but the ratios do not, which is what
        makes this regime-agnostic in the same sense as the --physics-loss-weight gate.
      * log10, because the prior is about ORDERS OF MAGNITUDE. In the log domain "the residual
        sits at the J2 rung" is a subtraction, so a linear layer can express it; in the linear
        domain it is a ratio spanning 4+ decades that a standardized channel cannot resolve.

    Channels 3-4 are the reference rung heights a_J2/g0 and a_J3toJ6/g0. They are deterministic
    functions of r and so carry no new information in principle, but they are what the residual
    rungs must be COMPARED against, and in log space channel_k - channel_3 is exactly "how many
    decades is the residual above the J2 rung" -- the comparison the domain fact is stated in.

    mu/Re/J_n are fixed to the qutils JGM2 constants throughout, matching
    _computePhysicsResidualTensors, so the two pathways cannot disagree about rung heights.
    """
    from qutils.orbital import (MU_EARTH_JGM2, orbitalEnergyZonal, twoBodyAccel,
                                j2AccelMag, j3to6AccelMag)

    r = eci_states[..., 0:3]                                          # [N,T,3]
    v_mag = np.linalg.norm(eci_states[..., 3:6], axis=-1)             # [N,T]
    g0 = twoBodyAccel(r, mu=MU_EARTH_JGM2)                            # [N,T]
    # |dE/dt| / |v| converts specific power to a thrust-acceleration scale; the further /g0 makes
    # it dimensionless. Folded into one denominator so it is clamped once.
    denom = np.maximum(v_mag * g0, 1e-30)

    T = eci_states.shape[1]
    rungs = []
    for degrees in ((), (2,), (2, 3, 4, 5, 6)):
        E = orbitalEnergyZonal(eci_states, degrees=degrees, mu=MU_EARTH_JGM2)   # [N,T]
        rate = np.zeros_like(E)
        if T > 1:
            rate[:, 1:] = np.diff(E, axis=1) / dt_seconds
            # t=0 has no prior sample. Copy t=1's rate rather than np.diff's usual prepend-zero:
            # an exact 0 here would floor to log10(1e-12) and put a -12 outlier in every single
            # trajectory's first frame, skewing --standardize's per-channel statistics.
            rate[:, 0] = rate[:, 1]
        rungs.append(np.abs(rate) / denom)

    rungs.append(j2AccelMag(r, mu=MU_EARTH_JGM2) / g0)
    rungs.append(j3to6AccelMag(r, mu=MU_EARTH_JGM2) / g0)

    ladder = np.stack(rungs, axis=-1)                                 # [N,T,5]
    if not full:
        ladder = ladder[..., list(RESIDUAL_LADDER_LEAN_INDICES)]      # [N,T,3]
    return np.log10(np.maximum(ladder, RESIDUAL_LADDER_FLOOR)).astype(np.float32)


def _load_and_label(loc, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std, usePhysicsLoss=False, useJ2Energy=False, useResidualLadder=False, useResidualLadderFull=False):
    states_by_class = {}
    thrusting_by_class = {}
    for class_name in CLASS_ORDER:
        npz = np.load(f"{loc}/statesArray{class_name}.npz")
        states = npz[f"statesArray{class_name}"]
        N, T = states.shape[0], states.shape[1]
        tt = _getThrustingTime(npz, class_name, N, T, warn_if_missing=(class_name != "NoThrust"))
        states_by_class[class_name] = states
        thrusting_by_class[class_name] = tt

    n_ic_per_class = [states_by_class[c].shape[0] for c in CLASS_ORDER]
    T_per_class = [states_by_class[c].shape[1] for c in CLASS_ORDER]
    assert len(set(T_per_class)) == 1, f"Timestep count mismatch across classes: {dict(zip(CLASS_ORDER, T_per_class))}"

    states_by_class, phys_eci_by_class = _applyTransforms(
        states_by_class, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids,
        pos_noise_std, vel_noise_std, usePhysicsLoss, useJ2Energy, useResidualLadder,
        useResidualLadderFull)

    states_cat = np.concatenate([states_by_class[c] for c in CLASS_ORDER], axis=0)

    joint_labels_list = [thrusting_by_class[c].squeeze(-1).astype(np.int64) * JOINT_LABELS[c] for c in CLASS_ORDER]
    y_joint = np.concatenate(joint_labels_list, axis=0)

    phys_target_ce, phys_gate_ce = None, None
    if usePhysicsLoss:
        h2_parts, h36_parts, g0_parts, resid_parts = [], [], [], []
        for c in CLASS_ORDER:
            h2_c, h36_c, g0_c, resid_c = _computePhysicsResidualTensors(phys_eci_by_class[c])
            h2_parts.append(h2_c); h36_parts.append(h36_c)
            g0_parts.append(g0_c); resid_parts.append(resid_c)
        h2 = np.concatenate(h2_parts, axis=0)
        h36 = np.concatenate(h36_parts, axis=0)
        g0 = np.concatenate(g0_parts, axis=0)
        residual_accel = np.concatenate(resid_parts, axis=0)

        log_resid = np.log(np.maximum(residual_accel, 1e-30))
        log_aJ2 = np.log(np.maximum(h2 * g0, 1e-30))
        log_aJ36 = np.log(np.maximum(h36 * g0, 1e-30))
        # Chemical-vs-Electric term (--mode joint/stage2): physics-only pseudo-target, independent
        # of the true label -- does the empirical thrust-accel-scale estimate look closer (in
        # log-space) to the J2 scale (Chemical-like, target=1) or the combined J3-J6 scale
        # (Electric-like, target=0)?
        pseudo_target_ce = (np.abs(log_resid - log_aJ2) < np.abs(log_resid - log_aJ36)).astype(np.float32)

        # Only Chemical/Electric-labeled frames get any physics-loss contribution; NoThrust and
        # Impulsive are gated out entirely (gate=0 there).
        is_chem_or_elec = (y_joint == JOINT_LABELS["Chemical"]) | (y_joint == JOINT_LABELS["Electric"])
        gate_ce = np.where(is_chem_or_elec, (h2 + h36).astype(np.float32), 0.0).astype(np.float32)

        if is_chem_or_elec.any():
            print(f"[physics-loss] chem/elec gate (h2+h36) over {int(is_chem_or_elec.sum())} "
                  f"Chemical/Electric frames: mean={gate_ce[is_chem_or_elec].mean():.3e}, "
                  f"max={gate_ce.max():.3e} (naturally shrinks toward 0 at higher altitude -- LEO "
                  f"~1e-3 vs. GEO several orders of magnitude smaller, since J2/J3-J6 fall off as "
                  f"~1/r^4..1/r^8)")

        # A Thrust-vs-NoThrust ('detect') term used to live here, targeting the NoThrust/Electric
        # boundary via 'is residual_accel above the J3-J6 floor'. It was REMOVED after
        # scripts/two_body/analyzeThrustSeparability.py measured the quantities it assumed:
        #   - measured residual_accel on pure COASTING frames is ~2.7e-6 km/s^2, while a_J3-J6 is
        #     ~3.8e-8 -- so that pseudo-target was true on essentially every frame, teaching the
        #     model "everything is thrust".
        #   - the background is a ZERO-MEAN J2 oscillation (median signed dE/dt on NoThrust is
        #     exactly 0.0) of amplitude ~2.7e-6, and electric thrust contributes a constant-sign
        #     bias of only ~+2.0e-7 -- a 1:13 ratio, giving per-frame Electric-vs-NoThrust
        #     ROC AUC ~0.52 (chance). No threshold rescues an uninformative feature: recalibrating
        #     it merely flips the target to 0 on true Electric frames, which would supervise the
        #     model to MISS Electric.
        # The signal that does separate them is the SIGNED, orbit-integrated energy change
        # (AUC 0.652 over one full orbit, rising with window length), which is a trajectory-level
        # quantity and does not fit this per-timestep loss. Chemical needs no such help: it sits
        # ~12x above the background and is already perfectly separable (AUC 1.000).
        phys_target_ce, phys_gate_ce = pseudo_target_ce, gate_ce

    return states_cat, y_joint, n_ic_per_class, phys_target_ce, phys_gate_ce


def _deriveStage1Stage2(y_joint, pad_idx=-100):
    """y_joint: [...,] any shape of 4-class per-timestep labels (0=NoThrust,1=Chemical,
    2=Electric,3=Impulsive) -> (y_stage1, y_stage2). y_stage1 is the binary 'is thrust
    occurring' label; y_stage2 remaps Chemical/Electric/Impulsive to 0/1/2 and masks every
    non-thrusting position to pad_idx, so a type classifier is never supervised on background
    frames. Shared by the DataLoader-building path and the classic-ML/PCA+MLP row-based paths."""
    y_stage1 = (y_joint > 0).astype(np.int64)
    y_stage2 = np.full_like(y_joint, pad_idx)
    for src, dst in ((1, 0), (2, 1), (3, 2)):
        y_stage2[y_joint == src] = dst
    return y_stage1, y_stage2


def _icGroupSplit(n_ic_per_class, train_ratio, val_ratio, test_ratio, seed=None):
    """One IC-index permutation applied identically to every class block, so the same underlying
    initial condition (IC index i, shared across all 4 class npz files since each generator run
    reseeds the same RNG) always lands in the same split regardless of which thrust-type file it
    appears in -- prevents leaking near-duplicate pre-thrust dynamics across train/val/test.
    Assumes every class shares the same number of ICs."""
    assert len(set(n_ic_per_class)) == 1, f"_icGroupSplit assumes equal IC counts per class, got {n_ic_per_class}"
    n_ic = n_ic_per_class[0]

    n_train = int(np.floor(train_ratio * n_ic))
    n_val = int(np.floor(val_ratio * n_ic))
    n_test = n_ic - n_train - n_val
    assert n_test > 0, "Ratios leave no ICs for test; reduce train/val."

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ic)
    train_ic = perm[:n_train]
    val_ic = perm[n_train:n_train + n_val]
    test_ic = perm[n_train + n_val:]

    groups = np.tile(np.arange(n_ic, dtype=np.int64), len(n_ic_per_class))
    train_mask = np.isin(groups, train_ic)
    val_mask = np.isin(groups, val_ic)
    test_mask = np.isin(groups, test_ic)
    return train_mask, val_mask, test_mask


def _computeOversamplingWeights(y_joint, num_classes):
    """y_joint: [N,T] training-split joint labels -> weights[N] for a WeightedRandomSampler.
    Random-oversamples whole trajectories (rather than individual timesteps, which would break
    the temporal context LSTM/Mamba need) by weighting each trajectory toward the rarest
    per-timestep class it contains: weight_i = max over classes c present in trajectory i of
    (1 / count_c), so a trajectory touching a rare class (e.g. a single Chemical burst minute) is
    drawn more often regardless of how much NoThrust background padding surrounds it, while
    NoThrust-only trajectories keep the baseline weight."""
    counts = np.bincount(y_joint.reshape(-1), minlength=num_classes).astype(np.float64)
    inv_freq = counts.sum() / np.clip(counts, 1.0, None)
    weights = np.array([inv_freq[np.unique(row)].max() for row in y_joint], dtype=np.float64)
    return weights


def _standardizeSplits(train_data, val_data, test_data, supress_print=False):
    """Z-scores each feature channel using TRAIN-split statistics only, applied identically to val
    and test. Returns (train, val, test, mu, sigma).

    Fitting on train alone is what keeps this from leaking: computing val/test statistics from
    their own splits would let information about held-out orbits reach the model through the
    scaling constants. Channels with zero variance in train (a constant feature) are left alone
    rather than divided by ~0."""
    C = train_data.shape[2]
    flat = train_data.reshape(-1, C)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    degenerate = sigma < 1e-12
    sigma = np.where(degenerate, 1.0, sigma)

    if not supress_print:
        if degenerate.any():
            print(f"Standardize: channels {np.flatnonzero(degenerate).tolist()} are constant in "
                  f"train; left unscaled.")
        print(f"Standardize: per-channel train mean range [{mu.min():.4g}, {mu.max():.4g}], "
              f"std range [{sigma.min():.4g}, {sigma.max():.4g}] -> all channels z-scored")

    out = tuple((d - mu) / sigma for d in (train_data, val_data, test_data))
    return out + (mu, sigma)


def prepareInSequenceThrustClassificationDatasets(
    yaml_config, data_config,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    pos_noise_std=1e-3, vel_noise_std=1e-3,
    batch_size=16, pad_idx=-100, seed=None,
    supress_print=False, return_meta=False, oversample=False, standardize=False,
):
    """Loads all 4 thrust-type classes (Chemical, Electric, ImpBurn, NoThrust) and builds
    per-timestep labels for three views of the same data:
      - joint:  4-class per-timestep label (0=NoThrust,1=Chemical,2=Electric,3=Impulsive)
      - stage1: binary per-timestep 'is thrust occurring' label (0/1)
      - stage2: 3-class per-timestep thrust-type label (0=Chemical,1=Electric,2=Impulsive),
                masked to pad_idx on every non-thrusting timestep so a type classifier is never
                supervised on background frames.
    Assumes equal ICs per class and that IC index i refers to the same underlying orbit across
    all 4 class files. Per-timestep 'thrustingTime' ground truth is defensively loaded -- a class
    file missing the key degrades to all-background labels with a printed warning rather than
    raising.

    oversample: if True, train_loader draws whole training trajectories with replacement via a
    WeightedRandomSampler (see _computeOversamplingWeights) instead of a plain shuffle, so
    trajectories containing rarer per-timestep classes are seen more often each epoch. val_loader
    and test_loader are never resampled.

    standardize: if True, z-score every feature channel using TRAIN-split statistics only (val and
    test are transformed with the train mean/std, never their own). Off by default so existing
    results stay reproducible, but strongly recommended for any flag combination that leaves
    channels on wildly different scales -- notably --OE, whose channels span the semi-major axis
    (~6.7e3 km) and eccentricity (~1e-5) simultaneously, an 8-order-of-magnitude spread that
    saturates LSTM/Mamba gates on the first layer and collapses training to majority-class
    prediction. See _standardizeSplits.

    yaml_config['numSinusoids']: if > 0, replaces every feature channel with this many dominant
    sinusoidal components extracted per trajectory via FFT (see _decomposeSinusoids), applied as
    the last step of _applyTransforms -- so it decomposes whatever channels --OE/--energy/
    --energyRate leave behind. C channels become C*numSinusoids (e.g. 6 ECI channels with
    numSinusoids=3 yields 18), and standardize (if also on) fits on those expanded channels.
    """
    useOE = yaml_config['useOE']
    useNorm = yaml_config['useNorm']
    useNoise = yaml_config['useNoise']
    useEnergy = yaml_config['useEnergy']
    useEnergyRate = yaml_config['useEnergyRate']
    numSinusoids = yaml_config['numSinusoids']
    usePhysicsLoss = yaml_config.get('usePhysicsLoss', False)
    useJ2Energy = yaml_config.get('useJ2Energy', False)
    useResidualLadder = yaml_config.get('useResidualLadder', False)
    useResidualLadderFull = yaml_config.get('useResidualLadderFull', False)

    numMinProp = yaml_config['prop_time']
    train_set = yaml_config['orbit']
    systems = yaml_config['systems']
    test_set = yaml_config['test_dataset']
    test_systems = yaml_config['test_systems']

    dataLoc = data_config['seqClassification'] + train_set + "/" + str(numMinProp) + "min-" + str(systems)
    dataLoc_test = data_config['seqClassification'] + test_set + "/" + str(numMinProp) + "min-" + str(test_systems)

    if not supress_print:
        print(f"Training data location: {dataLoc}")
        print(f"Test data location: {dataLoc_test}")

    states, y_joint, n_ic_per_class, phys_target_ce, phys_gate_ce = _load_and_label(
        dataLoc, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std,
        usePhysicsLoss, useJ2Energy, useResidualLadder, useResidualLadderFull
    )
    if phys_target_ce is None:
        phys_target_ce = np.zeros_like(y_joint, dtype=np.float32)
        phys_gate_ce = np.zeros_like(y_joint, dtype=np.float32)

    train_mask, val_mask, test_mask = _icGroupSplit(n_ic_per_class, train_ratio, val_ratio, test_ratio, seed=seed)

    train_data, train_joint = states[train_mask], y_joint[train_mask]
    train_phys_target_ce, train_phys_gate_ce = phys_target_ce[train_mask], phys_gate_ce[train_mask]
    val_data, val_joint = states[val_mask], y_joint[val_mask]
    val_phys_target_ce, val_phys_gate_ce = phys_target_ce[val_mask], phys_gate_ce[val_mask]

    if test_set != train_set or test_systems != systems:
        states_t, y_joint_t, n_ic_per_class_t, phys_target_ce_t, phys_gate_ce_t = _load_and_label(
            dataLoc_test, useOE, useNorm, useNoise, useEnergy, useEnergyRate, numSinusoids, pos_noise_std, vel_noise_std,
            usePhysicsLoss, useJ2Energy, useResidualLadder, useResidualLadderFull
        )
        if phys_target_ce_t is None:
            phys_target_ce_t = np.zeros_like(y_joint_t, dtype=np.float32)
            phys_gate_ce_t = np.zeros_like(y_joint_t, dtype=np.float32)
        _, _, test_mask_t = _icGroupSplit(n_ic_per_class_t, train_ratio, val_ratio, test_ratio, seed=seed)
        test_data, test_joint = states_t[test_mask_t], y_joint_t[test_mask_t]
        test_phys_target_ce, test_phys_gate_ce = phys_target_ce_t[test_mask_t], phys_gate_ce_t[test_mask_t]
    else:
        test_data, test_joint = states[test_mask], y_joint[test_mask]
        test_phys_target_ce, test_phys_gate_ce = phys_target_ce[test_mask], phys_gate_ce[test_mask]

    # Standardization sits after the split (so statistics come from train only) and before the
    # loaders/raw arrays are handed out, keeping the neural and classic-ML paths on identical
    # features -- the Hankel-window baselines consume train_data/val_data/test_data directly.
    if standardize:
        train_data, val_data, test_data, _, _ = _standardizeSplits(
            train_data, val_data, test_data, supress_print=supress_print)

    train_stage1, train_stage2 = _deriveStage1Stage2(train_joint, pad_idx)
    val_stage1, val_stage2 = _deriveStage1Stage2(val_joint, pad_idx)
    test_stage1, test_stage2 = _deriveStage1Stage2(test_joint, pad_idx)

    if not supress_print:
        print(f"train_data {train_data.shape}  val_data {val_data.shape}  test_data {test_data.shape}")

    def _make_loader(data, yj, y1, y2, y_pt_ce, y_pg_ce, shuffle, sampler=None):
        ds = TensorDataset(
            torch.from_numpy(data),
            torch.from_numpy(yj).long(),
            torch.from_numpy(y1).long(),
            torch.from_numpy(y2).long(),
            torch.from_numpy(y_pt_ce).double(),
            torch.from_numpy(y_pg_ce).double(),
        )
        if sampler is not None:
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, pin_memory=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    train_sampler = None
    if oversample:
        weights = _computeOversamplingWeights(train_joint, num_classes=len(JOINT_LABELS))
        train_sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        if not supress_print:
            print(f"Oversampling enabled: train_loader weights range [{weights.min():.3f}, {weights.max():.3f}]")

    train_loader = _make_loader(train_data, train_joint, train_stage1, train_stage2,
                                 train_phys_target_ce, train_phys_gate_ce,
                                 train_sampler is None, sampler=train_sampler)
    val_loader = _make_loader(val_data, val_joint, val_stage1, val_stage2,
                               val_phys_target_ce, val_phys_gate_ce, False)
    test_loader = _make_loader(test_data, test_joint, test_stage1, test_stage2,
                                test_phys_target_ce, test_phys_gate_ce, False)

    result = (train_loader, val_loader, test_loader,
              train_data, train_joint, val_data, val_joint, test_data, test_joint)

    if return_meta:
        meta = {
            "class_names_joint": JOINT_CLASS_NAMES,
            "class_names_stage1": STAGE1_CLASS_NAMES,
            "class_names_stage2": STAGE2_CLASS_NAMES,
            "n_ic_per_class": n_ic_per_class,
        }
        return result + (meta,)
    return result
