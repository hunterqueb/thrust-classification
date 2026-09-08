# Dataset-level separability analysis for NoThrust vs. Electric thrust. NO MODEL, NO TRAINING.
#
# Motivation: three architecturally unrelated backbones (LSTM, CNN, Transformer) all converge on
# Electric event-level precision of 22-24% with 94-100% recall. Models that different agreeing that
# closely is evidence of a ceiling imposed by the DATA, not by model capacity -- so the useful next
# question is whether the electric-thrust signature is separable from coasting at all, before any
# more architecture/loss tuning is spent chasing it.
#
# This measures that directly. For each propagation window it computes the same per-timestep
# thrust-accel-scale estimate the physics loss uses (residual_accel = |dE/dt| / |v|, see
# _computePhysicsResidualTensors) and asks how well that single physical quantity separates
# thrusting frames from coasting frames, via ROC AUC:
#
#   AUC ~ 0.5  the signature is indistinguishable from background -- no classifier fixes this, and
#              the real levers are longer windows, lower measurement noise, or a drag-aware feature
#   AUC ~ 1.0  the signature is cleanly present -- separability is a modeling/operating-point
#              problem, and tuning is worth the compute
#
# Two views, because they answer different questions:
#   instantaneous  per-timestep residual_accel. Does any single minute look like thrust?
#   cumulative     |E_final - E_initial| per trajectory. Low-thrust detection works by INTEGRATION
#                  (SPT-100 at 0.1 N is ~100x weaker than the chemical thruster but fires for
#                  ~80% of the window), so this is the view that should improve with window length
#                  -- and comparing it across 10/30/100 min is the direct test of that lever.
#
# Chemical is included throughout as an anchor: it is the "easy" class, so if the method shows
# Chemical separating cleanly while Electric does not, the gap is real rather than an artifact of
# the measurement. Two coasting baselines are reported for Electric, because they answer different
# questions: NoThrust-file frames are what the classifier actually has to reject, while the
# coasting frames INSIDE electric trajectories control for orbit geometry (same orbit, burn vs. no
# burn), isolating the thrust signature from altitude/drag variation across trajectories.
#
# Noise is deliberately NOT applied (--noise is opt-in in the main script and off here), so every
# number below is a BEST CASE. Real separability under measurement noise can only be worse.
#
# $ python scripts/two_body/analyzeThrustSeparability.py --orbit leo --systems 1500 --plot
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Measure NoThrust-vs-Electric separability in the data itself.")
parser.add_argument("--orbit", type=str, default="leo")
parser.add_argument("--systems", type=int, default=1500)
parser.add_argument("--propMins", type=int, nargs="+", default=[10, 30, 100],
                     help="Propagation windows to compare (each must exist on disk)")
parser.add_argument("--plot", action="store_true", help="Save distribution figure")
args = parser.parse_args()

# Import the main script for its loaders/physics. It parses argv at import time, so swap in a
# synthetic argv matching ITS parser first (see runSeqClassificationExperiment.py for the same dance).
_real_argv = sys.argv
sys.argv = ["mambaTimeSeriesSeqClassificationGMATThrusts.py",
            "--orbit", args.orbit, "--systems", str(args.systems), "--mode", "joint", "--no-classic"]
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mambaTimeSeriesSeqClassificationGMATThrusts as M
finally:
    sys.argv = _real_argv

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

# Okabe-Ito, the standard CVD-safe qualitative palette for scientific figures. Color follows the
# ENTITY (class), fixed order, never cycled -- and every series is also named in the legend, so
# identity never rests on color alone.
COLORS = {"No Thrust": "#4D4D4D", "Electric": "#0072B2", "Chemical": "#D55E00"}


def loadRawWindow(loc):
    """Loads one {propMin}min-{systems} directory WITHOUT any --OE/--norm/--noise transform, and
    returns per-class raw ECI states plus the per-timestep thrusting mask. Mirrors _load_and_label's
    own loading loop (same _getThrustingTime defensive read), but keeps the states dimensional so
    the physics quantities below are in real km/s^2."""
    states_by_class, thrusting_by_class = {}, {}
    for class_name in M.CLASS_ORDER:
        npz = np.load(f"{loc}/statesArray{class_name}.npz")
        states = npz[f"statesArray{class_name}"]
        N, T = states.shape[0], states.shape[1]
        tt = M._getThrustingTime(npz, class_name, N, T, warn_if_missing=False)
        states_by_class[class_name] = states
        thrusting_by_class[class_name] = tt.squeeze(-1).astype(bool)
    return states_by_class, thrusting_by_class


def analyzeWindow(loc):
    """Returns (populations, summary) for one propagation window.
    populations: {label: 1-D array} of instantaneous residual_accel, plus cumulative |dE| per
    trajectory keyed with a 'cum:' prefix."""
    states_by_class, thrusting_by_class = loadRawWindow(loc)

    resid, aJ2, aJ36, dE_total = {}, {}, {}, {}
    signed_rate, sma = {}, {}
    for c in M.CLASS_ORDER:
        h2, h36, g0, residual_accel = M._computePhysicsResidualTensors(states_by_class[c])
        resid[c] = residual_accel
        aJ2[c], aJ36[c] = h2 * g0, h36 * g0

        # Cumulative signal: net specific-energy change across the whole window. Constant under
        # pure two-body coasting, so any departure is drag/SRP/harmonics (NoThrust) or those plus
        # thrust (Chemical/Electric). This is the quantity a longer window is supposed to grow.
        s = states_by_class[c]
        r_mag = np.linalg.norm(s[..., 0:3], axis=-1)
        v_mag = np.linalg.norm(s[..., 3:6], axis=-1)
        from qutils.orbital import MU_EARTH_JGM2
        energy = 0.5 * v_mag**2 - MU_EARTH_JGM2 / r_mag
        # SIGNED, deliberately. The electric thruster fires along-track (+V, GMAT's ElectricThruster
        # default direction) so it only ever ADDS energy, while the background here is a zero-mean
        # J2 oscillation. Taking |dE| -- as residual_accel above does -- discards the sign, which is
        # the one thing that distinguishes a constant-sign thrust bias from a symmetric oscillation.
        # Both are reported so the cost of that choice is visible rather than assumed.
        dE_total[c] = energy[:, -1] - energy[:, 0]
        signed_rate[c] = np.diff(energy, axis=1, prepend=energy[:, :1]) / M.PHYS_LOSS_DT_SECONDS / np.maximum(v_mag, 1e-9)
        sma[c] = -MU_EARTH_JGM2 / (2.0 * energy)

    thrusting_e = thrusting_by_class["Electric"]
    thrusting_c = thrusting_by_class["Chemical"]

    pops = {
        # What the classifier actually has to reject: whole coasting trajectories.
        "No Thrust": resid["NoThrust"].reshape(-1),
        "Electric": resid["Electric"][thrusting_e],
        "Chemical": resid["Chemical"][thrusting_c],
        # Within-trajectory control: same orbits, burn frames vs. their own coast frames.
        "Electric (own coast)": resid["Electric"][~thrusting_e],
        "cum:No Thrust": dE_total["NoThrust"],
        "cum:Electric": dE_total["Electric"],
        "cum:Chemical": dE_total["Chemical"],
        "sgn:No Thrust": signed_rate["NoThrust"].reshape(-1),
        "sgn:Electric": signed_rate["Electric"][thrusting_e],
        "sgn:Chemical": signed_rate["Chemical"][thrusting_c],
    }

    def auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return roc_auc_score(y, np.concatenate([pos, neg]))

    # Mean orbital period across all trajectories, from the semi-major axis. The window/period
    # ratio is the number that matters for the cumulative view: the J2 background oscillation only
    # cancels out over a WHOLE orbit, so a window covering a fraction of one can't average it away.
    a_mean = float(np.mean(np.concatenate([sma[c].reshape(-1) for c in M.CLASS_ORDER])))
    from qutils.orbital import MU_EARTH_JGM2
    period_min = 2.0 * np.pi * np.sqrt(a_mean**3 / MU_EARTH_JGM2) / 60.0
    T = resid["NoThrust"].shape[1]

    summary = {
        "auc_elec_inst": auc(pops["Electric"], pops["No Thrust"]),
        "auc_elec_inst_own": auc(pops["Electric"], pops["Electric (own coast)"]),
        "auc_chem_inst": auc(pops["Chemical"], pops["No Thrust"]),
        "auc_elec_cum": auc(pops["cum:Electric"], pops["cum:No Thrust"]),
        "auc_chem_cum": auc(pops["cum:Chemical"], pops["cum:No Thrust"]),
        "auc_elec_sgn": auc(pops["sgn:Electric"], pops["sgn:No Thrust"]),
        "med_aJ2": float(np.median(aJ2["Electric"])),
        "med_aJ36": float(np.median(aJ36["Electric"])),
        "med_sgn_nothrust": float(np.median(pops["sgn:No Thrust"])),
        "med_sgn_elec": float(np.median(pops["sgn:Electric"])),
        "period_min": period_min,
        "orbits": T / period_min,
    }
    return pops, summary


def main():
    with open("data.yaml") as f:
        data_config = yaml.safe_load(f)
    base = data_config["seqClassification"] + args.orbit + "/"

    all_pops, all_summary = {}, {}
    for pm in args.propMins:
        loc = f"{base}{pm}min-{args.systems}"
        if not os.path.isdir(loc):
            print(f"[skip] {loc} does not exist")
            continue
        all_pops[pm], all_summary[pm] = analyzeWindow(loc)

    if not all_summary:
        print("No windows found -- nothing to analyze.")
        return

    print(f"\n{'='*78}\nInstantaneous per-frame residual_accel = |dE/dt|/|v|  (km/s^2)\n{'='*78}")
    print(f"{'Window':>8} | {'median NoThrust':>16} | {'median Electric':>16} | {'median Chemical':>16}")
    print("-" * 78)
    for pm, pops in all_pops.items():
        print(f"{pm:>6}min | {np.median(pops['No Thrust']):>16.3e} | "
              f"{np.median(pops['Electric']):>16.3e} | {np.median(pops['Chemical']):>16.3e}")

    print(f"\n{'='*78}\nSigned per-frame dE/dt / |v| -- the sign is the discriminator\n{'='*78}")
    print(f"{'Window':>8} | {'median NoThrust':>16} | {'median Electric':>16}")
    print("-" * 78)
    for pm, s in all_summary.items():
        print(f"{pm:>6}min | {s['med_sgn_nothrust']:>+16.3e} | {s['med_sgn_elec']:>+16.3e}")
    print("  NoThrust sits at exactly zero -- the background is a ZERO-MEAN oscillation (J2 short-"
          "period\n  energy variation), not a secular drag decay. Electric carries a small "
          "constant-sign positive\n  bias, because the thruster fires along-track. A zero-mean "
          "background is exactly the case where\n  INTEGRATION recovers a buried signal -- but only "
          "over whole orbits.")

    print(f"\n{'='*78}\nSeparability (ROC AUC on that single physical quantity; 0.5 = no signal)\n{'='*78}")
    print(f"{'Window':>8} | {'orbits':>7} | {'Elec inst':>10} | {'Elec inst*':>11} | {'Elec sgn':>9} | "
          f"{'Elec cumul':>11} | {'Chem inst':>10}")
    print("-" * 78)
    for pm, s in all_summary.items():
        print(f"{pm:>6}min | {s['orbits']:>7.2f} | {s['auc_elec_inst']:>10.3f} | {s['auc_elec_inst_own']:>11.3f} | "
              f"{s['auc_elec_sgn']:>9.3f} | {s['auc_elec_cum']:>11.3f} | {s['auc_chem_inst']:>10.3f}")
    print("  * 'Elec inst*' compares electric burn frames against the coast frames of their OWN "
          "trajectories\n    (controls for altitude variation between trajectories).")
    for pm, s in all_summary.items():
        print(f"  {pm} min = {s['orbits']:.2f} orbits (period {s['period_min']:.1f} min)")

    print(f"\n{'='*78}\nAgainst the analytic zonal-harmonic scales (the --physics-loss-weight premise)\n{'='*78}")
    for pm, s in all_summary.items():
        med_e = np.median(all_pops[pm]["Electric"])
        med_c = np.median(all_pops[pm]["Chemical"])
        print(f"{pm:>6}min | a_J2 {s['med_aJ2']:.3e} | a_J3-J6 {s['med_aJ36']:.3e} | "
              f"measured Electric {med_e:.3e} | measured Chemical {med_c:.3e}")
    print("  Premise under test: Chemical should sit near a_J2 and Electric near a_J3-J6.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        windows = list(all_pops.keys())
        fig, axes = plt.subplots(2, len(windows), figsize=(5.2 * len(windows), 7.5), squeeze=False)

        for j, pm in enumerate(windows):
            pops, s = all_pops[pm], all_summary[pm]

            ax = axes[0][j]
            for name in ("No Thrust", "Electric", "Chemical"):
                d = pops[name]
                d = d[d > 0]
                ax.hist(np.log10(d), bins=60, density=True, alpha=0.55,
                        color=COLORS[name], label=name)
            ax.axvline(np.log10(s["med_aJ2"]), color="black", lw=1.2, ls="--")
            ax.text(np.log10(s["med_aJ2"]), ax.get_ylim()[1] * 0.96, " a_J2",
                    fontsize=8, va="top", color="black")
            ax.axvline(np.log10(s["med_aJ36"]), color="black", lw=1.2, ls=":")
            ax.text(np.log10(s["med_aJ36"]), ax.get_ylim()[1] * 0.96, " a_J3-J6",
                    fontsize=8, va="top", color="black")
            ax.set_title(f"{pm} min -- per-frame residual\nElectric vs NoThrust AUC = {s['auc_elec_inst']:.3f}",
                         fontsize=10)
            ax.set_xlabel("log10 residual_accel  (km/s$^2$)")
            ax.set_ylabel("density" if j == 0 else "")
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.legend(fontsize=8, frameon=False)

            ax = axes[1][j]
            # Signed, on a linear axis clipped to the bulk: the message is the SHIFT of Electric
            # off a NoThrust distribution centred on zero, which a log-of-absolute-value axis would
            # destroy by folding the negative half onto the positive.
            allcum = np.concatenate([pops["cum:" + n] for n in ("No Thrust", "Electric", "Chemical")])
            lo, hi = np.percentile(allcum, [1, 99])
            bins = np.linspace(lo, hi, 60)
            for name in ("No Thrust", "Electric", "Chemical"):
                ax.hist(np.clip(pops["cum:" + name], lo, hi), bins=bins, density=True, alpha=0.55,
                        color=COLORS[name], label=name)
            ax.axvline(0.0, color="black", lw=1.0, ls="--")
            ax.set_title(f"{pm} min ({s['orbits']:.2f} orbits) -- signed $\\Delta$E per trajectory\n"
                         f"Electric vs NoThrust AUC = {s['auc_elec_cum']:.3f}", fontsize=10)
            ax.set_xlabel("$\\Delta$E  (km$^2$/s$^2$)")
            ax.set_ylabel("density" if j == 0 else "")
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.legend(fontsize=8, frameon=False)

        fig.suptitle(f"Thrust separability in the data alone -- {args.orbit}, {args.systems} systems "
                     f"(no model, no measurement noise)", fontsize=12)
        fig.tight_layout()

        plot_dir = f"gmat/data/seqClassification/{args.orbit}"
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"separability_{args.orbit}_{args.systems}.png")
        fig.savefig(save_path, dpi=150)
        print(f"\nSaved separability figure -> {save_path}")


if __name__ == "__main__":
    main()
