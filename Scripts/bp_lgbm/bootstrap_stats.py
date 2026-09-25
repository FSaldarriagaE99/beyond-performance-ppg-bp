######## BOOTSTRAP STATISTICS — PAIRED CI / MDE / EQUIVALENCE #######################
#                                                                                    #
# Centralized math for paired bootstrap contrasts (e.g. filtered vs unfiltered      #
# training set, evaluated on the same resampled test draws). Consumed by:           #
#   - Experiment_MDE_Equivalence_Filtering.py (the original filtered_vs_unfiltered  #
#     contrast at threshold 90)                                                     #
#   - Sweep_Threshold_ML.py (the threshold sensitivity sweep, 80/85/90/95)          #
#                                                                                    #
# Kept separate from eval.py's compute_bootstrap_ci (which is generic single-arm    #
# CI over any metric distribution) because this module is specifically about        #
# PAIRED differences between two arms evaluated on matched resamples, plus the      #
# derived MDE / equivalence-test quantities that only make sense for that case.     #
#######################################################################################

import numpy as np

Z_MDE_DEFAULT = 2.8    # (1.96 + 0.84), two-sided alpha=0.05, 80% power
EQUIV_DELTA_DEFAULT = 2.0  # equivalence margin, pp of R2, declared a priori


def mde(sd_pp: float, z: float = Z_MDE_DEFAULT) -> float:
    """Minimum detectable effect, in the same units as sd_pp."""
    return z * sd_pp


def equivalence_test(ci90_lower_pp: float, ci90_upper_pp: float,
                      delta: float = EQUIV_DELTA_DEFAULT) -> bool:
    """
    TOST-style equivalence check: the 90% CI of the paired difference must
    fall entirely within (-delta, +delta) for the contrast to be declared
    equivalent (i.e. the null result is a real equivalence, not just an
    underpowered failure to reject).
    """
    return bool((ci90_lower_pp > -delta) and (ci90_upper_pp < delta))


def paired_diff_stats(
    diff_pp: np.ndarray,
    observed_diff_pp: float,
    z_mde: float = Z_MDE_DEFAULT,
    equiv_delta: float = EQUIV_DELTA_DEFAULT,
) -> dict:
    """
    Full paired-difference bootstrap summary for one contrast/target.

    Parameters
    ----------
    diff_pp : np.ndarray
        Bootstrap distribution of the paired difference (arm_a - arm_b),
        in percentage points, one value per resample. Resamples must be
        matched row-for-row between the two arms (same subject draw
        evaluated on both models) for the pairing to be meaningful.
    observed_diff_pp : float
        Point estimate of the difference on the full, non-resampled
        evaluation set (not the bootstrap mean).
    z_mde : float, default=2.8
        Multiplier for MDE = z_mde * SD.
    equiv_delta : float, default=2.0
        Equivalence margin in pp for the TOST-style check on the 90% CI.

    Returns
    -------
    dict with observed_diff_pp, sd_pp, mde_pp, ci95_lower_pp, ci95_upper_pp,
    ci90_lower_pp, ci90_upper_pp, equivalent.
    """
    sd_pp = float(np.std(diff_pp, ddof=1))
    mde_pp = mde(sd_pp, z=z_mde)

    ci95_lower_pp, ci95_upper_pp = np.percentile(diff_pp, [2.5, 97.5])
    ci90_lower_pp, ci90_upper_pp = np.percentile(diff_pp, [5, 95])

    equivalent = equivalence_test(ci90_lower_pp, ci90_upper_pp, delta=equiv_delta)

    return {
        "observed_diff_pp": float(observed_diff_pp),
        "sd_pp": sd_pp,
        "mde_pp": mde_pp,
        "ci95_lower_pp": float(ci95_lower_pp),
        "ci95_upper_pp": float(ci95_upper_pp),
        "ci90_lower_pp": float(ci90_lower_pp),
        "ci90_upper_pp": float(ci90_upper_pp),
        "equivalent": equivalent,
    }
