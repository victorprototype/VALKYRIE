# =============================================================================
# VALKYRIE | trigrs_physics.py
# Stage   : C. Physics: rain -> water pressure -> slope stability (TRIGRS)
# Purpose : The two TRIGRS equations used by VALKYRIE:
#             (1) transient pore-water pressure after rain infiltrates (Iverson 2000, eq. 1A)
#             (2) infinite-slope Factor of Safety including that pressure (eq. 18)
#           Implemented directly from the published equations, not the TRIGRS program.
# Reads   : terrain slope, soil parameters, hourly rainfall
# Writes  : pressure head psi (m) and Factor of Safety FS
# Used by : run_simulation.py
# =============================================================================

"""
TRIGRS transient infiltration + infinite-slope factor of safety, implementing
Iverson (2000) / Baum et al. (2008) equations 1A and 18 from the TRIGRS
manual, directly (not a black-box reimplementation of the TRIGRS executable
itself -- we don't have that binary -- this is the published governing
equations evaluated ourselves).

IMPORTANT CAVEAT the TRIGRS manual itself states, which applies directly
here since we have no field-measured initial water table depth for this
basin: "Model results are very sensitive to the steady-seepage initial
condition ... In the absence of accurate initial conditions, use of TRIGRS
is limited to modeling hypothetical scenarios." We use an assumed initial
water table depth and steady background flux -- this is a hypothetical
scenario run, not a calibrated hazard forecast.
"""
import numpy as np
from scipy.special import erfc

# Constants: gravity (m/s2) and unit weight of water (kN/m3).
G = 9.81
GAMMA_W = 9.81  # kN/m^3, unit weight of water


def ierfc(eta):
    """ierfc(eta) = (1/sqrt(pi)) exp(-eta^2) - eta * erfc(eta)   [TRIGRS eq. 1B]"""
    return np.exp(-eta ** 2) / np.sqrt(np.pi) - eta * erfc(eta)


def pressure_head_transient(Z, t, slope_rad, Ks, D0, rainfall_intensity_seq, dt_hours,
                             water_table_depth_m, steady_flux_frac=0.02):
    """
    Iverson (2000) eq. 1A -- infinitely deep basal boundary solution.

    Z: vertical depth below ground surface (m), scalar or array matching node shape
    t: evaluation time (s) since simulation start
    slope_rad, Ks (m/s), D0 (m^2/s), water_table_depth_m: per-node arrays
    rainfall_intensity_seq: [N_bins, n_nodes] array, m/s, one column per node
    dt_hours: duration of each rainfall bin, hours
    steady_flux_frac: I_ZLT expressed as a fraction of Ks (the manual's own
        "background" steady infiltration rate -- we have no measured value,
        so this is an assumed small fraction, flagged as an assumption)
    """
    # Step 1 - terms that depend only on slope and soil: cos^2(slope), the background
    # seepage term beta, and D1 (how fast pressure diffuses through the soil).
    cos2 = np.cos(slope_rad) ** 2
    I_ZLT = steady_flux_frac * Ks
    beta = cos2 - (I_ZLT / np.maximum(Ks, 1e-12))
    D1 = D0 / np.maximum(cos2, 1e-6)

    # Step 2 - starting state before the storm: steady-state pressure head.
    # Negative above the water table (soil suction), zero at it.
    psi = beta * (Z - water_table_depth_m)

    # Step 3 - add the effect of each hour of rain. Each hour is a block of constant
    # intensity I_n, built from two step changes (rain switches on, then off).
    n_bins = rainfall_intensity_seq.shape[0]
    dt_s = dt_hours * 3600.0

    for n in range(n_bins):
        t_n = n * dt_s
        t_n1 = (n + 1) * dt_s
        I_n = rainfall_intensity_seq[n]

        # Rain block 'switches on' at t_n: extra pressure that builds with time since then.
        if t > t_n:
            tau = t - t_n
            arg = Z / (2.0 * np.sqrt(np.maximum(D1 * tau, 1e-12)))
            psi = psi + 2.0 * (I_n / np.maximum(Ks, 1e-12)) * np.sqrt(D1 * tau) * ierfc(arg)

        # Same block 'switches off' at t_n1: subtract the delayed copy, leaving a pulse of
        # exactly one hour (Iverson's superposition of step responses).
        if t > t_n1:
            tau = t - t_n1
            arg = Z / (2.0 * np.sqrt(np.maximum(D1 * tau, 1e-12)))
            psi = psi - 2.0 * (I_n / np.maximum(Ks, 1e-12)) * np.sqrt(D1 * tau) * ierfc(arg)

    # Iverson's physical limit: pressure head can't exceed a water table at the surface
    psi = np.minimum(psi, Z)
    return psi


def factor_of_safety(Z, psi, slope_rad, cohesion_kpa, friction_deg, unit_weight_kn_m3):
    """TRIGRS eq. 18, infinite-slope model with pore pressure."""
    # FS = (friction term) + (cohesion minus pore-pressure term). Rising pressure psi
    # reduces the second term; FS < 1 means the slope can no longer hold.
    phi = np.radians(friction_deg)
    theta = slope_rad
    numerator = cohesion_kpa - psi * GAMMA_W * np.tan(phi)
    denominator = unit_weight_kn_m3 * Z * np.sin(theta) * np.cos(theta)
    # Guard against division by zero on perfectly flat or zero-depth cells.
    denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
    fs = np.tan(phi) / np.maximum(np.tan(theta), 1e-6) + numerator / denominator
    return fs


if __name__ == "__main__":
    # Run `python trigrs_physics.py` for a self-contained demo (no data files needed).
    # smoke test on synthetic values spanning a plausible parameter range
    n = 5
    slope = np.radians(np.array([10, 20, 30, 40, 50], dtype=np.float64))
    Ks = np.full(n, 2e-6)
    D0 = np.full(n, 1e-5)
    c = np.full(n, 10.0)
    phi = np.full(n, 27.0)
    uws = np.full(n, 18.0)
    wt_depth = np.full(n, 2.0)
    Z = np.full(n, 1.5)

    # 24 hourly bins, a peaked storm (SCS-Type-II-like) totalling 150mm/day
    hours = np.arange(24)
    shape = np.exp(-0.5 * ((hours - 14) / 2.5) ** 2)  # peak around hour 14
    shape /= shape.sum()
    daily_total_m = 0.150
    hourly_m = shape * daily_total_m
    intensity_m_s = hourly_m / 3600.0
    rainfall_seq = np.tile(intensity_m_s[:, None], (1, n))

    print("hour |  psi(m) |  FS")
    for h in range(1, 25):
        t = h * 3600.0
        psi = pressure_head_transient(Z, t, slope, Ks, D0, rainfall_seq, 1.0, wt_depth)
        fs = factor_of_safety(Z, psi, slope, c, phi, uws)
        print(f"{h:4d} | " + " ".join(f"{p:6.3f}" for p in psi) + " | " + " ".join(f"{f:6.2f}" for f in fs))
