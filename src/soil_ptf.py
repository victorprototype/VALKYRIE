# =============================================================================
# VALKYRIE | soil_ptf.py
# Stage   : A. Soil parameters
# Purpose : Pedotransfer functions: turns SoilGrids texture (sand/clay) and bulk density
#           into the soil properties the landslide physics needs (strength, permeability,
#           weight). These are generic literature-based estimates, not measurements.
# Reads   : sand %, clay %, bulk density
# Writes  : cohesion, friction angle, unit weight, Ksat, diffusivity, ...
# Used by : run_simulation.py, build_graph_v2.py, build_dem_3d_viewer.py
# =============================================================================

"""
Pedotransfer functions: SoilGrids texture/bulk-density -> TRIGRS parameters.

IMPORTANT CAVEAT (read before trusting any output of this module):
SoilGrids provides sand/clay/silt fractions and bulk density -- NOT
cohesion, friction angle, hydraulic conductivity, or any of the other
quantities TRIGRS' factor-of-safety equation (Eq. 18) actually needs.
Every function below is a generic, texture-class-based approximation from
the soil physics / geotechnical literature, not a measurement and not
calibrated to this specific basin. Treat all outputs as order-of-magnitude
planning estimates, not values to base any real hazard decision on.

Units in, per SoilGrids 2.0 convention:
  sand, clay: g/kg  (divide by 10 for %)
  bdod (bulk density): cg/cm3 (divide by 100 for g/cm3)
"""
import numpy as np


def soilgrids_to_fractions(sand_gkg, clay_gkg, bdod_cgcm3):
    """Convert SoilGrids units to ordinary ones: g/kg -> %, cg/cm3 -> g/cm3.
    Silt is whatever is left after sand and clay (clipped to 0-100 %).
    """
    sand_pct = sand_gkg / 10.0
    clay_pct = clay_gkg / 10.0
    silt_pct = np.clip(100.0 - sand_pct - clay_pct, 0, 100)
    bulk_density_gcm3 = bdod_cgcm3 / 100.0
    return sand_pct, silt_pct, clay_pct, bulk_density_gcm3


def saxton_rawls_hydraulic(sand_pct, clay_pct):
    """
    Approximate soil-water retention & Ksat from texture, in the style of
    Saxton & Rawls (2006, Soil Sci. Soc. Am. J. 70:1569-1578) -- organic
    matter term omitted (not available from these two SoilGrids bands),
    which biases theta_sat and Ksat somewhat low for organic-rich topsoil.
    Coefficients here are simplified illustrative fits to the published
    relationships, NOT transcribed digit-for-digit from the paper -- treat
    as approximate, and re-derive from the source paper (or run actual
    ROSETTA/Saxton-Rawls software) before using this for anything load-bearing.
    """
    # Use fractions (0-1), as in the published Saxton-Rawls equations.
    s = sand_pct / 100.0
    c = clay_pct / 100.0

    # Water content at the wilting point (-1500 kPa) and at field capacity (-33 kPa),
    # both estimated from texture.
    theta_1500 = -0.024 * s + 0.487 * c + 0.006 + 0.005 * (s * c) - 0.013 * (c ** 2)  # ~residual/wilting point
    theta_33 = -0.251 * s + 0.195 * c + 0.011 + 0.006 * (s * c) - 0.027 * (c ** 2) + 0.452 * (s * c)
    theta_33 = np.clip(theta_33, theta_1500 + 0.02, 0.6)

    # Saturated water content = field capacity + extra water held up to full saturation.
    theta_s_minus_33 = 0.278 * s + 0.034 * c + 0.022 - 0.018 * (s * c) - 0.027 * (c ** 2)
    theta_sat = np.clip(theta_33 + theta_s_minus_33, theta_33 + 0.02, 0.65)

    # Residual water content: water the soil keeps even when very dry.
    theta_res = np.clip(theta_1500 * 0.7, 0.01, theta_33 - 0.02)  # rough residual < wilting point

    # saturated hydraulic conductivity (mm/hr -> m/s), Saxton-Rawls style form
    B = (np.log(1500) - np.log(33)) / (np.log(theta_33) - np.log(theta_1500) + 1e-9)
    lam = 1.0 / B
    ksat_mm_hr = 1930.0 * (theta_sat - theta_33) ** (3 - lam)
    ksat_mm_hr = np.clip(ksat_mm_hr, 0.1, 500)
    ksat_m_s = ksat_mm_hr / 1000.0 / 3600.0

    alpha_1_m = np.clip(1.0 / np.clip((theta_33 * 100), 5, 200) * 100, 0.5, 20.0)  # very rough capillary-rise proxy
    diffusivity_m2_s = ksat_m_s / np.clip((theta_sat - theta_res), 0.05, 1.0) * 10  # order-of-magnitude only

    return {
        "theta_sat": theta_sat.astype(np.float32),
        "theta_res": theta_res.astype(np.float32),
        "ksat_m_s": ksat_m_s.astype(np.float32),
        "alpha_1_m": alpha_1_m.astype(np.float32),
        "diffusivity_m2_s": diffusivity_m2_s.astype(np.float32),
    }


def usda_texture_class(sand_pct, silt_pct, clay_pct):
    """Coarse USDA textural triangle classification -- vectorized, simplified
    (rectangular approximation of the true triangle boundaries; adequate for
    picking a strength-parameter bucket, not for cartographic soil mapping)."""
    # Everything starts as 'loam'; the rules below overwrite it with more specific
    # classes (later rules win where they overlap).
    cls = np.full(sand_pct.shape, "loam", dtype=object)
    cls = np.where(clay_pct >= 40, "clay", cls)
    cls = np.where((clay_pct >= 27) & (clay_pct < 40) & (sand_pct < 45), "clay_loam", cls)
    cls = np.where((silt_pct >= 80), "silt", cls)
    cls = np.where((silt_pct >= 50) & (clay_pct < 27), "silt_loam", cls)
    cls = np.where((sand_pct >= 70) & (clay_pct < 15), "sandy_loam", cls)
    cls = np.where((sand_pct >= 85), "sand", cls)
    return cls


# Literature-typical ranges (midpoints used here), gathered from standard
# geotechnical references (e.g. NAVFAC DM-7 style texture-strength tables).
# c' in kPa, phi' in degrees. WIDE uncertainty -- these are illustrative
# buckets, not site-specific measurements.
TEXTURE_STRENGTH = {
    "sand":        (1.0, 33.0),
    "sandy_loam":  (4.0, 30.0),
    "loam":        (8.0, 28.0),
    "silt_loam":   (10.0, 27.0),
    "silt":        (6.0, 26.0),
    "clay_loam":   (15.0, 24.0),
    "clay":        (20.0, 20.0),
}


def texture_to_strength(texture_class):
    """Look up cohesion (kPa) and friction angle (degrees) for each texture class
    from TEXTURE_STRENGTH above.
    """
    cohesion = np.zeros(texture_class.shape, dtype=np.float32)
    friction = np.zeros(texture_class.shape, dtype=np.float32)
    for cls, (c, phi) in TEXTURE_STRENGTH.items():
        mask = texture_class == cls
        cohesion[mask] = c
        friction[mask] = phi
    return cohesion, friction


def bulk_density_to_unit_weight(bulk_density_gcm3):
    """Dry unit weight from dry bulk density (gamma = rho * g). This is the
    DRY unit weight -- TRIGRS wants total (moist) unit weight, which runs
    higher; without a measured water content we cannot correct for that
    here, so this under-estimates uws. Flagged, not silently corrected."""
    rho_kg_m3 = bulk_density_gcm3 * 1000.0
    g = 9.81
    return (rho_kg_m3 * g / 1000.0).astype(np.float32)  # kN/m^3


# Manual check on the full SoilGrids file (uses data/soilgrids_mandakini.tif).
if __name__ == "__main__":
    from read_soilgrids import read_tiled_lzw_tiff
    from paths import data_file
    arr = read_tiled_lzw_tiff(data_file("soilgrids_mandakini.tif"))
    sand = arr[:, :, 0].astype(np.float32)
    clay = arr[:, :, 2].astype(np.float32)
    bdod = arr[:, :, 4].astype(np.float32)

    sand_pct, silt_pct, clay_pct, bd_gcm3 = soilgrids_to_fractions(sand, clay, bdod)
    hydro = saxton_rawls_hydraulic(sand_pct, clay_pct)
    texture = usda_texture_class(sand_pct, silt_pct, clay_pct)
    cohesion, friction = texture_to_strength(texture)
    uws = bulk_density_to_unit_weight(bd_gcm3)

    print("texture class counts:")
    vals, counts = np.unique(texture, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  {v:12s} {c:6d} px")
    print()
    for name, val in [("theta_sat", hydro["theta_sat"]), ("theta_res", hydro["theta_res"]),
                       ("ksat_m_s", hydro["ksat_m_s"]), ("cohesion_kpa", cohesion),
                       ("friction_deg", friction), ("uws_kn_m3", uws)]:
        print(f"{name:14s} min={val.min():.4g} max={val.max():.4g} mean={val.mean():.4g}")
