# =============================================================================
# VALKYRIE | run_simulation.py
# Stage   : C. Physics: the 24-hour simulation (main driver)
# Purpose : Runs the whole science pipeline: soil estimates -> TRIGRS pore pressure and
#           Factor of Safety for each hour -> Voellmy runout of every slope that fails.
# Reads   : DEM rasters, soilgrids_mandakini.tif, chirps-*.tif
# Writes  : simulation_24h.npz (FS, pressure, debris depth for each of 24 hours)
# Used by : pack_simulation_data.py -> build_final_with_simulation.py
# =============================================================================

"""
Runs the full 24-hour TRIGRS transient infiltration + factor-of-safety
simulation over the real unified graph (DEM + hydro + CHIRPS + SoilGrids-
derived params), then routes any triggered failures downslope using a
simplified Voellmy-rheology shallow-flow model (see voellmy_runout.py).

Outputs a per-hour time series (24 frames) of: pore pressure, factor of
safety, failure state, and flow depth -- at full graph resolution
(169,420 nodes) -- saved as a single .npz for the HTML viewer to consume.

CAVEATS (see soil_ptf.py, trigrs_physics.py, voellmy_runout.py for detail):
  - soil strength/hydraulic params are pedotransfer-function estimates, not measurements
  - initial water table depth and steady background flux are assumed, not measured
    (TRIGRS's own manual: results are "very sensitive" to this and calling this
    a "hypothetical scenario" without field data, which is exactly our situation)
  - the hourly storm shape is a synthetic SCS-Type-II-like disaggregation of
    CHIRPS' known daily total -- the real storm's true sub-daily timing is unknown
  - Voellmy mu/xi are literature-typical values, not calibrated to this basin
This is a demonstration/methodology pipeline, not a validated hazard forecast.
"""
import time
import numpy as np

import build_graph as bg
from soil_ptf import (soilgrids_to_fractions, saxton_rawls_hydraulic,
                       usda_texture_class, texture_to_strength, bulk_density_to_unit_weight)
from resample_to_grid import resample_soilgrids_to_points, resample_chirps_to_points
from trigrs_physics import pressure_head_transient, factor_of_safety
from voellmy_runout import route_failures
from paths import data_file, output_file

# Input data files (in the data/ folder), then the three ASSUMED scenario values. None of the three
# is measured for this basin; see the caveats in the docstring above.
SOILGRIDS_FILE = data_file("soilgrids_mandakini.tif")
CHIRPS_FILE = data_file("chirps-v3.0.sat.2013.06.17.tif")

FAILURE_DEPTH_M = 1.5          # assumed uniform shallow-failure slip depth
INITIAL_WATER_TABLE_M = 2.0    # assumed -- no field data available
STEADY_FLUX_FRAC = 0.02        # I_ZLT as a fraction of Ksat -- assumed
N_HOURS = 24


def build_node_lonlat(graph):
    """Longitude/latitude of every graph node (same geometry as build_graph_v2.node_lonlat)."""
    gh, gw = graph["grid_shape"]
    B = bg.BORDER_CROP
    step = round((2160 - 2 * B) / bg.GRAPH_TARGET_W)
    lat_top = 30.85000001111111 - B * 0.0002777777777777778
    lon_left = 78.84999998888891 + B * 0.0002777777777777778
    rows = np.arange(gh) * step
    cols = np.arange(gw) * step
    lat = lat_top - rows * 0.0002777777777777778
    lon = lon_left + cols * 0.0002777777777777778
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon_grid, lat_grid


def scs_type_ii_hourly_shape(peak_hour=14.0, width=2.8):
    """Synthetic single-peak storm disaggregation (Gaussian approximation of
    an SCS Type II-like shape) -- NOT the real storm's actual hyetograph,
    which we don't have at hourly resolution. Flagged."""
    hours = np.arange(N_HOURS)
    shape = np.exp(-0.5 * ((hours - peak_hour) / width) ** 2)
    return shape / shape.sum()


def main():
    """Run the full simulation and save simulation_24h.npz.

    Steps: 1 terrain graph | 2 soil parameters | 3 rainfall | 4 hourly storm |
    5 TRIGRS pressure + Factor of Safety per hour | 6 Voellmy runout | 7 save.
    """
    # STEP 1 - terrain graph: slope and elevation at every node.
    print("building base graph (DEM + hydrology)...")
    graph = bg.build_graph()
    names = graph["feature_names"]
    x = graph["x"]
    gh, gw = graph["grid_shape"]
    n = x.shape[0]
    slope_rad = np.radians(x[:, names.index("slope_deg")])
    elevation = x[:, names.index("elevation_m")]

    lon_grid, lat_grid = build_node_lonlat(graph)

    # STEP 2 - soil: sand/clay/density -> cohesion, friction angle, unit weight,
    # permeability (Ksat) and diffusivity at every node.
    print("resampling SoilGrids + deriving TRIGRS params...")
    soil = resample_soilgrids_to_points(SOILGRIDS_FILE, lon_grid, lat_grid)
    sand_pct, silt_pct, clay_pct, bd_gcm3 = soilgrids_to_fractions(
        soil["sand_gkg"], soil["clay_gkg"], soil["bdod_cgcm3"])
    hydro = saxton_rawls_hydraulic(sand_pct, clay_pct)
    texture = usda_texture_class(sand_pct, silt_pct, clay_pct)
    cohesion, friction = texture_to_strength(texture)
    uws = bulk_density_to_unit_weight(bd_gcm3)
    has_soil = soil["has_soil_data"].reshape(-1)

    cohesion = cohesion.reshape(-1)
    friction = friction.reshape(-1)
    uws = uws.reshape(-1)
    Ks = hydro["ksat_m_s"].reshape(-1)
    D0 = hydro["diffusivity_m2_s"].reshape(-1)

    # nodes outside the mapped watershed: fall back to mid-range literature
    # defaults rather than zero (zero cohesion/friction would make them
    # spuriously "fail" instantly, which misrepresents missing data as
    # instability -- a completely different kind of wrong)
    cohesion = np.where(has_soil, cohesion, 10.0)
    friction = np.where(has_soil, friction, 27.0)
    uws = np.where(has_soil, uws, 18.0)
    Ks = np.where(has_soil, Ks, 1e-6)
    D0 = np.where(has_soil, D0, 1e-5)

    # STEP 3 - daily rainfall total (mm) at every node.
    print("resampling CHIRPS precipitation...")
    precip_mm = resample_chirps_to_points(CHIRPS_FILE, lon_grid, lat_grid).reshape(-1)

    # STEP 4 - spread each node's daily total over 24 hours using the synthetic storm shape,
    # converted to metres per second as TRIGRS expects.
    storm_shape = scs_type_ii_hourly_shape()
    hourly_m = storm_shape[:, None] * (precip_mm[None, :] / 1000.0)  # [24, n_nodes] meters
    rainfall_seq_m_s = hourly_m / 3600.0

    # Same slip-surface depth and same starting water-table depth everywhere (assumed).
    Z = np.full(n, FAILURE_DEPTH_M)
    wt = np.full(n, INITIAL_WATER_TABLE_M)

    # STEP 5 - for each hour: (a) TRIGRS pore-water pressure at the end of that hour,
    # (b) Factor of Safety from that pressure. FS is capped to 0-5 for display.
    print("running 24-hour TRIGRS transient simulation...")
    t0 = time.time()
    psi_series = np.zeros((N_HOURS, n), dtype=np.float32)
    fs_series = np.zeros((N_HOURS, n), dtype=np.float32)
    for h in range(N_HOURS):
        t = (h + 1) * 3600.0
        psi = pressure_head_transient(Z, t, slope_rad, Ks, D0, rainfall_seq_m_s, 1.0, wt,
                                       steady_flux_frac=STEADY_FLUX_FRAC)
        fs = factor_of_safety(Z, psi, slope_rad, cohesion, friction, uws)
        psi_series[h] = psi
        fs_series[h] = np.clip(fs, 0, 5)  # clip for sane display; physics already computed
    print(f"  done in {time.time()-t0:.2f}s")

    # STEP 6 - every cell that reaches FS < 1 releases soil, which flows downhill.
    print("routing triggered failures (Voellmy runout)...")
    t0 = time.time()
    flow_depth_series, cum_erosion = route_failures(
        fs_series, elevation.reshape(gh, gw), slope_rad.reshape(gh, gw),
        friction.reshape(gh, gw), grid_shape=(gh, gw), n_hours=N_HOURS, max_depth_m=5.0)
    print(f"  done in {time.time()-t0:.2f}s")

    # STEP 7 - save all results in one file for the viewer packer (pack_simulation_data.py).
    np.savez_compressed(
        output_file("simulation_24h.npz"),
        psi_series=psi_series, fs_series=fs_series,
        flow_depth_series=flow_depth_series, cum_erosion=cum_erosion,
        precip_mm=precip_mm, storm_shape=storm_shape,
        grid_shape=np.array([gh, gw]), elevation=elevation,
        has_soil_data=has_soil,
    )
    print("saved simulation_24h.npz")
    # Quick summary printed for the log.
    print(f"FS range over sim: {fs_series.min():.2f} - {fs_series.max():.2f}")
    print(f"nodes ever failed (FS<1): {(fs_series.min(axis=0) < 1).sum()} / {n} "
          f"({100*(fs_series.min(axis=0)<1).mean():.1f}%)")
    print(f"max flow depth reached: {flow_depth_series.max():.3f} m")


# Entry point: `python run_simulation.py`.
if __name__ == "__main__":
    main()
