# =============================================================================
# VALKYRIE | build_graph_v2.py
# Stage   : B. Terrain + soil + rainfall graph (export)
# Purpose : Extends build_graph.py by filling the reserved soil and rainfall feature
#           columns with real values, giving one complete graph file (unified_graph_v2.npz).
#           Not required for the simulation itself: run_simulation.py repeats these steps.
# Reads   : everything build_graph.py reads + SoilGrids + CHIRPS
# Writes  : unified_graph_v2.npz
# Used by : standalone export (e.g. for machine-learning work)
# =============================================================================

"""
Unified mesh-graph: DEM + hydrology + CHIRPS precipitation + SoilGrids-derived
TRIGRS parameters, in MeshGraphNets' node/edge formalism.

Run order matters: this imports build_graph.py's DEM/hydro logic and extends
it with the two new data sources. See build_graph.py's module docstring for
the base schema and MeshGraphNets edge-encoding rationale.

CAVEAT ON SOIL-DERIVED FEATURES: cohesion, friction angle, unit weight, Ks,
theta_sat, theta_res, alpha, and diffusivity are NOT measurements. They are
generic pedotransfer-function estimates from SoilGrids texture/bulk-density
(see soil_ptf.py's module docstring). Treat as order-of-magnitude planning
values only. A `has_soil_data` mask marks nodes outside the actual Mandakini
watershed polygon (~25% of this wider DEM extent), where soil features are
left at 0 and should be excluded from any downstream calculation.
"""
import numpy as np

from build_graph import build_graph, GRAPH_TARGET_W, BORDER_CROP
from soil_ptf import (soilgrids_to_fractions, saxton_rawls_hydraulic,
                       usda_texture_class, texture_to_strength, bulk_density_to_unit_weight)
from resample_to_grid import resample_soilgrids_to_points, resample_chirps_to_points
from paths import data_file, output_file

# Input files, expected in the repo's data/ folder (see data/README.md). The names are
# fixed here because, unlike the DEM/hydrology rasters, these two come from separate sources.
SOILGRIDS_PATH = data_file("soilgrids_mandakini.tif")
CHIRPS_PATH = data_file("chirps-v3.0.sat.2013.06.17.tif")

# same geotransform constants used throughout this project's scripts
PIXEL_SCALE_DEG = 0.0002777777777777778
LAT_TOP0 = 30.85000001111111
LON_LEFT0 = 78.84999998888891


def node_lonlat(gh, gw, step):
    """Longitude/latitude of every graph node, from the DEM's known corner and pixel size."""
    B = BORDER_CROP
    lat_top = LAT_TOP0 - B * PIXEL_SCALE_DEG
    lon_left = LON_LEFT0 + B * PIXEL_SCALE_DEG
    rows = np.arange(gh) * step
    cols = np.arange(gw) * step
    lat = lat_top - rows * PIXEL_SCALE_DEG
    lon = lon_left + cols * PIXEL_SCALE_DEG
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon_grid, lat_grid


def build_unified_graph():
    """Build the terrain graph, then add rainfall and soil-derived values to every node."""
    # Step 1 - base terrain graph (soil/rain columns still zero).
    graph = build_graph()
    gh, gw = graph["grid_shape"]
    # recover the same downsample step build_graph() used internally
    step = round((2160 - 2 * BORDER_CROP) / GRAPH_TARGET_W)

    # Step 2 - find each node's geographic position (needed to sample the other datasets).
    lon_grid, lat_grid = node_lonlat(gh, gw, step)

    # Step 3 - SoilGrids texture -> strength, permeability and weight of the soil.
    print("resampling SoilGrids onto graph nodes ...")
    soil = resample_soilgrids_to_points(SOILGRIDS_PATH, lon_grid, lat_grid)
    sand_pct, silt_pct, clay_pct, bd_gcm3 = soilgrids_to_fractions(
        soil["sand_gkg"], soil["clay_gkg"], soil["bdod_cgcm3"])
    hydro = saxton_rawls_hydraulic(sand_pct, clay_pct)
    texture = usda_texture_class(sand_pct, silt_pct, clay_pct)
    cohesion, friction = texture_to_strength(texture)
    uws = bulk_density_to_unit_weight(bd_gcm3)
    has_soil = soil["has_soil_data"]

    # Step 4 - CHIRPS daily rainfall at every node.
    print("resampling CHIRPS onto graph nodes ...")
    precip = resample_chirps_to_points(CHIRPS_PATH, lon_grid, lat_grid)

    # Step 5 - write the new values into the reserved feature columns. Soil values outside
    # the mapped watershed are left at 0 and flagged by the has_soil_data column.
    names = graph["feature_names"]
    x = graph["x"]

    def set_col(name, values_2d, mask=None):
        """Write a 2-D array into the named feature column (optionally zeroing cells outside `mask`)."""
        col = names.index(name)
        flat = values_2d.reshape(-1).astype(np.float32)
        if mask is not None:
            flat = np.where(mask.reshape(-1), flat, 0.0)
        x[:, col] = flat

    set_col("precip_mm", precip)  # CHIRPS covers the whole area, no masking needed
    set_col("cohesion_kpa", cohesion, has_soil)
    set_col("friction_angle_deg", friction, has_soil)
    set_col("unit_weight_kn_m3", uws, has_soil)
    set_col("ksat_m_s", hydro["ksat_m_s"], has_soil)
    set_col("diffusivity_m2_s", hydro["diffusivity_m2_s"], has_soil)
    set_col("theta_sat", hydro["theta_sat"], has_soil)
    set_col("theta_res", hydro["theta_res"], has_soil)
    set_col("alpha_1_m", hydro["alpha_1_m"], has_soil)

    graph["x"] = x
    # Append has_soil_data as a final feature column.
    graph["feature_names"] = names + ["has_soil_data"]
    graph["x"] = np.concatenate([x, has_soil.reshape(-1, 1).astype(np.float32)], axis=1)
    graph["lon_grid"] = lon_grid
    graph["lat_grid"] = lat_grid
    return graph


# Running this file directly builds the full graph, prints a per-feature summary
# (min / max / NaN count) and saves unified_graph_v2.npz.
if __name__ == "__main__":
    g = build_unified_graph()
    print()
    print("nodes:", g["x"].shape[0], " features/node:", g["x"].shape[1])
    print("feature order:", g["feature_names"])
    for i, name in enumerate(g["feature_names"]):
        col = g["x"][:, i]
        print(f"{name:28s} min={col.min():>12.4f} max={col.max():>12.4f} nan={np.isnan(col).sum()}")
    np.savez_compressed(
        output_file("unified_graph_v2.npz"),
        pos=g["pos"], x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"],
        grid_shape=g["grid_shape"], feature_names=np.array(g["feature_names"]),
        lon_grid=g["lon_grid"], lat_grid=g["lat_grid"],
    )
    print("\nsaved unified_graph_v2.npz")
