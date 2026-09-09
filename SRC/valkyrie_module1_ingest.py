"""
VALKYRIE - Module 1: Data Ingestor
===================================
Loads DEM, multi-depth SoilGrids properties, and daily CHIRPS rainfall,
reconciles them onto ONE common projected (meters) grid, computes terrain
derivatives, and builds the spatial node/edge graph consumed by Module 2
(Graphical Network Creator).

Handles the real-world mismatch this dataset has: DEM/soil/CHIRPS typically
arrive in geographic (degree) coordinates at three different resolutions
and extents. Everything is reprojected/resampled onto the DEM's native
grid, converted to a local UTM CRS so distances are in meters (required
for slope and the physics engine), before graph construction.
"""
from __future__ import annotations
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

EPS = 1e-8


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class IngestConfig:
    dem_path: str

    # soil rasters keyed by (property, depth) -> file path
    # properties: bdod (bulk density), clay, sand, silt (all as %/g-kg per
    # SoilGrids convention - keep raw units, Module 3 handles conversion)
    soil_paths: dict = field(default_factory=dict)

    # either an explicit sorted list of 30 daily CHIRPS paths, or a directory
    # + glob pattern (files sorted alphabetically - CHIRPS filenames embed
    # the date and sort chronologically by default, e.g. chirps-v2.0.2013.06.01.tif)
    chirps_paths: Optional[list] = None
    chirps_dir: Optional[str] = None
    chirps_glob: str = "*.tif"

    downsample_factor: int = 1  # applied AFTER reprojection, on the DEM grid
    soil_resampling: Resampling = Resampling.bilinear
    rainfall_resampling: Resampling = Resampling.bilinear


# =============================================================================
# CRS / GRID HELPERS
# =============================================================================
def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _load_and_project_dem(dem_path: str):
    """Reads the DEM; if it's in geographic (degree) coordinates, reprojects
    it to the local UTM zone so pixel size is in meters. Returns
    (elevation_array, transform, crs)."""
    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        if src.crs is None:
            raise ValueError(f"{dem_path} has no CRS defined - cannot georeference.")

        if src.crs.is_geographic:
            lon_c = (src.bounds.left + src.bounds.right) / 2
            lat_c = (src.bounds.top + src.bounds.bottom) / 2
            dst_crs = CRS.from_epsg(_utm_epsg_from_lonlat(lon_c, lat_c))
            dst_transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            dem = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=dem,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                src_nodata=nodata, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            return dem, dst_transform, dst_crs
        else:
            dem = src.read(1).astype(np.float32)
            if nodata is not None:
                dem = np.where(dem == nodata, np.nan, dem)
            return dem, src.transform, src.crs


def _downsample_mean(arr, transform, factor):
    """Block-mean downsampling (NaN-aware) by an integer factor, adjusting
    the affine transform to match the coarser grid."""
    if factor <= 1:
        return arr, transform
    H, W = arr.shape
    Hc, Wc = H // factor, W // factor
    trimmed = arr[: Hc * factor, : Wc * factor]
    reshaped = trimmed.reshape(Hc, factor, Wc, factor)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(reshaped, axis=(1, 3))
    new_transform = transform * transform.scale(factor, factor)
    return out.astype(np.float32), new_transform


def _align_to_grid(path, ref_transform, ref_crs, ref_shape, resampling):
    """Reprojects/resamples any raster onto the exact reference grid
    (same transform, crs, shape) as the working DEM."""
    with rasterio.open(path) as src:
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            src_nodata=src.nodata, dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


# =============================================================================
# MAIN INGESTION
# =============================================================================
REQUIRED_SOIL_KEYS = [
    ("bdod", "0-5cm"), ("bdod", "5-15cm"),
    ("clay", "0-5cm"), ("clay", "5-15cm"),
    ("sand", "0-5cm"), ("sand", "5-15cm"),
    ("silt", "0-5cm"), ("silt", "5-15cm"),
]

# node_features column order - Module 2/3 rely on this exact order
NODE_FEATURE_NAMES = [
    "elev_norm", "slope_rad", "aspect_sin", "aspect_cos",
    "bdod_0_5", "bdod_5_15",
    "clay_0_5", "clay_5_15",
    "sand_0_5", "sand_5_15",
    "silt_0_5", "silt_5_15",
]


def _resolve_chirps_paths(cfg: IngestConfig) -> list[str]:
    if cfg.chirps_paths:
        paths = list(cfg.chirps_paths)
    elif cfg.chirps_dir:
        paths = sorted(glob.glob(os.path.join(cfg.chirps_dir, cfg.chirps_glob)))
    else:
        raise ValueError("Provide either chirps_paths or chirps_dir in IngestConfig.")
    if len(paths) == 0:
        raise FileNotFoundError("No CHIRPS rasters found - check chirps_dir/chirps_glob.")
    return paths


def ingest(cfg: IngestConfig):
    # 1. DEM -> working projected grid (meters)
    dem, transform, crs = _load_and_project_dem(cfg.dem_path)
    dem, transform = _downsample_mean(dem, transform, cfg.downsample_factor)
    H, W = dem.shape
    cell_res = abs(transform.a)  # meters/pixel (square pixels assumed post-reproj)

    # 2. Soil layers -> resampled onto the working grid
    missing = [k for k in REQUIRED_SOIL_KEYS if k not in cfg.soil_paths]
    if missing:
        raise ValueError(f"Missing soil raster paths for: {missing}")

    soil_layers = {}
    for key in REQUIRED_SOIL_KEYS:
        soil_layers[key] = _align_to_grid(
            cfg.soil_paths[key], transform, crs, (H, W), cfg.soil_resampling
        )

    # 3. CHIRPS -> resampled onto the working grid, one layer per day
    chirps_files = _resolve_chirps_paths(cfg)
    rainfall_grids = np.stack([
        _align_to_grid(p, transform, crs, (H, W), cfg.rainfall_resampling)
        for p in chirps_files
    ], axis=0)  # (n_days, H, W)
    # rainfall nodata -> 0mm rather than excluding nodes; missing a day of
    # rain data over land is far more likely a coastal/edge sampling gap
    # than "no rain", and zero-filling avoids shrinking the valid terrain mask
    rainfall_grids = np.nan_to_num(rainfall_grids, nan=0.0)

    # 4. Valid mask: DEM + all soil layers must be present (rainfall excluded
    # from the mask per the above)
    valid = ~np.isnan(dem)
    for arr in soil_layers.values():
        valid &= ~np.isnan(arr)

    # 5. Terrain derivatives (finite differences, now correctly in meters)
    dzdy, dzdx = np.gradient(dem, cell_res)
    slope_rad_full = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad_full = np.arctan2(-dzdx, dzdy)  # 0 = north-facing descent direction

    # a pixel's own elevation can be valid while a NEIGHBOR feeding its
    # gradient stencil is NaN (common at the projected-grid edge fringe left
    # by reprojection skew) - exclude those too, or slope/aspect leak NaN
    # into otherwise-valid nodes
    valid &= ~np.isnan(slope_rad_full) & ~np.isnan(aspect_rad_full)

    n_nodes = int(valid.sum())
    if n_nodes == 0:
        raise ValueError("Zero valid overlapping pixels across DEM + soil layers "
                          "after alignment - check that all inputs actually cover "
                          "the same geographic area.")

    # 6. Node table
    node_id_grid = -np.ones((H, W), dtype=np.int64)
    node_id_grid[valid] = np.arange(n_nodes)

    dem_valid = dem[valid]
    elev_norm = (dem_valid - np.nanmin(dem_valid)) / (
        np.nanmax(dem_valid) - np.nanmin(dem_valid) + EPS
    )

    feature_cols = [
        elev_norm,
        slope_rad_full[valid],
        np.sin(aspect_rad_full[valid]),
        np.cos(aspect_rad_full[valid]),
        soil_layers[("bdod", "0-5cm")][valid],
        soil_layers[("bdod", "5-15cm")][valid],
        soil_layers[("clay", "0-5cm")][valid],
        soil_layers[("clay", "5-15cm")][valid],
        soil_layers[("sand", "0-5cm")][valid],
        soil_layers[("sand", "5-15cm")][valid],
        soil_layers[("silt", "0-5cm")][valid],
        soil_layers[("silt", "5-15cm")][valid],
    ]
    node_features = np.stack(feature_cols, axis=1).astype(np.float32)

    # 7. 4-way grid connectivity edges (only between valid neighbor pixels)
    senders, receivers, distances, delta_z = [], [], [], []
    ii_all, jj_all = np.where(valid)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = ii_all + di, jj_all + dj
        in_bounds = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
        ii_b, jj_b, ni_b, nj_b = ii_all[in_bounds], jj_all[in_bounds], ni[in_bounds], nj[in_bounds]
        nbr_valid = valid[ni_b, nj_b]
        ii_v, jj_v, ni_v, nj_v = ii_b[nbr_valid], jj_b[nbr_valid], ni_b[nbr_valid], nj_b[nbr_valid]

        senders.append(node_id_grid[ii_v, jj_v])
        receivers.append(node_id_grid[ni_v, nj_v])
        dist = cell_res * np.sqrt(di**2 + dj**2)
        distances.append(np.full(ii_v.shape[0], dist, dtype=np.float32))
        delta_z.append((dem[ni_v, nj_v] - dem[ii_v, jj_v]).astype(np.float32))

    senders = np.concatenate(senders).astype(np.int32)
    receivers = np.concatenate(receivers).astype(np.int32)
    edge_features = np.stack(
        [np.concatenate(distances), np.concatenate(delta_z)], axis=1
    ).astype(np.float32)

    # 8. Rainfall forcing at valid node locations, shape (n_days, n_nodes)
    rainfall_timeseries = rainfall_grids[:, valid]

    graph = dict(
        node_features=node_features,
        node_feature_names=NODE_FEATURE_NAMES,
        senders=senders,
        receivers=receivers,
        edge_features=edge_features,
        n_node=n_nodes,
        n_edge=senders.shape[0],
        node_id_grid=node_id_grid,
        valid_mask=valid,
        raw_elevation=dem_valid,
        cell_res=cell_res,
        out_shape=(H, W),
        crs=crs.to_string(),
        transform=transform,
    )
    return graph, rainfall_timeseries, chirps_files


if __name__ == "__main__":
    cfg = IngestConfig(
        dem_path="testdata/dem.tif",
        soil_paths={
            ("bdod", "0-5cm"): "testdata/bdod_0-5cm.tif",
            ("bdod", "5-15cm"): "testdata/bdod_5-15cm.tif",
            ("clay", "0-5cm"): "testdata/clay_0-5cm.tif",
            ("clay", "5-15cm"): "testdata/clay_5-15cm.tif",
            ("sand", "0-5cm"): "testdata/sand_0-5cm.tif",
            ("sand", "5-15cm"): "testdata/sand_5-15cm.tif",
            ("silt", "0-5cm"): "testdata/silt_0-5cm.tif",
            ("silt", "5-15cm"): "testdata/silt_5-15cm.tif",
        },
        chirps_dir="testdata",
        chirps_glob="chirps-v2.0.2013.06.*.tif",
        downsample_factor=1,
    )
    graph, rainfall, files = ingest(cfg)
    print("CRS after reprojection:", graph["crs"])
    print("cell_res (m):", graph["cell_res"])
    print("out_shape:", graph["out_shape"])
    print("n_nodes:", graph["n_node"], " n_edges:", graph["n_edge"])
    print("node_features shape:", graph["node_features"].shape,
          "(expect 12 cols):", graph["node_feature_names"])
    print("edge_features shape:", graph["edge_features"].shape)
    print("rainfall_timeseries shape:", rainfall.shape, "(expect 30, n_nodes)")
    print("rainfall day-16 mean (should spike):", rainfall[15].mean())
    print("rainfall day-1 mean (baseline):", rainfall[0].mean())
    print("n_chirps_files_used:", len(files))
    print("sample node features (row 0):", graph["node_features"][0])
    print("any NaNs in node_features:", np.isnan(graph["node_features"]).any())
    print("any NaNs in rainfall:", np.isnan(rainfall).any())
