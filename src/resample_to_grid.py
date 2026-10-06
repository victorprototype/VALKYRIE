# =============================================================================
# VALKYRIE | resample_to_grid.py
# Stage   : A. Data readers (alignment to the DEM grid)
# Purpose : Puts SoilGrids (Mollweide projection) and CHIRPS rainfall (global lat/lon grid)
#           onto the DEM's grid, i.e. gives every terrain node a soil value and a rain value.
# Reads   : soilgrids_mandakini.tif, chirps-*.tif, target lon/lat of each node
# Writes  : dict of resampled soil arrays / rainfall array
# Used by : run_simulation.py, build_graph_v2.py, build_dem_3d_viewer.py
# =============================================================================

"""
Resamples SoilGrids (Mollweide, tiled-LZW) and CHIRPS (WGS84, global) onto
an arbitrary set of target (lon, lat) points -- used to populate the DEM
graph's node features.
"""
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from PIL import Image

from read_soilgrids import read_tiled_lzw_tiff

Image.MAX_IMAGE_PIXELS = None

# Earth radius used by the ESRI:54009 'World Mollweide' projection that SoilGrids is stored in.
MOLLWEIDE_R = 6371007.1809  # ESRI:54009 World Mollweide authalic sphere radius


def mollweide_inverse(x, y, lon0=0.0):
    """Convert Mollweide map coordinates (metres) back to longitude/latitude (degrees).

    The SoilGrids file positions pixels in metres on a Mollweide map; the DEM uses
    plain lon/lat, so we need this inverse projection to line the two up.
    """
    # Step 1: auxiliary angle from y. Step 2: latitude from that angle. Step 3: longitude from x.
    theta = np.arcsin(np.clip(y / (MOLLWEIDE_R * np.sqrt(2)), -1, 1))
    lat = np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi, -1, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        lon = lon0 + (np.pi * x) / (2 * MOLLWEIDE_R * np.sqrt(2) * np.cos(theta))
    return np.degrees(lon), np.degrees(lat)


def load_soilgrids_lonlat(path):
    """Read SoilGrids and return, for every pixel, its lon/lat plus sand, clay and
    bulk density (0-5 cm layer), and a mask of pixels that hold real data.
    """
    arr = read_tiled_lzw_tiff(path).astype(np.float32)
    H, W, _ = arr.shape
    # geotransform (from the file's own tiepoint/pixel-scale tags)
    tx, ty = 7189735.290828891, 3743288.1316389814
    scale = 250.0
    # Metre coordinates of every pixel centre -> converted to lon/lat below.
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = tx + cols * scale
    Y = ty - rows * scale
    lon, lat = mollweide_inverse(X, Y)

    # Bands 0, 2 and 4 are the 0-5 cm layers of sand, clay and bulk density.
    sand, clay, bdod = arr[:, :, 0], arr[:, :, 2], arr[:, :, 4]
    valid = ~((bdod == 0) & (sand == 0))  # outside actual watershed polygon
    return lon, lat, sand, clay, bdod, valid


def resample_soilgrids_to_points(path, target_lon, target_lat):
    """Interpolate sand / clay / bulk density onto arbitrary target points.

    Also returns `has_soil_data`: True only where a target point lies inside the
    area SoilGrids actually covers (the mapped watershed), so callers can treat
    everything else as 'no soil information'.
    """
    lon, lat, sand, clay, bdod, valid = load_soilgrids_lonlat(path)
    # Source points = valid SoilGrids pixels; target points = the graph nodes.
    pts = np.stack([lon[valid], lat[valid]], axis=1)
    tgt = np.stack([target_lon.ravel(), target_lat.ravel()], axis=1)

    out = {}
    # For each variable: linear interpolation inside the data, nearest-neighbour
    # to fill small gaps at the edges.
    for name, src in [("sand_gkg", sand), ("clay_gkg", clay), ("bdod_cgcm3", bdod)]:
        vals = src[valid]
        interp = griddata(pts, vals, tgt, method="linear")
        nearest = griddata(pts, vals, tgt, method="nearest")
        interp = np.where(np.isnan(interp), nearest, interp)  # fill edge gaps
        out[name] = interp.reshape(target_lon.shape).astype(np.float32)

    # a node counts as "has real basin soil data" only if inside the convex
    # hull of valid source points AND not itself flagged invalid
    hull_check = griddata(pts, np.ones(len(pts)), tgt, method="linear")
    out["has_soil_data"] = (~np.isnan(hull_check)).reshape(target_lon.shape)
    return out


def resample_chirps_to_points(path, target_lon, target_lat):
    """Bilinear-interpolate CHIRPS daily rainfall (mm) onto target points.

    CHIRPS is a global 0.05-degree grid (about 5.5 km), far coarser than the DEM,
    so the result is smooth: it does not contain real fine-scale rain variation.
    """
    chirps = np.array(Image.open(path))  # (2400, 7200), WGS84 0.05deg, tiepoint (-180,60)
    lon0, lat0, scale = -180.0, 60.0, 0.05
    # Longitude/latitude of each CHIRPS pixel centre.
    lons = lon0 + (np.arange(chirps.shape[1]) + 0.5) * scale
    lats = lat0 - (np.arange(chirps.shape[0]) + 0.5) * scale  # descending
    # RegularGridInterpolator needs ascending coords
    lats_asc = lats[::-1]
    chirps_asc = chirps[::-1, :]
    interp = RegularGridInterpolator((lats_asc, lons), chirps_asc, method="linear",
                                      bounds_error=False, fill_value=None)
    pts = np.stack([target_lat.ravel(), target_lon.ravel()], axis=1)
    vals = interp(pts).reshape(target_lon.shape)
    return np.clip(vals, 0, None).astype(np.float32)  # negative from interpolation noise -> 0


# Manual smoke test on a small grid (uses the files in data/).
if __name__ == "__main__":
    # quick smoke test against the DEM graph's actual node grid
    from paths import data_file

    gh, gw = 50, 55  # small test grid for speed
    lat_top, lat_bottom = 30.848333344444445, 30.301666677777778
    lon_left, lon_right = 78.85166665555558, 79.44833332222224
    lats = np.linspace(lat_top, lat_bottom, gh)
    lons = np.linspace(lon_left, lon_right, gw)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    soil = resample_soilgrids_to_points(data_file("soilgrids_mandakini.tif"), lon_grid, lat_grid)
    for k, v in soil.items():
        if k == "has_soil_data":
            print(k, "fraction True:", v.mean())
        else:
            print(k, "min", np.nanmin(v), "max", np.nanmax(v), "nan_count", np.isnan(v).sum())

    precip = resample_chirps_to_points(data_file("chirps-v3.0.sat.2013.06.17.tif"), lon_grid, lat_grid)
    print("precip mm: min", precip.min(), "max", precip.max(), "mean", precip.mean())
