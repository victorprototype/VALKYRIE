"""
Build an interactive 3D Three.js viewer from a Copernicus GLO-30 DEM + gdaldem
derivative rasters (hillshade, hillshade-color, color-relief, slope, aspect,
roughness).

Run this on Kaggle (or anywhere with the input files on disk). It writes a
single self-contained HTML file you can download and open in any browser --
no server needed.

Usage on Kaggle:
    python build_dem_3d_kaggle.py

Edit DATA_DIR / FILES below if your filenames differ.
"""
import base64
import io
import json
import os

import numpy as np
import matplotlib
from PIL import Image

try:
    from soil_ptf import (soilgrids_to_fractions, saxton_rawls_hydraulic,
                           usda_texture_class, texture_to_strength, bulk_density_to_unit_weight)
    from resample_to_grid import resample_soilgrids_to_points, resample_chirps_to_points
    _EXTERNAL_DATA_MODULES_AVAILABLE = True
except ImportError:
    _EXTERNAL_DATA_MODULES_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# Config -- edit these for your environment
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/thevictorprototype/dem-sim-data"
OUT_PATH = "/kaggle/working/valkyrie/physics/dem_3d_viewer.html"

FILES = {
    "dem": "output_hh.tif",
    "hillshade": "viz.hh_hillshade.tif",
    "hillshade_color": "viz.hh_hillshade-color.tif",
    "color_relief": "viz.hh_color-relief.tif",
    "roughness": "viz.hh_roughness.tif",
    "aspect": "viz.hh_aspect.tif",
    "slope": "viz.hh_slope.tif",
}

# Optional hydrology derivatives -- rendered as extra switchable layers if
# present. Missing files here are skipped (not an error), so this list is
# safe to edit even if your dataset doesn't have all of them.
OPTIONAL_FILES = {
    "flow_accum": "D8_Flow_Accumulation.tif",
    "flow_dir_d8": "D8_Flow_Direction.tif",
    "flow_dir_dinf": "DInf_Flow_Direction.tif",
    "sca": "DInf_Specific_Catchment_Area.tif",
    "twi": "TWI.tif",
    "hydro_dem": "Hydro_Conditioned_DEM.tif",
}

# External (non-co-registered) data sources -- these are NOT on the DEM's
# pixel grid or CRS, so they go through a resampling step (see
# resample_to_grid.py) rather than the simple crop-and-slice OPTIONAL_FILES
# path above. Set either to None to skip it.
CHIRPS_FILE = "chirps-v3.0.sat.2013.06.17.tif"
SOILGRIDS_FILE = "soilgrids_mandakini.tif"

BORDER_CROP = 6          # trim gdaldem's nodata edge ring (px)
GEOMETRY_TARGET_W = 420  # mesh resolution (columns) -- raise for more detail,
                         # lower to shrink the output file / speed up the browser
TEXTURE_TARGET_W = 1200  # texture resolution (columns)
VERTICAL_EXAGGERATION_DEFAULT = 1.6
NODATA = -9000           # gdaldem slope/aspect/roughness nodata sentinel (<=)

# ---------------------------------------------------------------------------
# Raster IO -- prefers rasterio (gives you a real CRS/transform), falls back
# to Pillow, then to tifffile for anything Pillow's TIFF reader can't handle
# (e.g. float64 samples, which Pillow's "F" mode does not support).
# ---------------------------------------------------------------------------
def _load_with_rasterio(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1) if src.count == 1 else np.moveaxis(src.read(), 0, -1)
        transform = src.transform
        return arr, transform


def _load_with_pil(path):
    return np.array(Image.open(path)), None


def _load_with_tifffile(path):
    import tifffile
    return tifffile.imread(path), None


def _norm(s):
    """Lowercase and strip everything but letters/digits, so 'viz_hh_slope.tif',
    'viz.hh-slope.TIF', and 'VizHHSlope.tif' all compare equal."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _resolve_path(filename):
    """Find the file even if it's nested in a subfolder, or the exact name
    uses different separators/case than requested -- searches DATA_DIR
    recursively and matches on a normalized (punctuation-stripped) stem."""
    direct = os.path.join(DATA_DIR, filename)
    if os.path.isfile(direct):
        return direct
    wanted_norm = _norm(os.path.splitext(filename)[0])
    candidates = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            stem_norm = _norm(os.path.splitext(f)[0])
            if stem_norm == wanted_norm:
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a file matching '{filename}' under {DATA_DIR}."
        )
    if len(candidates) > 1:
        print(f"[warn] multiple matches for '{filename}', using first: {candidates}")
    return candidates[0]


def load(filename):
    path = _resolve_path(filename)
    try:
        arr, transform = _load_with_rasterio(path)
        return arr, transform
    except Exception:
        pass
    try:
        arr, transform = _load_with_pil(path)
        return arr, transform
    except Exception:
        pass
    # last resort: tifffile handles cases Pillow's TIFF reader rejects
    # (e.g. float64 samples), at the cost of not returning a geotransform
    arr, transform = _load_with_tifffile(path)
    return arr, transform


def load_optional(filename):
    """Like load(), but returns (None, None) instead of raising if the file
    can't be found or can't be read -- used for the hydrology extras, which
    may not exist in every dataset."""
    try:
        return load(filename)
    except Exception as e:
        print(f"[skip] '{filename}' not available ({e})")
        return None, None


def main():
    dem, transform = load(FILES["dem"])
    hillshade, _ = load(FILES["hillshade"])
    hillshade_color, _ = load(FILES["hillshade_color"])
    color_relief, _ = load(FILES["color_relief"])
    roughness, _ = load(FILES["roughness"])
    aspect, _ = load(FILES["aspect"])
    slope, _ = load(FILES["slope"])

    dem = dem.astype(np.float32)
    H, W = dem.shape
    B = BORDER_CROP

    def crop(a):
        return a[B:H - B, B:W - B, ...] if a.ndim == 2 else a[B:H - B, B:W - B, :]

    dem = crop(dem)
    hillshade = crop(hillshade.astype(np.float32))
    hillshade_color = crop(hillshade_color.astype(np.uint8))
    color_relief = crop(color_relief.astype(np.uint8))
    roughness = crop(roughness.astype(np.float32))
    aspect = crop(aspect.astype(np.float32))
    slope = crop(slope.astype(np.float32))
    H, W = dem.shape

    # ---- optional hydrology layers (skipped individually if missing/mismatched) ----
    optional_arrays = {}
    for key, fname in OPTIONAL_FILES.items():
        arr, _ = load_optional(fname)
        if arr is None:
            continue
        arr = arr.astype(np.float32)
        if arr.shape[:2] != (H + 2 * B, W + 2 * B):
            print(f"[skip] '{fname}' shape {arr.shape} doesn't match the DEM grid, skipping")
            continue
        optional_arrays[key] = crop(arr)

    # ---- real-world extent ----
    if transform is not None:
        # rasterio Affine: x = c + a*col, y = f + e*row
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        lon_left = transform.c + B * px_w
        lat_top = transform.f - B * px_h
    else:
        # fall back to the pixel scale baked into this dataset (WGS84, ~1 arcsec)
        px_w = px_h = 0.0002777777777777778
        lon_left = 78.84999998888891 + B * px_w
        lat_top = 30.85000001111111 - B * px_h

    lat_bottom = lat_top - H * px_h
    lon_right = lon_left + W * px_w
    avg_lat = (lat_top + lat_bottom) / 2
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(avg_lat))
    width_m = (lon_right - lon_left) * m_per_deg_lon
    height_m = (lat_top - lat_bottom) * m_per_deg_lat

    elev_min, elev_max = float(np.nanmin(dem)), float(np.nanmax(dem))

    # ---- geometry (heightmap) ----
    step = max(1, round(W / GEOMETRY_TARGET_W))
    geo_dem = dem[::step, ::step]
    gh, gw = geo_dem.shape
    heights_b64 = base64.b64encode(geo_dem.astype(np.float32).tobytes()).decode("ascii")

    meta = {
        "gridW": gw, "gridH": gh,
        "widthM": width_m, "heightM": height_m,
        "elevMin": elev_min, "elevMax": elev_max,
        "latTop": lat_top, "latBottom": lat_bottom,
        "lonLeft": lon_left, "lonRight": lon_right,
        "defaultExagg": VERTICAL_EXAGGERATION_DEFAULT,
    }

    # ---- textures ----
    tstep = max(1, round(W / TEXTURE_TARGET_W))

    def to_rgb_uint8(arr3):
        return Image.fromarray(arr3[::tstep, ::tstep, :])

    def grayscale_to_rgb(arr):
        a = arr[::tstep, ::tstep].astype(np.uint8)
        return Image.fromarray(np.stack([a, a, a], axis=-1))

    # ---- legend helpers ----
    def _bar_from_cmap(cmap_name, w=240, h=16):
        xs = np.linspace(0, 1, w)
        cmap = matplotlib.colormaps[cmap_name]
        row = (cmap(xs)[:, :3] * 255).astype(np.uint8)
        bar = np.tile(row[np.newaxis, :, :], (h, 1, 1))
        buf = io.BytesIO()
        Image.fromarray(bar).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _bar_from_samples(rgb_stops, w=240, h=16):
        """Build a smooth gradient bar image by interpolating between a
        small number of (already computed) RGB stop colors."""
        n = len(rgb_stops)
        xs_src = np.linspace(0, 1, n)
        xs_dst = np.linspace(0, 1, w)
        stops = np.array(rgb_stops, dtype=np.float64)
        r = np.interp(xs_dst, xs_src, stops[:, 0])
        g = np.interp(xs_dst, xs_src, stops[:, 1])
        b = np.interp(xs_dst, xs_src, stops[:, 2])
        row = np.stack([r, g, b], axis=-1).astype(np.uint8)
        bar = np.tile(row[np.newaxis, :, :], (h, 1, 1))
        buf = io.BytesIO()
        Image.fromarray(bar).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _fmt_num(v):
        """Compact human-readable number for legend tick labels."""
        av = abs(v)
        if av >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if av >= 1_000:
            return f"{v/1_000:.1f}k"
        if av >= 100:
            return f"{v:.0f}"
        if av >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    def colorize(arr, cmap_name, vmin=None, vmax=None, unit="", description=""):
        a = arr[::tstep, ::tstep].copy()
        mask = a <= NODATA
        valid = a[~mask]
        if vmin is None:
            vmin = float(np.percentile(valid, 1))
        if vmax is None:
            vmax = float(np.percentile(valid, 99))
        a = np.clip(a, vmin, vmax)
        norm = (a - vmin) / max(1e-6, (vmax - vmin))
        cmap = matplotlib.colormaps[cmap_name]
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        rgb[mask] = [20, 20, 24]
        image = Image.fromarray(rgb)
        legend = {
            "kind": "gradient",
            "description": description,
            "bar": _bar_from_cmap(cmap_name),
            "minLabel": f"{_fmt_num(vmin)}{unit}",
            "maxLabel": f"{_fmt_num(vmax)}{unit}",
        }
        return image, legend

    def colorize_log(arr, cmap_name, unit="", description=""):
        """For heavily right-skewed rasters (flow accumulation, specific
        catchment area) -- log1p then percentile-clip so the drainage
        network stands out instead of being washed out by a few huge cells."""
        a = arr[::tstep, ::tstep].copy()
        mask = (a <= NODATA) | ~np.isfinite(a) | (a < 0)
        a = np.where(mask, 0, a)
        a = np.log1p(a)
        valid = a[~mask]
        vmin = float(np.percentile(valid, 2)) if valid.size else 0.0
        vmax = float(np.percentile(valid, 99.5)) if valid.size else 1.0
        a = np.clip(a, vmin, vmax)
        norm = (a - vmin) / max(1e-6, (vmax - vmin))
        cmap = matplotlib.colormaps[cmap_name]
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        rgb[mask] = [10, 10, 12]
        image = Image.fromarray(rgb)
        # ticks in real (non-log) units, evenly spaced in *displayed* color
        # position -- not in raw value -- since that's what the bar shows
        tick_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        tick_values = [float(np.expm1(vmin + t * (vmax - vmin))) for t in tick_positions]
        legend = {
            "kind": "gradient",
            "description": description,
            "bar": _bar_from_cmap(cmap_name),
            "minLabel": f"{_fmt_num(tick_values[0])}{unit}",
            "maxLabel": f"{_fmt_num(tick_values[-1])}{unit}",
            "midLabel": f"{_fmt_num(tick_values[2])}{unit}",
            "logScale": True,
        }
        return image, legend

    def colorize_cyclic_direction(arr, description=""):
        """DInf flow direction is a continuous angle -- auto-detect radians
        vs degrees, then use the same cyclic (hsv) treatment as aspect.
        Nodata here is a small sentinel (commonly -1), not a large negative
        like gdaldem's -9999, so mask on sign/finiteness rather than NODATA."""
        a = arr[::tstep, ::tstep].copy()
        mask = (a < 0) | ~np.isfinite(a)
        valid = a[~mask]
        if valid.size and float(np.nanmax(valid)) <= 6.5:
            a = np.degrees(a)  # looks like radians (0-2pi) -> convert
        a = np.where(mask, 0, a)
        norm = np.clip(a, 0, 360) / 360.0
        cmap = matplotlib.colormaps["hsv"]
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        rgb[mask] = [20, 20, 24]
        image = Image.fromarray(rgb)
        legend = {
            "kind": "cyclic",
            "description": description,
            "bar": _bar_from_cmap("hsv"),
            "ticks": [{"pos": 0.0, "label": "0\u00b0"}, {"pos": 0.25, "label": "90\u00b0"},
                      {"pos": 0.5, "label": "180\u00b0"}, {"pos": 0.75, "label": "270\u00b0"},
                      {"pos": 1.0, "label": "360\u00b0"}],
        }
        return image, legend

    def colorize_aspect(arr, description=""):
        """gdaldem aspect uses a documented, fixed convention: 0 deg = North,
        measured clockwise (90=East, 180=South, 270=West). Safe to label
        with actual compass letters, unlike the DInf direction raster below
        whose axis convention isn't independently confirmed here."""
        image, legend = colorize(arr, "hsv", vmin=0, vmax=360, description=description)
        legend["kind"] = "cyclic"
        legend["ticks"] = [{"pos": 0.0, "label": "N"}, {"pos": 0.25, "label": "E"},
                            {"pos": 0.5, "label": "S"}, {"pos": 0.75, "label": "W"},
                            {"pos": 1.0, "label": "N"}]
        legend.pop("minLabel", None)
        legend.pop("maxLabel", None)
        return image, legend

    def colorize_d8_categorical(arr, description=""):
        """D8 flow direction codes come in one of two conventions depending
        on the tool that generated them:
          - TauDEM style: sequential integers 1-8
          - ESRI/ArcGIS style: powers of two 1,2,4,8,16,32,64,128 (8 = a
            bitmask position, not a magnitude)
        Auto-detect which one this file uses and map each distinct code to
        its own flat color -- these are directions, not a continuum, so a
        gradient colormap would be misleading."""
        a = arr[::tstep, ::tstep]
        mask = ~np.isfinite(a) | (a < 0)
        codes_raw = np.where(mask, 0, np.round(a)).astype(np.int64)
        present = set(np.unique(codes_raw).tolist())
        powers_of_two = {1, 2, 4, 8, 16, 32, 64, 128}
        esri_convention = present - {0} <= powers_of_two
        if esri_convention:
            # ESRI's documented bit -> direction mapping
            code_to_index = {0: 0, 1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6, 64: 7, 128: 8}
            code_to_dir = {0: "flat / nodata", 1: "E", 2: "SE", 4: "S", 8: "SW",
                           16: "W", 32: "NW", 64: "N", 128: "NE"}
        else:
            # sequential 1-8 (commonly TauDEM); the exact compass mapping
            # for this convention isn't independently confirmed here, so
            # label by code number rather than guessing a direction
            code_to_index = {i: i for i in range(9)}
            code_to_dir = {0: "flat / nodata", **{i: f"code {i}" for i in range(1, 9)}}
        lut_size = int(max(present, default=0)) + 1
        lut = np.zeros(lut_size, dtype=np.int64)
        for code, idx in code_to_index.items():
            if code < lut_size:
                lut[code] = idx
        codes_raw = np.clip(codes_raw, 0, lut_size - 1)
        idx = lut[codes_raw]
        palette = (np.array(matplotlib.colormaps["tab10"](np.linspace(0, 1, 9)))[:, :3] * 255).astype(np.uint8)
        rgb = palette[idx]
        rgb[mask] = [20, 20, 24]
        image = Image.fromarray(rgb)

        items = []
        for code in sorted(code_to_index.keys()):
            if code == 0:
                continue  # skip flat/nodata swatch -- not a real direction
            hexcolor = "#%02x%02x%02x" % tuple(int(c) for c in palette[code_to_index[code]])
            items.append({"color": hexcolor, "label": code_to_dir.get(code, str(code))})
        legend = {
            "kind": "categorical",
            "description": description + (
                "" if esri_convention else
                " (sequential direction codes; exact compass mapping for this "
                "convention isn't independently verified here, so codes are "
                "labeled by number rather than compass letter.)"
            ),
            "items": items,
        }
        return image, legend

    def sampled_gradient_legend(rgb_full, description, n_bins=14):
        """Empirical elevation legend for the two pre-rendered RGB textures
        (color-relief, hillshade-color) -- we don't know the exact color
        ramp gdaldem used, but we DO have the real elevation at every pixel
        of the same raster, so we can recover an honest approximate gradient
        by sampling actual colors per elevation band instead of guessing."""
        dem_ds = dem[::tstep, ::tstep]
        rgb_ds = rgb_full[::tstep, ::tstep, :]
        edges = np.linspace(elev_min, elev_max, n_bins + 1)
        bin_idx = np.clip(np.digitize(dem_ds, edges) - 1, 0, n_bins - 1)
        stops = []
        last = np.array([40, 40, 44])
        for i in range(n_bins):
            sel = rgb_ds[bin_idx == i]
            if sel.size:
                last = np.median(sel.reshape(-1, 3), axis=0)
            stops.append(last)
        return {
            "kind": "gradient",
            "description": description,
            "bar": _bar_from_samples(stops),
            "minLabel": f"{_fmt_num(elev_min)} m",
            "maxLabel": f"{_fmt_num(elev_max)} m",
            "approx": True,
        }

    layers = {}
    legends = {}

    layers["hillshade-color"] = to_rgb_uint8(hillshade_color)
    legends["hillshade-color"] = sampled_gradient_legend(
        hillshade_color,
        "Elevation-tinted color ramp with simulated sun shading (slopes facing "
        "the light source appear brighter). Gradient below is an approximate "
        "reconstruction sampled from the actual image, since the original "
        "gdaldem color ramp definition isn't stored in the file."
    )

    layers["color-relief"] = to_rgb_uint8(color_relief)
    legends["color-relief"] = sampled_gradient_legend(
        color_relief,
        "Elevation-only color ramp (no shading) -- color depends purely on "
        "height, not slope or sun angle. Gradient below is an approximate "
        "reconstruction sampled from the actual image."
    )

    layers["hillshade"] = grayscale_to_rgb(hillshade)
    legends["hillshade"] = {
        "kind": "gradient",
        "description": "Simulated illumination from a virtual sun, independent "
                        "of elevation -- dark = slope faces away from the light "
                        "source, bright = slope faces toward it.",
        "bar": _bar_from_cmap("gray"),
        "minLabel": "shadowed",
        "maxLabel": "sunlit",
    }

    layers["slope"], legends["slope"] = colorize(
        slope, "inferno", vmin=0, vmax=60, unit="\u00b0",
        description="Steepness of the terrain at each point, in degrees from horizontal."
    )
    layers["aspect"], legends["aspect"] = colorize_aspect(
        aspect,
        description="Compass direction the slope faces (which way a ball would "
                     "roll downhill), using gdaldem's documented convention: "
                     "0\u00b0 = north, measured clockwise."
    )
    layers["roughness"], legends["roughness"] = colorize(
        roughness, "viridis", vmin=0,
        description="Local elevation variability -- how much height changes "
                     "between a cell and its neighbors. Higher = more jagged/broken terrain."
    )

    if "flow_accum" in optional_arrays:
        layers["flow-accumulation"], legends["flow-accumulation"] = colorize_log(
            optional_arrays["flow_accum"], "magma",
            description="Number of upstream cells draining through each cell (D8 "
                         "routing) -- high values trace the river network. Color "
                         "scale is logarithmic since a few channels carry vastly "
                         "more flow than the surrounding hillslopes."
        )
    if "sca" in optional_arrays:
        layers["specific-catchment-area"], legends["specific-catchment-area"] = colorize_log(
            optional_arrays["sca"], "magma",
            description="Upstream contributing area per unit contour width (D\u221e "
                         "routing) -- similar purpose to flow accumulation, but "
                         "distributes flow across multiple downslope neighbors "
                         "instead of a single one. Also shown on a log scale."
        )
    if "twi" in optional_arrays:
        layers["twi"], legends["twi"] = colorize(
            optional_arrays["twi"], "YlGnBu",
            description="Topographic Wetness Index -- estimates where water tends "
                         "to accumulate and soils stay wetter, combining upslope "
                         "contributing area with local slope. Higher (blue) = wetter, "
                         "flatter, more water-accumulating; lower (pale) = drier ridges."
        )
    if "flow_dir_dinf" in optional_arrays:
        layers["flow-direction-dinf"], legends["flow-direction-dinf"] = colorize_cyclic_direction(
            optional_arrays["flow_dir_dinf"],
            description="D\u221e direction of steepest descent, as an angle in degrees. "
                         "Shown as raw degrees rather than compass letters because "
                         "the axis convention this file uses (commonly measured "
                         "counterclockwise from east in TauDEM's native output) "
                         "isn't independently confirmed here."
        )
    if "flow_dir_d8" in optional_arrays:
        layers["flow-direction-d8"], legends["flow-direction-d8"] = colorize_d8_categorical(
            optional_arrays["flow_dir_d8"],
            description="D8 direction of steepest descent -- each cell drains "
                         "into exactly one of its 8 neighbors. Colors are arbitrary "
                         "labels, not a gradient; look up the compass direction "
                         "in the key below."
        )
    if "hydro_dem" in optional_arrays:
        layers["hydro-conditioned-dem"], legends["hydro-conditioned-dem"] = colorize(
            optional_arrays["hydro_dem"], "terrain",
            vmin=elev_min, vmax=elev_max, unit=" m",
            description="Elevation after hydrologic conditioning (pits/depressions "
                        "filled or breached so water can always route downhill) -- "
                        "compare against the main elevation color layer to spot "
                        "where the DEM was altered."
        )

    # ---- external data: CHIRPS precipitation + SoilGrids-derived TRIGRS params ----
    if _EXTERNAL_DATA_MODULES_AVAILABLE and (CHIRPS_FILE or SOILGRIDS_FILE):
        # Resample onto a grid at TEXTURE resolution (cheap: ~1200 cols vs
        # ~2100 full-res), then tile each pixel back up by `tstep` so the
        # result is (H, W)-shaped. colorize() below does its own internal
        # [::tstep, ::tstep] stride (same convention as every other layer
        # in this script) -- tiling first means that stride exactly recovers
        # the values actually computed here, instead of either being
        # downsampled a second time (wrong, smaller, misaligned) or
        # resampled at wasteful full DEM resolution just to be strided away.
        th = len(range(0, H, tstep))
        tw = len(range(0, W, tstep))
        rows = np.arange(th) * tstep
        cols = np.arange(tw) * tstep
        lat_row = lat_top - rows * px_h
        lon_col = lon_left + cols * px_w
        lon_grid, lat_grid = np.meshgrid(lon_col, lat_row)

        def tile_to_full_res(arr_tex):
            tiled = np.repeat(np.repeat(arr_tex, tstep, axis=0), tstep, axis=1)
            out = np.empty((H, W), dtype=arr_tex.dtype)
            out[:min(H, tiled.shape[0]), :min(W, tiled.shape[1])] = tiled[:H, :W]
            if tiled.shape[0] < H or tiled.shape[1] < W:  # pad any ragged edge
                out[tiled.shape[0]:, :] = out[tiled.shape[0] - 1, :]
                out[:, tiled.shape[1]:] = out[:, tiled.shape[1] - 1][:, None]
            return out

        if CHIRPS_FILE:
            try:
                chirps_path = _resolve_path(CHIRPS_FILE)
                precip = resample_chirps_to_points(chirps_path, lon_grid, lat_grid)
                layers["precipitation"], legends["precipitation"] = colorize(
                    tile_to_full_res(precip), "GnBu", vmin=float(precip.min()), vmax=float(precip.max()), unit=" mm",
                    description="CHIRPS satellite-derived rainfall estimate for this date, "
                                "resampled from its native ~5.5km grid onto the DEM -- note the "
                                "underlying data is far coarser than the DEM itself, so fine "
                                "spatial detail in this layer is an artifact of resampling, not "
                                "real local variation."
                )
                print(f"[ok] precipitation layer: {precip.min():.1f}-{precip.max():.1f} mm")
            except Exception as e:
                print(f"[skip] CHIRPS layer failed: {e}")

        if SOILGRIDS_FILE:
            try:
                soilgrids_path = _resolve_path(SOILGRIDS_FILE)
                soil = resample_soilgrids_to_points(soilgrids_path, lon_grid, lat_grid)
                sand_pct, silt_pct, clay_pct, bd_gcm3 = soilgrids_to_fractions(
                    soil["sand_gkg"], soil["clay_gkg"], soil["bdod_cgcm3"])
                texture = usda_texture_class(sand_pct, silt_pct, clay_pct)
                cohesion, friction = texture_to_strength(texture)
                has_soil = soil["has_soil_data"]

                SOIL_CAVEAT = (
                    " APPROXIMATE: derived from SoilGrids texture/bulk-density via generic "
                    "pedotransfer functions, not measured or calibrated to this basin -- "
                    "order-of-magnitude planning estimate only. Dark gray = outside the "
                    "mapped watershed polygon (no source data)."
                )

                friction_masked = np.where(has_soil, friction, -9999.0)
                layers["soil-friction-angle"], legends["soil-friction-angle"] = colorize(
                    tile_to_full_res(friction_masked), "cividis", vmin=float(friction[has_soil].min()),
                    vmax=float(friction[has_soil].max()), unit="\u00b0",
                    description="Estimated internal friction angle, from SoilGrids texture class."
                    + SOIL_CAVEAT
                )

                cohesion_masked = np.where(has_soil, cohesion, -9999.0)
                layers["soil-cohesion"], legends["soil-cohesion"] = colorize(
                    tile_to_full_res(cohesion_masked), "cividis", vmin=float(cohesion[has_soil].min()),
                    vmax=float(cohesion[has_soil].max()), unit=" kPa",
                    description="Estimated soil cohesion, from SoilGrids texture class."
                    + SOIL_CAVEAT
                )
                print(f"[ok] soil layers: {has_soil.mean()*100:.0f}% of view inside mapped watershed")
            except Exception as e:
                print(f"[skip] SoilGrids layers failed: {e}")

    textures_b64 = {}
    for name, im in layers.items():
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=87)
        textures_b64[name] = base64.b64encode(buf.getvalue()).decode("ascii")

    render_html(meta, heights_b64, textures_b64, OUT_PATH, layer_order=list(layers.keys()), legends=legends)
    print(f"wrote {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
    print(f"grid {gw}x{gh}  extent {width_m/1000:.1f} x {height_m/1000:.1f} km  "
          f"elev {elev_min:.0f}-{elev_max:.0f} m")
    print(f"layers: {', '.join(layers.keys())}")


def render_html(meta, heights_b64, textures_b64, out_path, layer_order, legends):
    layer_labels = {
        "hillshade-color": "Shaded relief + color",
        "color-relief": "Elevation color",
        "hillshade": "Hillshade (grayscale)",
        "slope": "Slope",
        "aspect": "Aspect",
        "roughness": "Roughness",
        "flow-accumulation": "Flow accumulation",
        "specific-catchment-area": "Specific catchment area",
        "twi": "Topographic wetness (TWI)",
        "flow-direction-dinf": "Flow direction (D\u221e)",
        "flow-direction-d8": "Flow direction (D8)",
        "hydro-conditioned-dem": "Hydro-conditioned DEM",
        "precipitation": "Precipitation (CHIRPS)",
        "soil-friction-angle": "Soil friction angle (est.)",
        "soil-cohesion": "Soil cohesion (est.)",
    }
    buttons_html = "\n".join(
        f'<button class="layer-btn{" active" if l == layer_order[0] else ""}" '
        f'data-layer="{l}">{layer_labels[l]}</button>'
        for l in layer_order
    )
    textures_js = json.dumps({k: f"data:image/jpeg;base64,{v}" for k, v in textures_b64.items()})
    meta_js = json.dumps(meta)
    legends_js = json.dumps({k: {**v, "title": layer_labels.get(k, k)} for k, v in legends.items()})
    default_exagg = meta.get("defaultExagg", 1.6)

    template = TEMPLATE
    template = template.replace("__BUTTONS__", buttons_html)
    template = template.replace("__META__", meta_js)
    template = template.replace("__HEIGHTS__", heights_b64)
    template = template.replace("__TEXTURES__", textures_js)
    template = template.replace("__LEGENDS__", legends_js)
    template = template.replace("__DEFAULT_EXAGG__", f"{default_exagg}")

    with open(out_path, "w") as f:
        f.write(template)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DEM 3D Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --bg: #14161a; --panel: #1c1f24; --panel-border: #2a2d33;
    --text: #e8e5dd; --text-dim: #9a9da4; --accent: #c8873f;
    --mono: 'SF Mono', 'Roboto Mono', ui-monospace, Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin:0; padding:0; width:100%; height:100%; background:var(--bg); overflow:hidden; font-family:var(--sans); }
  #canvas-wrap { position:absolute; inset:0; }
  canvas { display:block; }
  #panel {
    position:absolute; top:16px; left:16px; width:250px;
    background:rgba(28,31,36,0.88); backdrop-filter: blur(8px);
    border:1px solid var(--panel-border); border-radius:10px;
    padding:16px; color:var(--text); font-size:13px;
    max-height: calc(100vh - 32px); overflow-y:auto;
    transition: width .15s ease, padding .15s ease;
  }
  #panel.collapsed {
    width:auto; max-height:none; padding:8px; overflow:visible;
  }
  #panel.collapsed #panel-body,
  #panel.collapsed #panel-title { display:none; }
  #panel-header { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  #panel-toggle {
    flex-shrink:0; width:22px; height:22px; padding:0;
    background:#22262c; border:1px solid var(--panel-border); border-radius:6px;
    color:var(--text); font-size:13px; line-height:1; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    font-family:var(--sans);
  }
  #panel-toggle:hover { background:#282c33; border-color:var(--accent); }
  #panel h1 { font-size:14px; font-weight:600; margin:0 0 2px 0; letter-spacing:0.2px; }
  #panel .sub { font-size:11px; color:#93969c; margin:0 0 14px 0; line-height:1.5; }

  #legend-panel {
    position:absolute; top:16px; right:16px; width:250px;
    background:rgba(28,31,36,0.88); backdrop-filter: blur(8px);
    border:1px solid var(--panel-border); border-radius:10px;
    padding:16px; color:var(--text); font-size:13px;
    max-height: calc(100vh - 32px); overflow-y:auto;
    transition: width .15s ease, padding .15s ease;
  }
  #legend-panel.collapsed { width:auto; max-height:none; padding:8px; overflow:visible; }
  #legend-panel.collapsed #legend-body,
  #legend-panel.collapsed #legend-title { display:none; }
  #legend-header { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  #legend-toggle {
    flex-shrink:0; width:22px; height:22px; padding:0;
    background:#22262c; border:1px solid var(--panel-border); border-radius:6px;
    color:var(--text); font-size:13px; line-height:1; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    font-family:var(--sans);
  }
  #legend-toggle:hover { background:#282c33; border-color:var(--accent); }
  #legend-panel h1 { font-size:14px; font-weight:600; margin:0 0 10px 0; letter-spacing:0.2px; }
  #legend-layer-name { font-size:12.5px; font-weight:600; color:#f0d9b8; margin-bottom:6px; }
  #legend-desc { font-size:12px; color:#c3c6cb; line-height:1.55; margin:0 0 14px 0; }
  #legend-bar-wrap { margin-bottom:6px; }
  #legend-bar-img { width:100%; height:16px; border-radius:4px; display:block; image-rendering:auto; }
  #legend-labels { display:flex; justify-content:space-between; font-family:var(--mono); font-size:10.5px; color:var(--text-dim); margin-top:4px; }
  #legend-labels .mid { position:absolute; left:50%; transform:translateX(-50%); }
  #legend-tick-row { position:relative; height:14px; margin-top:4px; }
  #legend-tick-row span { position:absolute; transform:translateX(-50%); font-family:var(--mono); font-size:10.5px; color:var(--text-dim); }
  .legend-approx-note { font-size:10.5px; color:#77797e; margin-top:8px; font-style:italic; line-height:1.4; }
  .legend-swatch-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px; }
  .legend-swatch { width:14px; height:14px; border-radius:3px; flex-shrink:0; border:1px solid rgba(255,255,255,0.15); }
  .section-label { font-size:11px; color:#93969c; margin:14px 0 6px 0; }
  .section-label:first-of-type { margin-top:0; }
  .layer-btn {
    display:block; width:100%; text-align:left; padding:7px 10px; margin-bottom:5px;
    background:#22262c; border:1px solid transparent; border-radius:6px;
    color:var(--text); font-size:12.5px; cursor:pointer; font-family:var(--sans);
    transition:background .12s, border-color .12s;
  }
  .layer-btn:hover { background:#282c33; }
  .layer-btn.active { border-color:var(--accent); background:#2c2416; color:#f0d9b8; }
  .row { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
  .row label { font-size:12px; color:var(--text-dim); }
  .row input[type=range] { width:120px; accent-color:var(--accent); }
  .row .val { font-family:var(--mono); font-size:11px; color:var(--accent); width:34px; text-align:right; }
  .toggle-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; font-size:12px; color:var(--text-dim); }
  .toggle-row input { accent-color:var(--accent); }
  #stats { margin-top:14px; padding-top:12px; border-top:1px solid var(--panel-border); font-family:var(--mono); font-size:11px; color:#8d9096; line-height:1.9; }
  #stats b { color:var(--text); font-weight:500; }
  #hint { position:absolute; bottom:14px; left:16px; font-size:11px; color:#6d7076; font-family:var(--sans); pointer-events:none; }
  #loading { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#8d9096; font-size:13px; letter-spacing:0.3px; background:var(--bg); z-index:10; }
</style>
</head>
<body>
<div id="canvas-wrap"></div>
<div id="loading">Building terrain mesh&hellip;</div>

<div id="panel">
  <div id="panel-header">
    <h1 id="panel-title">DEM 3D Viewer</h1>
    <button id="panel-toggle" title="Collapse panel">&#8249;</button>
  </div>
  <div id="panel-body">
  <p class="sub">Copernicus GLO-30 derived terrain</p>
  <div class="section-label">Surface layer</div>
  __BUTTONS__
  <div class="section-label">Display</div>
  <div class="row">
    <label>Vertical exaggeration</label>
    <input type="range" id="exagg" min="0.5" max="4" step="0.1" value="__DEFAULT_EXAGG__">
    <span class="val" id="exagg-val">__DEFAULT_EXAGG__x</span>
  </div>
  <div class="toggle-row"><input type="checkbox" id="wireframe"> Wireframe overlay</div>
  <div class="toggle-row"><input type="checkbox" id="autorotate"> Auto-rotate</div>
  <div id="stats">
    <div>elev &nbsp;<b id="stat-elev"></b></div>
    <div>extent <b id="stat-extent"></b></div>
    <div>grid &nbsp;<b id="stat-grid"></b></div>
  </div>
  </div>
</div>

<div id="legend-panel">
  <div id="legend-header">
    <h1 id="legend-title">Legend</h1>
    <button id="legend-toggle" title="Collapse legend">&#8250;</button>
  </div>
  <div id="legend-body">
    <div id="legend-layer-name"></div>
    <p id="legend-desc"></p>
    <div id="legend-visual"></div>
  </div>
</div>

<div id="hint">drag to orbit &middot; scroll to zoom &middot; right-drag to pan</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const META = __META__;
const HEIGHTS_B64 = "__HEIGHTS__";
const TEXTURES = __TEXTURES__;
const LEGENDS = __LEGENDS__;

function b64ToFloat32(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const buf = new ArrayBuffer(len);
  const view = new Uint8Array(buf);
  for (let i = 0; i < len; i++) view[i] = binary.charCodeAt(i);
  return new Float32Array(buf);
}

const heights = b64ToFloat32(HEIGHTS_B64);
const gw = META.gridW, gh = META.gridH;
const elevMin = META.elevMin, elevMax = META.elevMax;
const widthM = META.widthM, heightM = META.heightM;

const wrap = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
scene.fog = new THREE.FogExp2(0x14161a, 0.0000085);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth/window.innerHeight, 10, 400000);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xfff2d9, 1.15);
sun.position.set(-widthM*0.35, elevMax*2.2, heightM*0.25);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x9db4d9, 0.35);
fill.position.set(widthM*0.4, elevMax*1.2, -heightM*0.4);
scene.add(fill);

const geometry = new THREE.PlaneGeometry(widthM, heightM, gw - 1, gh - 1);
geometry.rotateX(-Math.PI/2);
const posAttr = geometry.attributes.position;
let currentExagg = META.defaultExagg || 1.6;

function applyHeights(exagg) {
  for (let i = 0; i < heights.length; i++) {
    posAttr.setY(i, (heights[i] - elevMin) * exagg);
  }
  posAttr.needsUpdate = true;
  geometry.computeVertexNormals();
}
applyHeights(currentExagg);

const loader = new THREE.TextureLoader();
const texCache = {};
function getTexture(name) {
  if (!texCache[name]) {
    const t = loader.load(TEXTURES[name]);
    t.colorSpace = THREE.SRGBColorSpace || t.colorSpace;
    t.wrapS = THREE.ClampToEdgeWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    texCache[name] = t;
  }
  return texCache[name];
}

const material = new THREE.MeshStandardMaterial({
  map: getTexture('hillshade-color'), roughness: 0.92, metalness: 0.02, side: THREE.DoubleSide,
});
const terrain = new THREE.Mesh(geometry, material);
scene.add(terrain);

const wireMat = new THREE.MeshBasicMaterial({ color: 0xc8873f, wireframe: true, transparent:true, opacity:0.18 });
const wireMesh = new THREE.Mesh(geometry, wireMat);
wireMesh.visible = false;
scene.add(wireMesh);

const target = new THREE.Vector3(0, (elevMax-elevMin)*currentExagg*0.18, 0);
let radius = Math.max(widthM, heightM) * 0.95;
let theta = Math.PI * 0.62;
let phi = Math.PI * 0.32;
const minPhi = 0.08, maxPhi = 1.45;

function updateCamera() {
  const sp = Math.sin(phi);
  camera.position.set(
    target.x + radius * sp * Math.sin(theta),
    target.y + radius * Math.cos(phi),
    target.z + radius * sp * Math.cos(theta)
  );
  camera.lookAt(target);
}
updateCamera();

let dragging = false, panning = false, lastX = 0, lastY = 0;
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
renderer.domElement.addEventListener('mousedown', e => {
  if (e.button === 2) panning = true; else dragging = true;
  lastX = e.clientX; lastY = e.clientY;
});
window.addEventListener('mouseup', () => { dragging = false; panning = false; });
window.addEventListener('mousemove', e => {
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (dragging) {
    theta -= dx * 0.0055; phi -= dy * 0.0045;
    phi = Math.max(minPhi, Math.min(maxPhi, phi));
    updateCamera();
  } else if (panning) {
    const panScale = radius * 0.0009;
    const right = new THREE.Vector3(Math.cos(theta), 0, -Math.sin(theta));
    const fwd = new THREE.Vector3(Math.sin(theta), 0, Math.cos(theta));
    target.addScaledVector(right, -dx * panScale);
    target.addScaledVector(fwd, dy * panScale);
    updateCamera();
  }
});
renderer.domElement.addEventListener('wheel', e => {
  e.preventDefault();
  radius *= Math.pow(1.001, e.deltaY);
  radius = Math.max(1200, Math.min(220000, radius));
  updateCamera();
}, { passive: false });

let touchLastDist = null;
renderer.domElement.addEventListener('touchstart', e => {
  if (e.touches.length === 1) { dragging = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY; }
  else if (e.touches.length === 2) { touchLastDist = null; }
}, { passive: true });
renderer.domElement.addEventListener('touchmove', e => {
  if (e.touches.length === 1 && dragging) {
    const dx = e.touches[0].clientX - lastX, dy = e.touches[0].clientY - lastY;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    theta -= dx * 0.0055; phi -= dy * 0.0045;
    phi = Math.max(minPhi, Math.min(maxPhi, phi));
    updateCamera();
  } else if (e.touches.length === 2) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (touchLastDist !== null) {
      radius *= touchLastDist / dist;
      radius = Math.max(1200, Math.min(220000, radius));
      updateCamera();
    }
    touchLastDist = dist;
  }
}, { passive: true });
window.addEventListener('touchend', () => { dragging = false; });

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

document.querySelectorAll('.layer-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    material.map = getTexture(btn.dataset.layer);
    material.needsUpdate = true;
    renderLegend(btn.dataset.layer);
  });
});

function renderLegend(name) {
  const legend = LEGENDS[name];
  const nameEl = document.getElementById('legend-layer-name');
  const descEl = document.getElementById('legend-desc');
  const visualEl = document.getElementById('legend-visual');
  visualEl.innerHTML = '';
  if (!legend) {
    nameEl.textContent = name || '';
    descEl.textContent = '';
    return;
  }
  nameEl.textContent = legend.title || name;
  descEl.textContent = legend.description || '';

  if (legend.kind === 'gradient' || legend.kind === 'cyclic') {
    const wrap = document.createElement('div');
    wrap.id = 'legend-bar-wrap';
    const img = document.createElement('img');
    img.id = 'legend-bar-img';
    img.src = legend.bar;
    wrap.appendChild(img);
    visualEl.appendChild(wrap);

    if (legend.ticks && legend.ticks.length) {
      const tickRow = document.createElement('div');
      tickRow.id = 'legend-tick-row';
      legend.ticks.forEach(t => {
        const span = document.createElement('span');
        span.style.left = (t.pos * 100) + '%';
        span.textContent = t.label;
        tickRow.appendChild(span);
      });
      visualEl.appendChild(tickRow);
    } else {
      const labels = document.createElement('div');
      labels.id = 'legend-labels';
      const lo = document.createElement('span');
      lo.textContent = legend.minLabel || '';
      const hi = document.createElement('span');
      hi.textContent = legend.maxLabel || '';
      labels.appendChild(lo);
      labels.appendChild(hi);
      visualEl.appendChild(labels);
      if (legend.midLabel) {
        const mid = document.createElement('div');
        mid.style.textAlign = 'center';
        mid.style.fontFamily = 'var(--mono)';
        mid.style.fontSize = '10.5px';
        mid.style.color = 'var(--text-dim)';
        mid.style.marginTop = '2px';
        mid.textContent = legend.midLabel + ' (mid)';
        visualEl.appendChild(mid);
      }
    }
    if (legend.approx) {
      const note = document.createElement('div');
      note.className = 'legend-approx-note';
      note.textContent = 'Approximate -- reconstructed from the image itself, not the original color-ramp definition.';
      visualEl.appendChild(note);
    }
    if (legend.logScale) {
      const note = document.createElement('div');
      note.className = 'legend-approx-note';
      note.textContent = 'Logarithmic scale -- most of the color range is spent on the low end so the drainage network stays visible.';
      visualEl.appendChild(note);
    }
  } else if (legend.kind === 'categorical' && legend.items) {
    legend.items.forEach(item => {
      const row = document.createElement('div');
      row.className = 'legend-swatch-row';
      const sw = document.createElement('span');
      sw.className = 'legend-swatch';
      sw.style.background = item.color;
      const lbl = document.createElement('span');
      lbl.textContent = item.label;
      row.appendChild(sw);
      row.appendChild(lbl);
      visualEl.appendChild(row);
    });
  }
}

const exaggSlider = document.getElementById('exagg');
const exaggVal = document.getElementById('exagg-val');
exaggSlider.addEventListener('input', () => {
  currentExagg = parseFloat(exaggSlider.value);
  exaggVal.textContent = currentExagg.toFixed(1) + 'x';
  applyHeights(currentExagg);
  target.y = (elevMax-elevMin)*currentExagg*0.18;
  updateCamera();
});
document.getElementById('wireframe').addEventListener('change', e => { wireMesh.visible = e.target.checked; });
let autorotate = false;
document.getElementById('autorotate').addEventListener('change', e => { autorotate = e.target.checked; });

const panelEl = document.getElementById('panel');
const panelToggle = document.getElementById('panel-toggle');
panelToggle.addEventListener('click', () => {
  const collapsed = panelEl.classList.toggle('collapsed');
  panelToggle.innerHTML = collapsed ? '&#8250;' : '&#8249;';
  panelToggle.title = collapsed ? 'Expand panel' : 'Collapse panel';
});

const legendPanelEl = document.getElementById('legend-panel');
const legendToggle = document.getElementById('legend-toggle');
legendToggle.addEventListener('click', () => {
  const collapsed = legendPanelEl.classList.toggle('collapsed');
  legendToggle.innerHTML = collapsed ? '&#8249;' : '&#8250;';
  legendToggle.title = collapsed ? 'Expand legend' : 'Collapse legend';
});

const initialActiveBtn = document.querySelector('.layer-btn.active');
renderLegend(initialActiveBtn ? initialActiveBtn.dataset.layer : null);

document.getElementById('stat-elev').textContent = Math.round(elevMin) + ' \u2013 ' + Math.round(elevMax) + ' m';
document.getElementById('stat-extent').textContent = (widthM/1000).toFixed(1) + ' \u00d7 ' + (heightM/1000).toFixed(1) + ' km';
document.getElementById('stat-grid').textContent = gw + ' \u00d7 ' + gh + ' verts';
document.getElementById('loading').style.display = 'none';

function animate() {
  requestAnimationFrame(animate);
  if (autorotate) { theta += 0.0016; updateCamera(); }
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
