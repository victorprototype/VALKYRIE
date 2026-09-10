"""
Builds a unified mesh-graph from the DEM + derived hydrology rasters, in the
same node/edge formalism MeshGraphNets uses (Pfaff et al. 2021, Sec. 3):
  - nodes V at mesh-space coordinates u_i
  - mesh edges E^M with relative-displacement features u_ij = u_i - u_j, |u_ij|
  - node feature vector v_i holding the physical quantities at that node

This does NOT train anything (no torch here, no network in this sandbox).
It produces a static graph + feature tensors, saved as .npz, meant to be
loaded into PyTorch/PyG on Kaggle for the next stage.

CURRENTLY POPULATED node features (from data already available):
  elevation, slope, sin(aspect), cos(aspect), roughness, TWI,
  log1p(flow accumulation), log1p(specific catchment area)

NOT YET POPULATED (blocked on missing uploads):
  precipitation (CHIRPS)               -> node feature slot reserved
  cohesion, friction angle, unit weight,
  Ks, D0, theta_s, theta_r, alpha       -> node feature slots reserved,
                                            derived from SoilGrids via
                                            pedotransfer functions (TODO,
                                            see derive_trigrs_params() stub)
"""
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SRC = "/kaggle/input/datasets/thevictorprototype/dem-sim-data"

# Same downsample target as the 3D viewer, so this graph lines up 1:1 with
# what's rendered there if we ever want to cross-reference a node to a
# vertex on screen.
GRAPH_TARGET_W = 420
BORDER_CROP = 6


def load(name):
    return np.array(Image.open(f"{SRC}/{name}"))


def build_graph():
    dem = load("output_hh.tif").astype(np.float32)
    slope = load("viz.hh_slope.tif").astype(np.float32)
    aspect = load("viz.hh_aspect.tif").astype(np.float32)
    roughness = load("viz.hh_roughness.tif").astype(np.float32)
    twi = load("TWI.tif").astype(np.float32)
    flow_accum = load("D8_Flow_Accumulation.tif").astype(np.float32)
    sca = load("DInf_Specific_Catchment_Area.tif").astype(np.float32)

    H, W = dem.shape
    B = BORDER_CROP
    crop = lambda a: a[B:H - B, B:W - B]
    dem, slope, aspect, roughness, twi = map(crop, (dem, slope, aspect, roughness, twi))
    flow_accum, sca = map(crop, (flow_accum, sca))
    H, W = dem.shape

    step = max(1, round(W / GRAPH_TARGET_W))
    dem_g = dem[::step, ::step]
    slope_g = slope[::step, ::step]
    aspect_g = aspect[::step, ::step]
    rough_g = roughness[::step, ::step]
    twi_g = twi[::step, ::step]
    flowacc_g = flow_accum[::step, ::step]
    sca_g = sca[::step, ::step]
    gh, gw = dem_g.shape

    # ---- mesh-space node coordinates (meters), same conversion as viewer ----
    pixel_scale_deg = 0.0002777777777777778
    lat_top = 30.85000001111111 - B * pixel_scale_deg
    lon_left = 78.84999998888891 + B * pixel_scale_deg
    avg_lat = lat_top - (H * pixel_scale_deg) / 2
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(avg_lat))
    dx = pixel_scale_deg * step * m_per_deg_lon
    dy = pixel_scale_deg * step * m_per_deg_lat

    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    u_i = np.stack([xs * dx, ys * dy], axis=-1).astype(np.float32)  # mesh-space coords

    # ---- node features v_i ----
    aspect_rad = np.radians(np.where(aspect_g < -900, 0, aspect_g))  # -9999 nodata -> 0
    node_features = {
        "elevation_m": dem_g,
        "slope_deg": np.clip(slope_g, 0, None),
        "aspect_sin": np.sin(aspect_rad),
        "aspect_cos": np.cos(aspect_rad),
        "roughness": np.clip(rough_g, 0, None),
        "twi": twi_g,
        "log_flow_accum": np.log1p(np.clip(flowacc_g, 0, None)),
        "log_specific_catchment_area": np.log1p(np.clip(sca_g, 0, None)),
        # --- reserved, currently zero-filled placeholders ---
        "precip_mm": np.zeros((gh, gw), dtype=np.float32),        # CHIRPS -- awaiting upload
        "cohesion_kpa": np.zeros((gh, gw), dtype=np.float32),      # SoilGrids-derived -- awaiting upload
        "friction_angle_deg": np.zeros((gh, gw), dtype=np.float32),
        "unit_weight_kn_m3": np.zeros((gh, gw), dtype=np.float32),
        "ksat_m_s": np.zeros((gh, gw), dtype=np.float32),
        "diffusivity_m2_s": np.zeros((gh, gw), dtype=np.float32),
        "theta_sat": np.zeros((gh, gw), dtype=np.float32),
        "theta_res": np.zeros((gh, gw), dtype=np.float32),
        "alpha_1_m": np.zeros((gh, gw), dtype=np.float32),
    }
    feature_names = list(node_features.keys())
    x = np.stack([node_features[k].reshape(-1) for k in feature_names], axis=1)  # [N, F]

    # ---- mesh edges E^M: 4-connected grid, MeshGraphNets-style relative encoding ----
    node_id = np.arange(gh * gw).reshape(gh, gw)
    edges = []
    edge_uij = []  # relative mesh-space displacement u_ij = u_i - u_j
    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # bidirectional 4-connectivity
        ni = node_id[max(0, di):gh + min(0, di), max(0, dj):gw + min(0, dj)]
        nj = node_id[max(0, -di):gh + min(0, -di), max(0, -dj):gw + min(0, -dj)]
        src = ni.reshape(-1)
        dst = nj.reshape(-1)
        edges.append(np.stack([src, dst], axis=0))
        edge_uij.append((u_i.reshape(-1, 2)[src] - u_i.reshape(-1, 2)[dst]))
    edge_index = np.concatenate(edges, axis=1)  # [2, E]
    u_ij = np.concatenate(edge_uij, axis=0)      # [E, 2]
    norm_uij = np.linalg.norm(u_ij, axis=1, keepdims=True)
    edge_attr = np.concatenate([u_ij, norm_uij], axis=1).astype(np.float32)  # [E, 3] per MeshGraphNets encoder

    return {
        "pos": u_i.reshape(-1, 2),
        "x": x.astype(np.float32),
        "feature_names": feature_names,
        "edge_index": edge_index.astype(np.int64),
        "edge_attr": edge_attr,
        "grid_shape": (gh, gw),
        "elev_min": float(dem.min()),
        "elev_max": float(dem.max()),
    }


def derive_trigrs_params(soilgrids_arrays):
    """
    STUB -- not implemented yet, blocked on soilgrids_mandakini.tif.

    Plan once the file is available:
      1. Inspect band layout (SoilGrids 2.0 typically ships clay/sand/silt %,
         bulk density, organic carbon, CFVO, pH -- at multiple depths: 0-5cm,
         5-15cm, 15-30cm, etc. Need to confirm which bands this specific file has).
      2. theta_sat, theta_res, alpha, Ks via Rosetta pedotransfer functions
         (Schaap et al. 2001) driven by clay/sand/silt % + bulk density --
         this is the standard, defensible choice for exactly this derivation.
      3. cohesion, friction angle via texture-class lookup tables (e.g. the
         ranges tabulated in NAVFAC DM-7 or standard geotechnical references
         for the USDA texture class implied by clay/sand/silt %) -- flagged
         explicitly as coarse, literature-range estimates, not measurements,
         since no direct geotechnical proxy exists in SoilGrids.
      4. unit weight from bulk density directly (uws = bulk_density * g),
         the one parameter that maps cleanly with no PTF needed.
    """
    raise NotImplementedError("waiting on soilgrids_mandakini.tif")


if __name__ == "__main__":
    graph = build_graph()
    print("nodes:", graph["x"].shape[0], " features/node:", graph["x"].shape[1])
    print("edges:", graph["edge_index"].shape[1])
    print("feature order:", graph["feature_names"])
    print("grid shape:", graph["grid_shape"])
    np.savez_compressed(
        "/kaggle/working/valkyrie/graph/unified_graph.npz",
        pos=graph["pos"], x=graph["x"], edge_index=graph["edge_index"],
        edge_attr=graph["edge_attr"], grid_shape=graph["grid_shape"],
        feature_names=np.array(graph["feature_names"]),
    )
    print("saved unified_graph.npz")
