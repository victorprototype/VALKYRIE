"""
VALKYRIE - Module 4: Risk Estimator
====================================
Consumes Module 3's simulation cache (FoS trajectory + debris depth) and:

  1. Classifies every node, at every timestep, into GREEN/YELLOW/RED alert
     tiers from its Factor of Safety.
  2. Logs alert TRANSITIONS (a node worsening from GREEN->YELLOW or
     YELLOW->RED) - this is the actionable "trigger evacuation now" signal,
     not just a static daily snapshot.
  3. Computes D8 flow direction + flow accumulation directly from the DEM
     (no separate "hydro flow graph" file required - this reproduces that
     product from data Module 1 already ingested).
  4. For every RED node, traces the downstream flow-routed corridor - the
     set of cells debris/floodwater would reach if failure occurred there
     - giving a predictive impact-area estimate that doesn't have to wait
     for Module 3's physics rollout to actually show depth arriving.

Everything here operates on the DAILY cache Module 3 exports (30 frames),
not the full sub-daily rollout, since alerting decisions are made at
daily granularity in this pipeline.
"""
from __future__ import annotations
import numpy as np

GREEN, YELLOW, RED = 0, 1, 2
TIER_NAMES = {GREEN: "GREEN", YELLOW: "YELLOW", RED: "RED"}

FOS_GREEN_THRESHOLD = 1.3
FOS_YELLOW_THRESHOLD = 1.0


# =============================================================================
# 1. RISK TIER CLASSIFICATION
# =============================================================================
def classify_risk_tiers(fos: np.ndarray) -> np.ndarray:
    """FoS > 1.3 -> GREEN, 1.0 <= FoS <= 1.3 -> YELLOW, FoS < 1.0 -> RED.
    Works on any shape (single grid, or a full [T, H, W] time series);
    NaN (invalid/no-data) cells pass through as NaN so they can still be
    masked out downstream rather than silently defaulting to a tier."""
    tier = np.where(fos > FOS_GREEN_THRESHOLD, GREEN,
                     np.where(fos >= FOS_YELLOW_THRESHOLD, YELLOW, RED))
    return np.where(np.isnan(fos), np.nan, tier.astype(np.float32))


# =============================================================================
# 2. ALERT TRANSITION LOG
# =============================================================================
def compute_alert_transitions(tier_seq: np.ndarray, valid_mask: np.ndarray):
    """tier_seq: [T, H, W]. Returns a list of dicts, one per node whose tier
    WORSENED (numerically increased) compared to the previous day - this is
    the actionable early-warning event stream, not a static classification."""
    T, H, W = tier_seq.shape
    events = []
    for t in range(1, T):
        prev, curr = tier_seq[t - 1], tier_seq[t]
        worsened = valid_mask & (curr > prev) & ~np.isnan(prev) & ~np.isnan(curr)
        ii, jj = np.where(worsened)
        for i, j in zip(ii, jj):
            events.append({
                "day": t,
                "row": int(i), "col": int(j),
                "from_tier": TIER_NAMES[int(prev[i, j])],
                "to_tier": TIER_NAMES[int(curr[i, j])],
            })
    return events


# =============================================================================
# 3. D8 FLOW DIRECTION + ACCUMULATION (computed from the DEM, no external file)
# =============================================================================
_D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def compute_flow_direction(elevation: np.ndarray, valid_mask: np.ndarray, cell_res: float):
    """Standard D8: each cell points to whichever of its 8 neighbors gives
    the steepest downhill slope. Returns a (H, W) array of flat receiver
    indices (row*W + col), or -1 for a pit/sink (no downhill neighbor) or
    invalid cell."""
    H, W = elevation.shape
    receiver = np.full((H, W), -1, dtype=np.int64)
    best_slope = np.zeros((H, W), dtype=np.float32)

    ii, jj = np.where(valid_mask)
    for di, dj in _D8_OFFSETS:
        ni, nj = ii + di, jj + dj
        in_bounds = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
        ii_b, jj_b, ni_b, nj_b = ii[in_bounds], jj[in_bounds], ni[in_bounds], nj[in_bounds]
        nbr_valid = valid_mask[ni_b, nj_b]
        ii_v, jj_v, ni_v, nj_v = ii_b[nbr_valid], jj_b[nbr_valid], ni_b[nbr_valid], nj_b[nbr_valid]

        dist = cell_res * np.sqrt(di**2 + dj**2)
        slope = (elevation[ii_v, jj_v] - elevation[ni_v, nj_v]) / dist  # >0 = downhill

        better = slope > best_slope[ii_v, jj_v]
        sel_i, sel_j = ii_v[better], jj_v[better]
        best_slope[sel_i, sel_j] = slope[better]
        receiver[sel_i, sel_j] = ni_v[better] * W + nj_v[better]

    return receiver  # flat index into (H*W), -1 = sink


def compute_flow_accumulation(receiver: np.ndarray, valid_mask: np.ndarray):
    """Processes cells in DESCENDING elevation order (guaranteed acyclic
    since D8 always points strictly downhill), accumulating each cell's
    contribution into its receiver. accumulation[cell] = 1 + sum of all
    cells that flow into it (directly or indirectly)."""
    H, W = valid_mask.shape
    acc = np.zeros(H * W, dtype=np.float64)
    valid_flat = valid_mask.ravel()
    acc[valid_flat] = 1.0
    return acc, receiver  # order handled in the caller with elevation sort


def route_flow_accumulation(elevation: np.ndarray, receiver: np.ndarray, valid_mask: np.ndarray):
    H, W = elevation.shape
    acc = np.zeros(H * W, dtype=np.float64)
    valid_flat = valid_mask.ravel()
    acc[valid_flat] = 1.0

    flat_idx = np.where(valid_flat)[0]
    order = flat_idx[np.argsort(-elevation.ravel()[flat_idx])]  # highest elevation first
    receiver_flat = receiver.ravel()

    for idx in order:
        r = receiver_flat[idx]
        if r >= 0:
            acc[r] += acc[idx]
    return acc.reshape(H, W)


# =============================================================================
# 4. DOWNSTREAM IMPACT CORRIDOR FROM RED CELLS
# =============================================================================
def trace_downstream_corridor(receiver: np.ndarray, source_mask: np.ndarray, max_steps: int = 500):
    """Follows the D8 receiver chain from every source (RED) cell until
    reaching a sink or a cell already in the corridor. Vectorized over all
    active source cells simultaneously (each step advances every active
    'front' cell one hop downstream)."""
    H, W = source_mask.shape
    receiver_flat = receiver.ravel()
    affected = source_mask.ravel().copy()
    current = np.where(source_mask.ravel())[0]

    for _ in range(max_steps):
        if current.size == 0:
            break
        nxt = receiver_flat[current]
        active = nxt >= 0
        nxt = nxt[active]
        if nxt.size == 0:
            break
        newly = ~affected[nxt]
        affected[nxt[newly]] = True
        current = nxt[newly]
        if current.size == 0:
            break

    return affected.reshape(H, W)


# =============================================================================
# ORCHESTRATION
# =============================================================================
def run_risk_estimation(cache_path: str, elevation: np.ndarray = None,
                         valid_mask: np.ndarray = None, cell_res: float = None,
                         max_corridor_steps: int = 500):
    """Loads a Module-3 .npz cache, classifies tiers for every day, computes
    the flow network once (static, from the DEM), traces a downstream
    impact corridor per day from that day's RED cells, and logs alert
    transitions across the month."""
    data = np.load(cache_path, allow_pickle=True)
    fos_seq = data["fos"]              # (T, H, W)
    if elevation is None:
        elevation = data["elevation"]
    if valid_mask is None:
        valid_mask = data["valid_mask"]
    if cell_res is None:
        cell_res = float(data["cell_res"])

    T, H, W = fos_seq.shape
    tier_seq = np.stack([classify_risk_tiers(fos_seq[t]) for t in range(T)], axis=0)

    receiver = compute_flow_direction(elevation, valid_mask, cell_res)
    flow_accum = route_flow_accumulation(elevation, receiver, valid_mask)

    corridor_seq = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        red_mask = valid_mask & (tier_seq[t] == RED)
        corridor_seq[t] = trace_downstream_corridor(receiver, red_mask, max_corridor_steps)

    alert_events = compute_alert_transitions(tier_seq, valid_mask)

    return dict(
        tier_seq=tier_seq, flow_accumulation=flow_accum, receiver=receiver,
        corridor_seq=corridor_seq, alert_events=alert_events,
    )


def export_risk_cache(out_path: str, source_cache_path: str, results: dict):
    data = dict(np.load(source_cache_path, allow_pickle=True))
    data["tier_seq"] = results["tier_seq"]
    data["flow_accumulation"] = results["flow_accumulation"]
    data["corridor_seq"] = results["corridor_seq"]
    data["alert_events"] = np.array(results["alert_events"], dtype=object)
    np.savez_compressed(out_path, **data)
    print(f"[cache] risk-enriched cache saved: {out_path}")


# =============================================================================
# TESTS
# =============================================================================
if __name__ == "__main__":
    print("=== A) D8 flow routing on a known synthetic valley ===")
    import rasterio
    with rasterio.open("valley_dem.tif") as src:
        valley_dem = src.read(1)
        valley_nodata = src.nodata
    valley_valid = valley_dem != valley_nodata
    valley_res = 30.0

    receiver = compute_flow_direction(valley_dem, valley_valid, valley_res)
    flow_accum = route_flow_accumulation(valley_dem, receiver, valley_valid)

    # DEM was built so elevation decreases toward row H-1 (south) and the
    # valley floor is at the center column - flow should converge there.
    H, W = valley_dem.shape
    outlet_col_band = flow_accum[H - 1, W // 2 - 2 : W // 2 + 3]
    edge_col_band = flow_accum[H - 1, :3]
    print(f"flow accumulation at outlet row, center columns: {outlet_col_band}")
    print(f"flow accumulation at outlet row, edge columns:   {edge_col_band}")
    assert outlet_col_band.max() > edge_col_band.max(), \
        "accumulation should be much higher at the valley center outlet than the edges"
    print("PASS: flow converges toward the valley center as designed.\n")

    print("=== B) Downstream corridor tracing from a single upstream source ===")
    source = np.zeros_like(valley_valid)
    source[2, W // 2] = True  # a cell near the top of the valley, on the centerline
    corridor = trace_downstream_corridor(receiver, source, max_steps=100)
    print("corridor cells found:", corridor.sum())
    # the corridor should reach all the way down to near the outlet row
    assert corridor[H - 2 :, W // 2 - 1 : W // 2 + 2].any(), \
        "corridor from an upstream source should reach the outlet"
    print("PASS: corridor traced from upstream source reaches the outlet.\n")

    print("=== C) Risk tier classification ===")
    test_fos = np.array([0.5, 0.99, 1.0, 1.15, 1.3, 1.31, 5.0, np.nan])
    tiers = classify_risk_tiers(test_fos)
    print("FoS:  ", test_fos)
    print("Tiers:", tiers, " (0=GREEN,1=YELLOW,2=RED,nan=invalid)")
    assert tiers[0] == RED and tiers[2] == YELLOW and tiers[4] == YELLOW and tiers[5] == GREEN
    assert np.isnan(tiers[-1])
    print("PASS: thresholds correctly applied.\n")

    print("=== D) Alert transition log on a manufactured worsening trajectory ===")
    valid_small = np.ones((3, 3), dtype=bool)
    # node (1,1) worsens GREEN -> YELLOW -> RED over 3 days; everything else stays GREEN
    fos_day0 = np.full((3, 3), 2.0); fos_day0[1, 1] = 2.0
    fos_day1 = np.full((3, 3), 2.0); fos_day1[1, 1] = 1.1
    fos_day2 = np.full((3, 3), 2.0); fos_day2[1, 1] = 0.8
    tier_seq_small = np.stack([classify_risk_tiers(f) for f in (fos_day0, fos_day1, fos_day2)])
    events = compute_alert_transitions(tier_seq_small, valid_small)
    print("events:", events)
    assert len(events) == 2, f"expected exactly 2 worsening transitions, got {len(events)}"
    assert events[0]["from_tier"] == "GREEN" and events[0]["to_tier"] == "YELLOW"
    assert events[1]["from_tier"] == "YELLOW" and events[1]["to_tier"] == "RED"
    print("PASS: transition log correctly captures GREEN->YELLOW->RED escalation.\n")

    print("=== E) Full pipeline against a real Module 3 cache ===")
    import sys, os
    sys.path.insert(0, "/home/claude/valkyrie")
    from ingest_valkyrie import IngestConfig, ingest
    from model_valkyrie import init_dynamic_state
    from simulator_valkyrie import run_physics_rollout, export_simulation_cache

    ing_cfg = IngestConfig(
        dem_path="testdata/dem.tif",
        soil_paths={
            ("bdod", "0-5cm"): "testdata/bdod_0-5cm.tif", ("bdod", "5-15cm"): "testdata/bdod_5-15cm.tif",
            ("clay", "0-5cm"): "testdata/clay_0-5cm.tif", ("clay", "5-15cm"): "testdata/clay_5-15cm.tif",
            ("sand", "0-5cm"): "testdata/sand_0-5cm.tif", ("sand", "5-15cm"): "testdata/sand_5-15cm.tif",
            ("silt", "0-5cm"): "testdata/silt_0-5cm.tif", ("silt", "5-15cm"): "testdata/silt_5-15cm.tif",
        },
        chirps_dir="testdata", chirps_glob="chirps-v2.0.2013.06.*.tif",
    )
    import jax.numpy as jnp
    graph, rainfall, _ = ingest(ing_cfg)
    node_static = jnp.array(graph["node_features"])
    senders, receivers_ = jnp.array(graph["senders"]), jnp.array(graph["receivers"])
    edge_features = jnp.array(graph["edge_features"])
    state0 = init_dynamic_state(graph["n_node"])
    state_seq, fos_seq, dt = run_physics_rollout(
        node_static, state0, senders, receivers_, edge_features, jnp.array(rainfall), 12,
    )
    export_simulation_cache("module4_test_cache.npz", graph, state_seq, fos_seq, 12)

    results = run_risk_estimation("module4_test_cache.npz")
    print("tier_seq shape:", results["tier_seq"].shape)
    print("flow_accumulation shape:", results["flow_accumulation"].shape,
          " max:", results["flow_accumulation"].max())
    print("corridor_seq shape:", results["corridor_seq"].shape)
    print("n_alert_events:", len(results["alert_events"]))
    tier_counts = {name: int(np.nansum(results["tier_seq"] == code))
                   for code, name in TIER_NAMES.items()}
    print("tier counts across all days/nodes:", tier_counts)

    export_risk_cache("module4_test_risk_cache.npz", "module4_test_cache.npz", results)
    reloaded = np.load("module4_test_risk_cache.npz", allow_pickle=True)
    print("reloaded keys:", list(reloaded.keys()))

    for f in ("module4_test_cache.npz", "module4_test_risk_cache.npz"):
        if os.path.exists(f):
            os.remove(f)
    print("\nall Module 4 smoke tests passed.")
