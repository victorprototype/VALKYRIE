"""
Simplified cellular downslope routing of triggered (FS<1) failures using
the Voellmy (1955) depth-averaged friction model -- the same rheology used
in RAMMS, AVAL-1D, and most operational debris-flow/avalanche runout tools:

    dv/dt = g*sin(theta) - sign(v) * [ mu*g*cos(theta) + g*v^2/xi ]

where mu is a dry-Coulomb friction coefficient and xi is a turbulent drag
coefficient (m/s^2).

CAVEATS:
  - mu is approximated as tan(friction_angle) -- a standard but rough proxy,
    not a measured dynamic friction coefficient (which differs from the
    static soil friction angle in reality, often substantially for
    remolded/saturated debris).
  - xi is a single literature-typical constant (not spatially varying, not
    calibrated to this basin). Typical published ranges for debris
    flows/avalanches span roughly 100-1000 m/s^2 depending on material and
    channel confinement (Voellmy 1955; Salm 1993; Rickenmann 2005-style
    debris-flow parameter reviews) -- this implementation uses a single
    mid-range value; a real analysis would calibrate this per-basin against
    observed runout distances.
  - Routing is single-direction (D8), which cannot represent flow spreading/
    braiding the way a full 2D shallow-water solver would.
  - This is a raster cellular-automaton approximation of Voellmy dynamics,
    not a validated 2D depth-averaged solver (like RAMMS or r.avaflow).
"""
import numpy as np

G = 9.81
XI_TURBULENT = 500.0  # m/s^2, literature-typical mid-range, UNCALIBRATED
SUBSTEPS_PER_HOUR = 120  # 30s internal timestep for numerical stability


def d8_downslope_offsets(elevation, grid_shape):
    """Steepest-descent D8 neighbor for every node, from the elevation grid
    itself (self-consistent with what's actually in this graph, rather than
    re-deriving from the separate D8_Flow_Direction.tif's own convention)."""
    gh, gw = grid_shape
    elev = elevation.reshape(gh, gw)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dists = [np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)]

    best_drop = np.zeros((gh, gw))
    best_di = np.zeros((gh, gw), dtype=np.int8)
    best_dj = np.zeros((gh, gw), dtype=np.int8)

    for (di, dj), dist in zip(offsets, dists):
        # shifted[i,j] := elev[i+di, j+dj] wherever that neighbor is in bounds
        out_i0, out_i1 = max(0, -di), gh - max(0, di)
        out_j0, out_j1 = max(0, -dj), gw - max(0, dj)
        src_i0, src_i1 = out_i0 + di, out_i1 + di
        src_j0, src_j1 = out_j0 + dj, out_j1 + dj

        shifted = np.full((gh, gw), np.nan)
        shifted[out_i0:out_i1, out_j0:out_j1] = elev[src_i0:src_i1, src_j0:src_j1]
        drop = (elev - shifted) / dist
        drop = np.nan_to_num(drop, nan=-np.inf)
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        best_di = np.where(better, di, best_di)
        best_dj = np.where(better, dj, best_dj)
    return best_di, best_dj, best_drop  # best_drop <=0 means local sink (no downslope neighbor)


def route_failures(fs_series, elevation, slope_rad, friction_deg, grid_shape, n_hours,
                    failure_depth_m=1.5, xi=XI_TURBULENT, max_depth_m=8.0):
    """max_depth_m: once a cell's mobile depth exceeds this, the excess is
    redistributed to its (up to 8) neighbors rather than allowed to pile up
    without limit. This is a crude stand-in for real lateral spreading --
    single-direction D8 routing has no mechanism for a flow to widen, so
    without this, mass converging at a channel confluence piles up
    vertically without physical bound (observed directly: an early version
    of this model produced ~190m "debris depths" at confluence points,
    which is obviously nonphysical -- real debris flows run a few meters
    deep at most). This is still a simplification, not real 2D spreading
    physics, but keeps output in a physically plausible range."""
    gh, gw = grid_shape
    n = gh * gw
    mu = np.tan(np.radians(friction_deg)).reshape(-1)
    slope_flat = slope_rad.reshape(-1)

    di, dj, drop = d8_downslope_offsets(elevation, grid_shape)
    di, dj = di.reshape(-1), dj.reshape(-1)
    has_downslope = (drop.reshape(-1) > 0)

    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    ti = np.clip(ii.reshape(-1) + di, 0, gh - 1)
    tj = np.clip(jj.reshape(-1) + dj, 0, gw - 1)
    target_idx = ti * gw + tj

    # all 8 neighbors, for spreading overflow -- precomputed flat index per
    # node per direction, clipped at grid edges (edge cells just spread
    # less effectively, which is fine for this purpose)
    all_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbor_idx = []
    for ndi, ndj in all_offsets:
        nti = np.clip(ii.reshape(-1) + ndi, 0, gh - 1)
        ntj = np.clip(jj.reshape(-1) + ndj, 0, gw - 1)
        neighbor_idx.append(nti * gw + ntj)

    h = np.zeros(n)
    v = np.zeros(n)
    already_failed = np.zeros(n, dtype=bool)
    cum_erosion = np.zeros(n)
    flow_depth_series = np.zeros((n_hours, n), dtype=np.float32)

    dt_sub = 3600.0 / SUBSTEPS_PER_HOUR

    for hr in range(n_hours):
        newly_failed = (fs_series[hr] < 1.0) & (~already_failed)
        h[newly_failed] += failure_depth_m
        cum_erosion[newly_failed] += failure_depth_m
        already_failed |= (fs_series[hr] < 1.0)

        for _ in range(SUBSTEPS_PER_HOUR):
            moving = h > 1e-4
            if moving.any():
                accel = np.zeros(n)
                drive = G * np.sin(slope_flat)
                resist_static = mu * G * np.cos(slope_flat)
                resist_turb = G * v ** 2 / xi
                accel[moving] = (drive - np.sign(v + 1e-9) * (resist_static + resist_turb))[moving]
                v[moving] = np.maximum(0.0, v[moving] + accel[moving] * dt_sub)
                v[~moving] = 0.0

                dist = v * dt_sub
                cell_size = 140.0
                frac_move = np.clip(dist / cell_size, 0, 1)
                transferred = h * frac_move * moving * has_downslope
                h = h - transferred
                np.add.at(h, target_idx, transferred)

            overflow_mask = h > max_depth_m
            if overflow_mask.any():
                excess = np.where(overflow_mask, h - max_depth_m, 0.0)
                h = h - excess
                share = excess / 8.0
                for nidx in neighbor_idx:
                    np.add.at(h, nidx, share)

        flow_depth_series[hr] = h.astype(np.float32)

    return flow_depth_series, cum_erosion


if __name__ == "__main__":
    gh, gw = 20, 20
    x, y = np.meshgrid(np.arange(gw), np.arange(gh))
    elevation = (gh - y) * 10.0  # simple ramp, high at top, flowing down
    slope_rad = np.full((gh, gw), np.radians(30))
    friction_deg = np.full((gh, gw), 25.0)
    fs = np.ones((6, gh * gw)) * 1.5
    fs[2, gh // 2 * gw + gw // 2] = 0.5  # trigger one failure mid-grid at hour 2
    flow, erosion = route_failures(fs, elevation, slope_rad, friction_deg, (gh, gw), n_hours=6)
    print("flow depth max per hour:", flow.max(axis=1))
    print("total mass conserved (sum h) per hour:", flow.sum(axis=1))
    print("cum erosion total:", erosion.sum())
