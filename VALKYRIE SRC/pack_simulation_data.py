"""Packs the 24-hour simulation into a compact quantized payload for the
animated viewer: per-hour factor-of-safety (risk) and flow depth, uint8
quantized, plus per-node static geometry/texture reused from the main
viewer pipeline."""
import base64
import numpy as np


def pack_simulation(npz_path, fs_max_display=3.0, depth_max_display=10.0, erosion_max_display=3.0):
    d = np.load(npz_path)
    fs = d["fs_series"]              # [24, n]
    depth = d["flow_depth_series"]   # [24, n]
    precip_mm = d["precip_mm"]
    storm_shape = d["storm_shape"]
    gh, gw = d["grid_shape"]

    fs_q = np.clip(fs / fs_max_display, 0, 1) * 255
    fs_q = fs_q.astype(np.uint8)
    depth_q = np.clip(depth / depth_max_display, 0, 1) * 255
    depth_q = depth_q.astype(np.uint8)

    packed = np.stack([fs_q, depth_q], axis=-1)  # [24, n, 2]
    b64 = base64.b64encode(packed.tobytes()).decode("ascii")

    # cum_erosion is a single per-node STATIC array (total accumulated slip
    # depth once a node has failed, summed over the whole 24h run) -- not a
    # per-hour series like fs/depth above, since route_failures() only
    # returns the final accumulated value. Packed separately as its own
    # byte array (length n_nodes, not n_hours*n_nodes). If you want this to
    # grow progressively on screen in sync with playback rather than only
    # being usable as a static "final scar extent" layer, route_failures()
    # in voellmy_runout.py would need to snapshot cum_erosion.copy() into a
    # [n_hours, n] array each hour, the same way flow_depth_series already
    # does -- it currently only keeps the running total, not the history.
    cum_erosion = d["cum_erosion"] if "cum_erosion" in d.files else np.zeros(fs.shape[1], dtype=np.float32)
    erosion_q = np.clip(cum_erosion / erosion_max_display, 0, 1) * 255
    erosion_q = erosion_q.astype(np.uint8)
    erosion_b64 = base64.b64encode(erosion_q.tobytes()).decode("ascii")

    failed_frac_per_hour = (fs < 1.0).mean(axis=1).tolist()
    hourly_rain_mm = (storm_shape * float(np.median(precip_mm))).tolist()

    return {
        "b64": b64,
        "erosion_b64": erosion_b64,
        "n_hours": int(fs.shape[0]),
        "n_nodes": int(fs.shape[1]),
        "grid_h": int(gh), "grid_w": int(gw),
        "fs_max_display": fs_max_display,
        "depth_max_display": depth_max_display,
        "erosion_max_display": erosion_max_display,
        "failed_frac_per_hour": failed_frac_per_hour,
        "hourly_rain_mm": hourly_rain_mm,
    }


if __name__ == "__main__":
    p = pack_simulation("/kaggle/working/valkyrie/physics/simulation_24h.npz")
    print("packed bytes (raw):", len(p["b64"]) * 3 // 4)
    print("n_hours", p["n_hours"], "n_nodes", p["n_nodes"])
    print("failed_frac_per_hour:", [f"{x:.3f}" for x in p["failed_frac_per_hour"]])
