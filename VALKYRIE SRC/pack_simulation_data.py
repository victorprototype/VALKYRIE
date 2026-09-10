"""Packs the 24-hour simulation into a compact quantized payload for the
animated viewer: per-hour factor-of-safety (risk) and flow depth, uint8
quantized, plus per-node static geometry/texture reused from the main
viewer pipeline."""
import base64
import numpy as np


def pack_simulation(npz_path, fs_max_display=3.0, depth_max_display=10.0,
                     erosion_max_display=5.0, failure_depth_m=1.5):
    d = np.load(npz_path)
    fs = d["fs_series"]              # [24, n]
    depth = d["flow_depth_series"]   # [24, n]
    precip_mm = d["precip_mm"]
    storm_shape = d["storm_shape"]
    gh, gw = d["grid_shape"]

    # reconstruct per-hour cumulative erosion depth (permanent elevation
    # loss at failure sources) from fs_series -- matches voellmy_runout.py's
    # `already_failed` logic: once a node crosses FS<1 it keeps
    # failure_depth_m of material permanently removed from that point on
    ever_failed_by_hour = np.minimum.accumulate(fs, axis=0) < 1.0   # [24, n] bool
    erosion_series = ever_failed_by_hour.astype(np.float32) * failure_depth_m

    fs_q = np.clip(fs / fs_max_display, 0, 1) * 255
    fs_q = fs_q.astype(np.uint8)
    depth_q = np.clip(depth / depth_max_display, 0, 1) * 255
    depth_q = depth_q.astype(np.uint8)
    erosion_q = np.clip(erosion_series / erosion_max_display, 0, 1) * 255
    erosion_q = erosion_q.astype(np.uint8)

    packed = np.stack([fs_q, depth_q, erosion_q], axis=-1)  # [24, n, 3]
    b64 = base64.b64encode(packed.tobytes()).decode("ascii")

    failed_frac_per_hour = (fs < 1.0).mean(axis=1).tolist()
    hourly_rain_mm = (storm_shape * float(np.median(precip_mm))).tolist()

    return {
        "b64": b64,
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
