"""
VALKYRIE - Module 3: Simulator
===============================
The compute-intensive physics core. Two layers:

  A) physics_step / run_physics_rollout - a DETERMINISTIC, equation-driven
     debris-flow + erosion + slope-stability simulator on the Module-1
     graph. This alone produces the June-2013 retrodiction: no training
     required to get a result.

  B) train_gnn_emulator - trains the Module-2 PIGNN, via a physics-informed
     loss, to imitate physics_step's per-step behavior along the computed
     trajectory (teacher forcing). This gives a fast learned emulator for
     later interactive "what-if" scenarios (Module 5) without re-running
     the full numerical solver every time. It is NOT how the retrodiction
     itself is produced - that comes from (A).

Governing equations (all per-node, on the terrain graph):
  1. Mass continuity      - finite-volume flux divergence between neighbor
                             nodes, PLUS mass source terms from (a) bed
                             entrainment (eroded material joins the flow)
                             and (b) direct rainfall runoff (the mechanism
                             that actually starts a flood: without it, h
                             stays at 0 everywhere forever since erosion
                             and momentum both require h>0 to activate).
  2. Voellmy-Salm momentum - dry Coulomb friction (mu) + turbulent drag (xi).
  3. Infinite-slope FoS   - soil cohesion + friction angle (derived from
                             sand/clay/silt via a simplified pedotransfer
                             function) vs. pore-water pressure loading from
                             soil saturation and ponded flow depth.
  4. Erosion              - excess shear-stress model (Hanson & Simon
                             style): erosion proceeds where flow-induced
                             bed shear exceeds the soil's critical shear
                             strength, at a rate governed by erodibility.
  5. Soil saturation      - a simple "bucket" infiltration/drainage model
                             driven by CHIRPS rainfall.

IMPORTANT CALIBRATION CAVEAT: the pedotransfer functions (soil -> cohesion,
friction angle, erodibility) and the constants (Z_SLIP, POROSITY,
DRAINAGE_COEFF, INFIL_ALPHA, EROSION_K0, EROSION_RESIST_FRAC) below are
simplified, illustrative relationships, not lab-calibrated geotechnical
parameters for the Mandakini basin specifically. They are chosen to be
physically directionally correct (more clay -> more cohesion, more
friction resistance, less erodible; more sand -> higher friction angle,
more erodible) and numerically stable, but treat absolute FoS/erosion
magnitudes as approximate until validated against real geotechnical data
or the known 16-17 June 2013 event footprint.
"""
from __future__ import annotations
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization as fs
import optax

from model_valkyrie import PIGNN, NetworkConfig, DYNAMIC_STATE_DIM, build_model

# =============================================================================
# PHYSICAL CONSTANTS (see calibration caveat above)
# =============================================================================
G = 9.81
RHO_WATER = 1000.0
EPS = 1e-6

Z_SLIP = 1.5             # m, representative shallow failure-plane depth
SOIL_POROSITY = 0.4      # dimensionless
DRAINAGE_COEFF = 1.5e-6  # 1/s, soil-moisture drainage rate
INFIL_ALPHA = 0.01       # dimensionless, infiltration efficiency scaler
RUNOFF_MIN_FRAC = 0.05   # baseline runoff fraction even on dry/unsaturated soil
EROSION_K0 = 2.0e-7      # erodibility base rate, m per (Pa*s)
EROSION_RESIST_FRAC = 0.4  # fraction of cohesion resisting erosion onset

# node_static column indices (must match Module 1's NODE_FEATURE_NAMES order)
COL_SLOPE_RAD, COL_ASPECT_SIN, COL_ASPECT_COS = 1, 2, 3
COL_BDOD_0_5, COL_BDOD_5_15 = 4, 5
COL_CLAY_0_5, COL_CLAY_5_15 = 6, 7
COL_SAND_0_5, COL_SAND_5_15 = 8, 9
COL_SILT_0_5, COL_SILT_5_15 = 10, 11


# =============================================================================
# SOIL PARAMETER DERIVATION (SoilGrids raw units -> mechanics parameters)
# =============================================================================
def derive_soil_params(node_static):
    """SoilGrids units: bdod in cg/cm3 (=> kg/m3 via *10), clay/sand/silt in
    g/kg (=> fraction via /1000). Averages the 0-5cm and 5-15cm layers as a
    representative near-surface failure-plane soil column."""
    bdod_avg = 0.5 * (node_static[:, COL_BDOD_0_5] + node_static[:, COL_BDOD_5_15])
    clay_avg = 0.5 * (node_static[:, COL_CLAY_0_5] + node_static[:, COL_CLAY_5_15])
    sand_avg = 0.5 * (node_static[:, COL_SAND_0_5] + node_static[:, COL_SAND_5_15])

    bulk_density_kgm3 = bdod_avg * 10.0
    clay_frac = jnp.clip(clay_avg / 1000.0, 0.0, 1.0)
    sand_frac = jnp.clip(sand_avg / 1000.0, 0.0, 1.0)

    # simplified pedotransfer functions - see module docstring caveat
    friction_deg = 15.0 + 20.0 * sand_frac          # ~15 deg (clay) .. 35 deg (sand)
    friction_rad = jnp.radians(friction_deg)
    cohesion_pa = (2.0 + 18.0 * clay_frac) * 1000.0  # 2-20 kPa -> Pa

    erodibility = EROSION_K0 / (1.0 + 5.0 * clay_frac)  # sandier = more erodible

    return dict(
        bulk_density_kgm3=bulk_density_kgm3,
        clay_frac=clay_frac,
        sand_frac=sand_frac,
        friction_rad=friction_rad,
        cohesion_pa=cohesion_pa,
        erodibility=erodibility,
    )


# =============================================================================
# DIAGNOSTIC: INFINITE-SLOPE FACTOR OF SAFETY
# =============================================================================
def compute_fos(state, node_static, soil_params):
    """Infinite-slope FoS:
        FoS = [c' + max(sigma_n - u_w, 0)*tan(phi')] / max(tau_driving, eps)
    sigma_n = gamma*z*cos^2(theta)         (total normal stress on slip plane)
    tau_driving = gamma*z*sin(theta)cos(theta)
    u_w = pore pressure from soil saturation + any ponded surface depth h
    """
    h = state[:, 0]
    slope_rad = node_static[:, COL_SLOPE_RAD]
    saturation = jnp.clip(state[:, 3], 0.0, 1.0)

    gamma_soil = soil_params["bulk_density_kgm3"] * G  # N/m^3
    sigma_n = gamma_soil * Z_SLIP * jnp.cos(slope_rad) ** 2
    tau_driving = gamma_soil * Z_SLIP * jnp.sin(slope_rad) * jnp.cos(slope_rad)

    u_w = RHO_WATER * G * (Z_SLIP * saturation + h)
    u_w = jnp.minimum(u_w, sigma_n)  # pore pressure can't exceed total overburden

    resisting = soil_params["cohesion_pa"] + jnp.maximum(sigma_n - u_w, 0.0) * jnp.tan(
        soil_params["friction_rad"]
    )
    fos = resisting / jnp.maximum(tau_driving, EPS)
    # flat ground (tau_driving ~ 0) is not meaningfully "infinite FoS" for
    # display purposes - cap it so downstream classification/plots stay sane
    return jnp.clip(fos, 0.0, 10.0)


# =============================================================================
# DETERMINISTIC PHYSICS ENGINE (Module 3, layer A)
# =============================================================================
def mass_continuity_delta(h, u, v, senders, receivers, edge_len, dt, n_nodes):
    h_s, h_r = h[senders], h[receivers]
    u_s, u_r = u[senders], u[receivers]
    v_s, v_r = v[senders], v[receivers]
    h_face = 0.5 * (h_s + h_r)
    vel_face_mag = 0.5 * (jnp.sqrt(u_s**2 + v_s**2) + jnp.sqrt(u_r**2 + v_r**2))
    flux_per_len = (h_face * vel_face_mag) / (edge_len + EPS)
    outflow = jax.ops.segment_sum(flux_per_len, senders, num_segments=n_nodes)
    inflow = jax.ops.segment_sum(flux_per_len, receivers, num_segments=n_nodes)
    divergence = outflow - inflow
    return -divergence * dt


def voellmy_momentum_update(u, v, h, slope_rad, aspect_sin, aspect_cos, dt, mu, xi):
    """Semi-implicit (operator-split) Voellmy update: gravity accelerates
    the flow explicitly, then friction DECELERATES it toward (but never
    past) zero speed. A naive explicit update of du = (drive - friction)*dt
    is unconditionally unstable here: once h is small, Sf ~ speed^2/h
    explodes, and explicit Euler overshoots past zero speed into a huge
    reversed velocity, which explodes further next step. Clamping the
    friction decrement to the current speed makes this stable at any dt."""
    h_safe = jnp.maximum(h, EPS)
    active = (h_safe > 1e-4).astype(jnp.float32)

    drive_x = G * jnp.sin(slope_rad) * aspect_cos
    drive_y = G * jnp.sin(slope_rad) * aspect_sin
    u1 = u + drive_x * dt
    v1 = v + drive_y * dt
    speed1 = jnp.sqrt(u1**2 + v1**2)

    Sf = mu * jnp.cos(slope_rad) + (speed1**2) / (xi * h_safe + EPS)
    friction_decel = G * Sf * dt
    new_speed = jax.nn.relu(speed1 - friction_decel)
    scale = new_speed / (speed1 + EPS)

    u_next = jnp.where(active > 0, u1 * scale, 0.0)
    v_next = jnp.where(active > 0, v1 * scale, 0.0)
    return u_next, v_next


def erosion_delta(state, node_static, soil_params, dt):
    """Excess shear-stress erosion (Hanson & Simon style): bed shear from
    the moving flow (tau_flow ~ rho*g*h*sin(theta)) erodes material once it
    exceeds the soil's critical shear resistance."""
    h = state[:, 0]
    slope_rad = node_static[:, COL_SLOPE_RAD]
    tau_flow = RHO_WATER * G * h * jnp.sin(slope_rad)
    tau_critical = EROSION_RESIST_FRAC * soil_params["cohesion_pa"]
    d_erosion = soil_params["erodibility"] * jax.nn.relu(tau_flow - tau_critical) * dt
    return d_erosion  # >= 0 always (monotonic accumulation)


def rainfall_runoff_delta(rainfall_t, saturation, dt):
    """Direct rainfall -> surface runoff/ponding source term for h. This is
    the missing link that actually starts a flood/debris event: without it,
    h=0 everywhere forever regardless of rainfall (erosion and momentum both
    require h>0 to activate, so the whole simulation would stay inert).
    Runoff fraction rises with soil saturation (a saturated slope sheds
    almost all further rain as runoff - the real mechanism behind flash
    floods following prolonged antecedent rain), with a small baseline
    fraction even on drier soil (impervious rock/thin soil in this terrain)."""
    rainfall_depth_m = (rainfall_t / 1000.0) * (dt / 86400.0)  # m of rain this substep
    runoff_frac = jnp.clip(saturation, RUNOFF_MIN_FRAC, 0.95)
    return runoff_frac * rainfall_depth_m


def saturation_delta(state, rainfall_t, dt):
    """Simple bucket infiltration/drainage model. rainfall_t is mm/day;
    infiltration is scaled by dt/86400 so repeated sub-daily steps don't
    each apply a full day's worth of infiltration."""
    saturation = jnp.clip(state[:, 3], 0.0, 1.0)
    infiltration = INFIL_ALPHA * rainfall_t * (dt / 86400.0) * (1.0 - saturation)
    drainage = DRAINAGE_COEFF * saturation * dt  # dt already in seconds
    d_sat = infiltration - drainage
    return jnp.clip(d_sat, -saturation, 1.0 - saturation)


def physics_step(state, node_static, soil_params, senders, receivers,
                  edge_features, rainfall_t, dt, mu, xi):
    """One deterministic physics micro-step. Returns (next_state, diagnostics)."""
    n_nodes = state.shape[0]
    h, u, v = state[:, 0], state[:, 1], state[:, 2]
    slope_rad = node_static[:, COL_SLOPE_RAD]
    aspect_sin, aspect_cos = node_static[:, COL_ASPECT_SIN], node_static[:, COL_ASPECT_COS]
    edge_len = edge_features[:, 0]

    d_mass_flux = mass_continuity_delta(h, u, v, senders, receivers, edge_len, dt, n_nodes)
    d_erosion = erosion_delta(state, node_static, soil_params, dt)
    d_runoff = rainfall_runoff_delta(rainfall_t, jnp.clip(state[:, 3], 0.0, 1.0), dt)
    # entrainment coupling + direct rainfall runoff both source mass into h
    dh = d_mass_flux + d_erosion + d_runoff

    u_next, v_next = voellmy_momentum_update(u, v, h, slope_rad, aspect_sin, aspect_cos, dt, mu, xi)
    d_sat = saturation_delta(state, rainfall_t, dt)

    h_next = jnp.maximum(h + dh, 0.0)
    sat_next = jnp.clip(state[:, 3] + d_sat, 0.0, 1.0)
    erosion_next = state[:, 4] + d_erosion  # monotonic, never clipped downward

    next_state = jnp.stack([h_next, u_next, v_next, sat_next, erosion_next], axis=1)
    fos = compute_fos(state, node_static, soil_params)  # FoS of the PRE-step state
    diagnostics = {"fos": fos, "d_erosion": d_erosion}
    return next_state, diagnostics


def run_physics_rollout(node_static, state0, senders, receivers, edge_features,
                         rainfall_daily, n_substeps_per_day, mu=0.22, xi=500.0):
    """Runs the deterministic physics simulator across n_days (inferred from
    rainfall_daily) x n_substeps_per_day micro-steps via jax.lax.scan - the
    actual June-2013 retrodiction. No GNN or training involved."""
    n_days, n_nodes = rainfall_daily.shape
    dt = 86400.0 / n_substeps_per_day  # seconds per micro-step

    # repeat each day's rainfall across its substeps -> one flat sequence
    rainfall_substeps = jnp.repeat(rainfall_daily, n_substeps_per_day, axis=0)
    soil_params = derive_soil_params(node_static)

    def step(state, rainfall_t):
        next_state, diag = physics_step(state, node_static, soil_params, senders,
                                         receivers, edge_features, rainfall_t, dt, mu, xi)
        return next_state, (next_state, diag["fos"])

    step_jit = jax.jit(step)
    final_state, (state_seq, fos_seq) = jax.lax.scan(step_jit, state0, rainfall_substeps)
    return state_seq, fos_seq, dt  # (n_days*n_substeps, n_nodes, 5), (.., n_nodes)


# =============================================================================
# GNN EMULATOR TRAINING (Module 3, layer B) - teacher forced on the physics
# trajectory computed above
# =============================================================================
def make_emulator_train_step(model, optimizer):
    @jax.jit
    def train_step(params, opt_state, node_static, state_batch, rainfall_batch,
                    physics_delta_batch, edge_features, senders, receivers):
        def loss_fn(p):
            def per_sample(state, rainfall_t, target_delta):
                pred_delta = model.apply(p, node_static, state, rainfall_t,
                                          edge_features, senders, receivers)
                return jnp.mean((pred_delta - target_delta) ** 2)

            losses = jax.vmap(per_sample)(state_batch, rainfall_batch, physics_delta_batch)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def build_teacher_forcing_dataset(node_static, state0, state_seq, rainfall_substeps, soil_params,
                                   senders, receivers, edge_features, dt, mu, xi):
    """For every state along the ALREADY-COMPUTED physics trajectory, records
    the physics engine's own per-step delta as the training target - this is
    what the GNN is taught to reproduce."""
    states_in = jnp.concatenate([state0[None, :, :], state_seq[:-1]], axis=0)  # state BEFORE each step
    def get_delta(state, rainfall_t):
        next_state, _ = physics_step(state, node_static, soil_params, senders,
                                      receivers, edge_features, rainfall_t, dt, mu, xi)
        return next_state - state

    deltas = jax.vmap(get_delta)(states_in, rainfall_substeps)
    return states_in, deltas


def train_emulator(model, params, node_static, states_in, rainfall_substeps, deltas,
                    edge_features, senders, receivers, n_epochs, lr,
                    checkpoint_path, checkpoint_meta_path, checkpoint_every, log_every,
                    batch_size=64):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    start_epoch = 0
    if os.path.exists(checkpoint_path) and os.path.exists(checkpoint_meta_path):
        with open(checkpoint_path, "rb") as f:
            params = fs.from_bytes(params, f.read())
        with open(checkpoint_meta_path, "rb") as f:
            meta = pickle.load(f)
        opt_state, start_epoch = meta["opt_state"], meta["epoch"]
        print(f"[checkpoint] resuming emulator training from epoch {start_epoch}")

    train_step = make_emulator_train_step(model, optimizer)
    n_steps = states_in.shape[0]
    key = jax.random.PRNGKey(0)

    for epoch in range(start_epoch, n_epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, n_steps, shape=(min(batch_size, n_steps),), replace=False)
        params, opt_state, loss = train_step(
            params, opt_state, node_static, states_in[idx], rainfall_substeps[idx],
            deltas[idx], edge_features, senders, receivers,
        )
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"[emulator] epoch {epoch:5d}/{n_epochs}  loss={float(loss):.6f}")
        if epoch % checkpoint_every == 0 and epoch > start_epoch:
            with open(checkpoint_path, "wb") as f:
                f.write(fs.to_bytes(params))
            with open(checkpoint_meta_path, "wb") as f:
                pickle.dump({"opt_state": opt_state, "epoch": epoch}, f)
            print(f"[checkpoint] saved at epoch {epoch}")

    with open(checkpoint_path, "wb") as f:
        f.write(fs.to_bytes(params))
    with open(checkpoint_meta_path, "wb") as f:
        pickle.dump({"opt_state": opt_state, "epoch": n_epochs}, f)
    return params


# =============================================================================
# CACHE EXPORT
# =============================================================================
def export_simulation_cache(path, graph, state_seq, fos_seq, n_substeps_per_day):
    H, W = graph["out_shape"]
    valid = graph["valid_mask"]
    node_id_grid = graph["node_id_grid"]
    ii, jj = np.where(valid)
    node_ids = node_id_grid[ii, jj]

    # subsample to one frame per DAY for the cache (30 frames), not every
    # micro-substep, to keep the dashboard file small - full substep detail
    # stays available in-memory during the run if needed
    daily_idx = np.arange(n_substeps_per_day - 1, state_seq.shape[0], n_substeps_per_day)
    state_seq_np = np.asarray(state_seq)[daily_idx]
    fos_seq_np = np.asarray(fos_seq)[daily_idx]
    T = state_seq_np.shape[0]

    depth = np.full((T, H, W), np.nan, dtype=np.float32)
    vel_u = np.full((T, H, W), np.nan, dtype=np.float32)
    vel_v = np.full((T, H, W), np.nan, dtype=np.float32)
    saturation = np.full((T, H, W), np.nan, dtype=np.float32)
    erosion = np.full((T, H, W), np.nan, dtype=np.float32)
    fos_grid = np.full((T, H, W), np.nan, dtype=np.float32)

    for t in range(T):
        depth[t, ii, jj] = state_seq_np[t, node_ids, 0]
        vel_u[t, ii, jj] = state_seq_np[t, node_ids, 1]
        vel_v[t, ii, jj] = state_seq_np[t, node_ids, 2]
        saturation[t, ii, jj] = state_seq_np[t, node_ids, 3]
        erosion[t, ii, jj] = state_seq_np[t, node_ids, 4]
        fos_grid[t, ii, jj] = fos_seq_np[t, node_ids]

    elevation_grid = np.full((H, W), np.nan, dtype=np.float32)
    elevation_grid[ii, jj] = graph["raw_elevation"]

    np.savez_compressed(
        path, elevation=elevation_grid, depth=depth, velocity_u=vel_u, velocity_v=vel_v,
        soil_saturation=saturation, cumulative_erosion=erosion, fos=fos_grid,
        cell_res=graph["cell_res"], valid_mask=valid,
    )
    print(f"[cache] saved {path}  (T={T} daily frames, grid={H}x{W})")


# =============================================================================
# TEST / SMOKE RUN
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/valkyrie")
    from ingest_valkyrie import IngestConfig, ingest

    ing_cfg = IngestConfig(
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
        chirps_dir="testdata", chirps_glob="chirps-v2.0.2013.06.*.tif",
    )
    graph, rainfall, _ = ingest(ing_cfg)
    node_static = jnp.array(graph["node_features"])
    senders, receivers = jnp.array(graph["senders"]), jnp.array(graph["receivers"])
    edge_features = jnp.array(graph["edge_features"])
    n_nodes = graph["n_node"]

    from model_valkyrie import init_dynamic_state
    state0 = init_dynamic_state(n_nodes)

    print("=== A) Deterministic physics rollout (the actual retrodiction) ===")
    n_substeps_per_day = 12  # 2-hour micro-steps; raise for real runs
    state_seq, fos_seq, dt = run_physics_rollout(
        node_static, state0, senders, receivers, edge_features,
        jnp.array(rainfall), n_substeps_per_day,
    )
    print("state_seq shape:", state_seq.shape, " dt(s):", dt)
    print("any NaN in state_seq:", bool(jnp.isnan(state_seq).any()))
    print("any NaN in fos_seq:", bool(jnp.isnan(fos_seq).any()))

    # sanity checks tied to the synthetic rainfall spike on days 15-18
    day_end_idx = lambda day: day * n_substeps_per_day - 1  # index of last substep of `day`
    fos_before = float(jnp.mean(fos_seq[day_end_idx(14)]))
    fos_during = float(jnp.mean(fos_seq[day_end_idx(17)]))
    h_before = float(jnp.mean(state_seq[day_end_idx(14), :, 0]))
    h_during = float(jnp.mean(state_seq[day_end_idx(17), :, 0]))
    erosion_before = float(jnp.mean(state_seq[day_end_idx(14), :, 4]))
    erosion_end = float(jnp.mean(state_seq[-1, :, 4]))
    sat_before = float(jnp.mean(state_seq[day_end_idx(14), :, 3]))
    sat_during = float(jnp.mean(state_seq[day_end_idx(17), :, 3]))

    print(f"\nmean FoS  day14={fos_before:.3f}  day17(spike)={fos_during:.3f}  "
          f"(expect FoS to DROP during the rainfall spike)")
    print(f"mean h    day14={h_before:.4f}  day17(spike)={h_during:.4f}  "
          f"(expect h to RISE during the spike)")
    print(f"mean sat  day14={sat_before:.3f}  day17(spike)={sat_during:.3f}  "
          f"(expect saturation to RISE during the spike)")
    print(f"mean erosion day14={erosion_before:.5f}  end-of-month={erosion_end:.5f}  "
          f"(expect monotonic increase)")
    assert erosion_end >= erosion_before, "erosion must be monotonically non-decreasing"

    print("\n=== B) Training GNN emulator (teacher-forced on physics trajectory) ===")
    net_cfg = NetworkConfig(hidden_dim=32, n_message_passing_steps=3)
    model = build_model(net_cfg)
    key = jax.random.PRNGKey(0)
    params = model.init(key, node_static, state0, jnp.array(rainfall[0]),
                         edge_features, senders, receivers)

    rainfall_substeps = jnp.repeat(jnp.array(rainfall), n_substeps_per_day, axis=0)
    soil_params = derive_soil_params(node_static)
    states_in, deltas = build_teacher_forcing_dataset(
        node_static, state0, state_seq, rainfall_substeps, soil_params,
        senders, receivers, edge_features, dt, mu=0.22, xi=500.0,
    )
    print("teacher-forcing dataset:", states_in.shape, deltas.shape)

    ckpt = "test_emulator_ckpt.msgpack"
    ckpt_meta = "test_emulator_ckpt_meta.pkl"
    for f in (ckpt, ckpt_meta):
        if os.path.exists(f):
            os.remove(f)

    params = train_emulator(model, params, node_static, states_in, rainfall_substeps,
                             deltas, edge_features, senders, receivers,
                             n_epochs=15, lr=1e-3, checkpoint_path=ckpt,
                             checkpoint_meta_path=ckpt_meta, checkpoint_every=5,
                             log_every=3, batch_size=32)

    print("\n=== Exporting cache ===")
    export_simulation_cache("test_valkyrie_cache.npz", graph, state_seq, fos_seq,
                             n_substeps_per_day)
    loaded = np.load("test_valkyrie_cache.npz")
    print("cache keys:", list(loaded.keys()))
    print("depth shape:", loaded["depth"].shape, " fos shape:", loaded["fos"].shape)

    for f in (ckpt, ckpt_meta, "test_valkyrie_cache.npz"):
        if os.path.exists(f):
            os.remove(f)
    print("\nall Module 3 smoke tests passed.")
