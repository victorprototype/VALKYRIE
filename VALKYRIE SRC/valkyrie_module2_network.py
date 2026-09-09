"""
VALKYRIE - Module 2: Graphical Network Creator
================================================
Defines the PIGNN architecture (Encoder-Processor-Decoder) that Module 3
(Simulator) will drive forward in time. This module owns:
  - The dynamic state schema (what physical quantities live on each node
    and evolve over time)
  - How static terrain/soil features, the current dynamic state, and the
    current day's rainfall forcing are fused into the network
  - The message-passing graph network itself

No physics equations live here (that's Module 3's job) - this module only
defines "given the current situation on the graph, what state-change does
the network propose", which Module 3 then constrains with physics laws.

Dynamic state schema (5 channels per node), in this fixed order:
    [h, u, v, soil_saturation, cumulative_erosion]
    h                 : debris/water depth (m)
    u, v              : depth-averaged velocity components (m/s)
    soil_saturation   : 0-1, proportion of pore space filled with water -
                        builds from rainfall infiltration, drives pore-water
                        pressure in Module 3's Factor-of-Safety calculation
    cumulative_erosion: m, total material removed from that node so far -
                        monotonically evolves as debris scours the slope
"""
from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn

DYNAMIC_STATE_NAMES = ["h", "u", "v", "soil_saturation", "cumulative_erosion"]
DYNAMIC_STATE_DIM = len(DYNAMIC_STATE_NAMES)
RAINFALL_INPUT_DIM = 1  # current day's rainfall rate, mm - one scalar per node


@dataclass
class NetworkConfig:
    hidden_dim: int = 64
    n_message_passing_steps: int = 8


# =============================================================================
# BUILDING BLOCKS
# =============================================================================
class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(self.out_dim)(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        return x


class MessagePassingLayer(nn.Module):
    """One message-passing round over the terrain graph: edges gather
    sender+receiver node state into a message (MLP), messages are summed
    at each receiver (segment_sum), and node state is updated from
    [old_state, aggregated_messages] (MLP). Residual connections on both
    node and edge updates stabilize deep processor stacks."""
    hidden_dim: int

    @nn.compact
    def __call__(self, node_h, edge_h, senders, receivers):
        n_nodes = node_h.shape[0]
        msg_in = jnp.concatenate([node_h[senders], node_h[receivers], edge_h], axis=-1)
        edge_h_new = MLP(self.hidden_dim, self.hidden_dim)(msg_in)

        agg = jax.ops.segment_sum(edge_h_new, receivers, num_segments=n_nodes)
        node_h_new = MLP(self.hidden_dim, self.hidden_dim)(
            jnp.concatenate([node_h, agg], axis=-1)
        )
        return node_h + node_h_new, edge_h + edge_h_new


# =============================================================================
# ENCODER - PROCESSOR - DECODER
# =============================================================================
class PIGNN(nn.Module):
    """
    Inputs per forward call (one simulation micro-step):
      node_static   (n_nodes, 12)  - terrain + soil features from Module 1
      dynamic_state (n_nodes, 5)   - current [h, u, v, saturation, erosion]
      rainfall_t    (n_nodes,) or (n_nodes, 1) - current timestep's rainfall (mm)
      edge_features (n_edges, 2)   - [distance, delta_z] from Module 1
      senders, receivers (n_edges,) - graph connectivity from Module 1

    Output:
      deltas (n_nodes, 5) - proposed [dh, du, dv, dsaturation, derosion]
      Module 3 applies physics constraints/corrections to these before
      advancing the state.
    """
    hidden_dim: int = 64
    n_message_passing_steps: int = 8

    @nn.compact
    def __call__(self, node_static, dynamic_state, rainfall_t, edge_features,
                 senders, receivers):
        rainfall_t = rainfall_t.reshape(-1, 1)  # ensure (n_nodes, 1)

        # --- Encoder: fuse static terrain/soil + dynamic state + today's
        # rainfall forcing into one latent representation per node
        node_in = jnp.concatenate([node_static, dynamic_state, rainfall_t], axis=-1)
        node_h = MLP(self.hidden_dim, self.hidden_dim)(node_in)
        edge_h = MLP(self.hidden_dim, self.hidden_dim)(edge_features)

        # --- Processor: message passing propagates influence across the
        # terrain graph (upslope rainfall/erosion affecting downslope nodes)
        for _ in range(self.n_message_passing_steps):
            node_h, edge_h = MessagePassingLayer(self.hidden_dim)(
                node_h, edge_h, senders, receivers
            )

        # --- Decoder: predict the 5 state increments
        return MLP(self.hidden_dim, DYNAMIC_STATE_DIM, layer_norm=False)(node_h)


# =============================================================================
# HELPERS
# =============================================================================
def init_dynamic_state(n_nodes: int, initial_saturation: float = 0.3):
    """Zero-initialized [h, u, v, erosion], with soil_saturation seeded at a
    baseline antecedent-moisture value rather than zero (bone-dry soil at
    the start of a simulated monsoon month is physically implausible and
    would delay pore-pressure buildup in Module 3's FoS calc). Override
    initial_saturation if you have a real antecedent-moisture estimate."""
    state = jnp.zeros((n_nodes, DYNAMIC_STATE_DIM), dtype=jnp.float32)
    state = state.at[:, 3].set(initial_saturation)
    return state


def build_model(cfg: NetworkConfig) -> PIGNN:
    return PIGNN(hidden_dim=cfg.hidden_dim, n_message_passing_steps=cfg.n_message_passing_steps)


def init_params(model: PIGNN, key, node_static, dynamic_state, rainfall_t,
                 edge_features, senders, receivers):
    return model.init(key, node_static, dynamic_state, rainfall_t,
                       edge_features, senders, receivers)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/valkyrie")
    from ingest_valkyrie import IngestConfig, ingest

    ing_cfg = IngestConfig(
        dem_path="/home/claude/valkyrie/testdata/dem.tif",
        soil_paths={
            ("bdod", "0-5cm"): "/home/claude/valkyrie/testdata/bdod_0-5cm.tif",
            ("bdod", "5-15cm"): "/home/claude/valkyrie/testdata/bdod_5-15cm.tif",
            ("clay", "0-5cm"): "/home/claude/valkyrie/testdata/clay_0-5cm.tif",
            ("clay", "5-15cm"): "/home/claude/valkyrie/testdata/clay_5-15cm.tif",
            ("sand", "0-5cm"): "/home/claude/valkyrie/testdata/sand_0-5cm.tif",
            ("sand", "5-15cm"): "/home/claude/valkyrie/testdata/sand_5-15cm.tif",
            ("silt", "0-5cm"): "/home/claude/valkyrie/testdata/silt_0-5cm.tif",
            ("silt", "5-15cm"): "/home/claude/valkyrie/testdata/silt_5-15cm.tif",
        },
        chirps_dir="/home/claude/valkyrie/testdata",
        chirps_glob="chirps-v2.0.2013.06.*.tif",
    )
    graph, rainfall, _ = ingest(ing_cfg)

    node_static = jnp.array(graph["node_features"])
    senders = jnp.array(graph["senders"])
    receivers = jnp.array(graph["receivers"])
    edge_features = jnp.array(graph["edge_features"])
    n_nodes = graph["n_node"]

    print("node_static shape (expect n_nodes x 12):", node_static.shape)

    state0 = init_dynamic_state(n_nodes)
    print("state0 shape (expect n_nodes x 5):", state0.shape,
          " baseline saturation sample:", state0[0, 3])

    rainfall_day1 = jnp.array(rainfall[0])  # (n_nodes,)
    print("rainfall_day1 shape:", rainfall_day1.shape)

    net_cfg = NetworkConfig(hidden_dim=64, n_message_passing_steps=8)
    model = build_model(net_cfg)
    key = jax.random.PRNGKey(0)
    params = init_params(model, key, node_static, state0, rainfall_day1,
                          edge_features, senders, receivers)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("trainable parameters:", n_params)

    deltas = model.apply(params, node_static, state0, rainfall_day1,
                          edge_features, senders, receivers)
    print("deltas shape (expect n_nodes x 5):", deltas.shape)
    print("sample deltas:", deltas[:3])
    print("any NaN in deltas:", jnp.isnan(deltas).any())

    # confirm gradients flow end to end (sanity check before Module 3 wires
    # in the actual physics-informed loss)
    def dummy_loss(p):
        d = model.apply(p, node_static, state0, rainfall_day1,
                         edge_features, senders, receivers)
        return jnp.mean(d**2)

    grads = jax.grad(dummy_loss)(params)
    leaf_norms = [jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(grads)]
    print("grad norms sample:", leaf_norms[:3],
          " any NaN:", any(bool(jnp.isnan(n)) for n in leaf_norms))

    # confirm the model handles a DIFFERENT rainfall day (dynamic forcing
    # actually changes the output, not just static features)
    rainfall_day16 = jnp.array(rainfall[15])
    deltas_day16 = model.apply(params, node_static, state0, rainfall_day16,
                                edge_features, senders, receivers)
    diff = jnp.mean(jnp.abs(deltas_day16 - deltas))
    print("mean |deltas(day16) - deltas(day1)| (should be > 0):", diff)
