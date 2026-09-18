"""
Injects the 24h simulation layer + timelapse control bar into the base DEM
viewer HTML, and restyles the whole UI to a plain, flat engineering-tool
look (light panels, square corners, no blur/glow effects).

Order of operations matters:
  1. inject simulation panel HTML
  2. inject simulation JS
  3. replace the ENTIRE <style> block last
Step 3 wipes whatever the base viewer put in <style>, so it must come after
everything else, and no other step may rely on the base viewer's CSS.
"""
from pack_simulation_data import pack_simulation

BASE_HTML_PATH = "/kaggle/working/valkyrie/physics/dem_3d_viewer.html"
OUT_PATH = "/kaggle/working/valkyrie/physics/dem_3d_viewer_simulation.html"
SIM_NPZ_PATH = "/kaggle/working/valkyrie/physics/simulation_24h.npz"

sim = pack_simulation(SIM_NPZ_PATH)

html = open(BASE_HTML_PATH).read()

# ---------------- simulation / timelapse panel ----------------
NEW_PANEL_HTML = r"""
<div id="sim-panel">
  <div id="sim-header">
    <h1>24h rainfall &rarr; failure &rarr; runout simulation (compressed to 30s)</h1>
    <button id="sim-toggle" title="Collapse">&#8250;</button>
  </div>
  <div id="sim-body">
    <div id="sim-controls">
      <button id="sim-play">&#9654; Play</button>
      <input type="range" id="sim-scrub" min="0" max="1000" value="0">
      <button id="sim-record">&#9679; Record 30s clip</button>
    </div>
    <div id="sim-readout">
      <div>sim hour <b id="sim-hour">0</b> / 24</div>
      <div>rainfall <b id="sim-rain">0</b> mm/hr</div>
      <div>basin area destabilized (FS&lt;1) <b id="sim-failed">0</b>%</div>
    </div>
    <div id="sim-legend-row">
      <div><span class="sim-swatch" style="background:#2e7d4f"></span>stable (FS&gt;2)</div>
      <div><span class="sim-swatch" style="background:#d4b83f"></span>marginal (FS&approx;1.2)</div>
      <div><span class="sim-swatch" style="background:#c0392b"></span>failing (FS&lt;1)</div>
      <div><span class="sim-swatch" style="background:#8a5a3b"></span>mobile debris (Voellmy runout)</div>
      <div><span class="sim-swatch" style="background:#705433"></span>scar (failed at some point, final extent)</div>
    </div>
    <div id="sim-caveat">
      Demonstration methodology, not a validated hazard forecast: initial water table depth,
      background infiltration rate, storm sub-daily timing, and Voellmy friction parameters are
      all assumed (see accompanying scripts for exact values) -- not measured or calibrated to
      this basin. TRIGRS' own manual notes results are "very sensitive" to the initial
      water-table assumption in particular.
    </div>
  </div>
</div>
"""
html = html.replace('<div id="hint">', NEW_PANEL_HTML + '<div id="hint">')

# ---------------- simulation JS ----------------
NEW_JS = r"""
// =========================== 24h SIMULATION LAYER ===========================
const SIM = __SIM_JSON__;
const SIM_B64 = "__SIM_B64__";
const SIM_EROSION_B64 = "__SIM_EROSION_B64__";

function b64ToUint8(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const buf = new Uint8Array(len);
  for (let i = 0; i < len; i++) buf[i] = binary.charCodeAt(i);
  return buf;
}
// packed layout: [hour][node][2] -> (fs_byte, depth_byte), row-major
const simPacked = b64ToUint8(SIM_B64);
const N_HOURS = SIM.n_hours, N_NODES = SIM.n_nodes;

// static per-node total accumulated slip depth (NOT a per-hour series --
// route_failures() in voellmy_runout.py only returns the final running
// total, so this is the end-of-simulation scar extent, not something that
// can animate progressively frame-by-frame yet)
const erosionPacked = b64ToUint8(SIM_EROSION_B64);
function erosionByteAt(nodeIdx) { return erosionPacked[nodeIdx]; }

function simByteAt(hourIdx, nodeIdx, channel) {
  return simPacked[(hourIdx * N_NODES + nodeIdx) * 2 + channel];
}

// FS colour thresholds derived from the quantisation scale actually used by
// pack_simulation(), rather than hardcoded against an assumed 3.0. fs01 is
// the stored byte / 255, i.e. FS / fs_max_display, so an absolute FS value
// maps to fs01 = FS / fs_max_display.
const FS_MAX_DISPLAY = SIM.fs_max_display;
const FS01_FAILING = 1.0 / FS_MAX_DISPLAY;   // FS = 1.0 -> fully opaque red
const FS01_STABLE  = 1.8 / FS_MAX_DISPLAY;   // FS >= 1.8 -> fully transparent

// overlay mesh sharing the SAME geometry as the terrain, so vertical
// exaggeration changes on the terrain automatically apply here too
const overlayColors = new Float32Array(N_NODES * 4);
geometry.setAttribute('color', new THREE.BufferAttribute(overlayColors, 4));

const overlayMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  side: THREE.DoubleSide,
  vertexShader: `
    attribute vec4 color;
    varying vec4 vColor;
    void main() {
      vColor = color;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    varying vec4 vColor;
    void main() {
      gl_FragColor = vColor;
    }
  `
});
const overlayMesh = new THREE.Mesh(geometry, overlayMaterial);
overlayMesh.renderOrder = 1;
scene.add(overlayMesh);

function riskColor(fs01, depth01, erosion01, out) {
  // fs01: stored FS / fs_max_display. Only failing/marginal ground is drawn;
  // stable ground is fully transparent so the DEM texture shows through.
  let r, g, b, a;
  if (fs01 < FS01_FAILING) {
    r = 0.831; g = 0.180; b = 0.180; // solid red
    a = 1.0;
  } else if (fs01 < FS01_STABLE) {
    const t = (fs01 - FS01_FAILING) / (FS01_STABLE - FS01_FAILING);
    r = 0.831; g = 0.180; b = 0.180;
    a = 1.0 - t; // fade out toward stable
  } else {
    r = 0; g = 0; b = 0; a = 0.0; // stable ground: invisible
  }

  // permanent low-opacity scar: ground that has failed at ANY point in the
  // run, shown faintly even after debris has moved on and FS has nominally
  // "recovered" -- this is a static end-state extent (see erosionByteAt),
  // not something that grows hour-by-hour with the rest of the playback
  if (erosion01 > 0.03 && a < 0.25) {
    const et = Math.min(1, erosion01 * 1.5) * 0.22;
    r = r * (1 - et) + 0.44 * et;
    g = g * (1 - et) + 0.33 * et;
    b = b * (1 - et) + 0.20 * et;
    a = Math.max(a, et);
  }

  if (depth01 > 0.02) {
    const dt = Math.min(1, depth01 * 2.0);
    r = r * (1 - dt) + 0.541 * dt;
    g = g * (1 - dt) + 0.353 * dt;
    b = b * (1 - dt) + 0.231 * dt;
    a = Math.max(a, dt); // mobile debris always shows, even if FS recovered
  }
  out[0] = r; out[1] = g; out[2] = b; out[3] = a;
}

let simPlaying = false;
let simHour = 0.0;   // fractional hour, 0..24
const SIM_DURATION_S = 30.0;
const tmpColor = [0, 0, 0, 0];

function applySimFrame(hourFloat) {
  const h0 = Math.max(0, Math.min(N_HOURS - 1, Math.floor(hourFloat)));
  const h1 = Math.min(N_HOURS - 1, h0 + 1);
  const frac = Math.max(0, Math.min(1, hourFloat - h0));

  const colorAttr = geometry.attributes.color;
  for (let i = 0; i < N_NODES; i++) {
    const fsA = simByteAt(h0, i, 0), fsB = simByteAt(h1, i, 0);
    const dpA = simByteAt(h0, i, 1), dpB = simByteAt(h1, i, 1);
    const fs01 = ((fsA * (1 - frac) + fsB * frac) / 255);
    const depth01 = ((dpA * (1 - frac) + dpB * frac) / 255);
    const erosion01 = erosionByteAt(i) / 255;
    riskColor(fs01, depth01, erosion01, tmpColor);
    colorAttr.setXYZW(i, tmpColor[0], tmpColor[1], tmpColor[2], tmpColor[3]);
  }
  colorAttr.needsUpdate = true;

  const hIdx = Math.min(N_HOURS - 1, Math.round(hourFloat));
  document.getElementById('sim-hour').textContent = hourFloat.toFixed(1);
  document.getElementById('sim-rain').textContent = (SIM.hourly_rain_mm[hIdx] || 0).toFixed(1);
  document.getElementById('sim-failed').textContent = (SIM.failed_frac_per_hour[hIdx] * 100).toFixed(1);
  document.getElementById('sim-scrub').value = Math.round((hourFloat / N_HOURS) * 1000);

  updateRain(SIM.hourly_rain_mm[hIdx] || 0);
}

// ---- simple rain particle system, density/speed driven by current rainfall ----
const RAIN_MAX = 4000;
const rainGeom = new THREE.BufferGeometry();
const rainPos = new Float32Array(RAIN_MAX * 3);
const rainSpan = Math.max(widthM, heightM);
for (let i = 0; i < RAIN_MAX; i++) {
  rainPos[i * 3 + 0] = (Math.random() - 0.5) * rainSpan;
  rainPos[i * 3 + 1] = Math.random() * elevMax * 2.5;
  rainPos[i * 3 + 2] = (Math.random() - 0.5) * rainSpan;
}
rainGeom.setAttribute('position', new THREE.BufferAttribute(rainPos, 3));
const rainMaterial = new THREE.PointsMaterial({ color: 0x9db4d9, size: 22, transparent: true, opacity: 0.0 });
const rainPoints = new THREE.Points(rainGeom, rainMaterial);
scene.add(rainPoints);
let currentActiveRain = 0;

function updateRain(mmPerHr) {
  currentActiveRain = Math.max(0, mmPerHr);
  rainMaterial.opacity = Math.min(0.55, currentActiveRain / 40.0);
}

function stepRain(dt) {
  if (currentActiveRain <= 0) return;
  const speed = (400 + currentActiveRain * 25) * dt;
  const pos = rainGeom.attributes.position;
  const activeCount = Math.min(RAIN_MAX, Math.round(RAIN_MAX * Math.min(1, currentActiveRain / 30)));
  for (let i = 0; i < activeCount; i++) {
    let y = pos.getY(i) - speed;
    if (y < 0) y = elevMax * 2.5;
    pos.setY(i, y);
  }
  pos.needsUpdate = true;
}

// ---- playback control ----
let simLastT = null;
function simTick(nowMs) {
  if (!simPlaying) return;
  const now = nowMs / 1000;
  if (simLastT === null) simLastT = now;
  const dt = now - simLastT;
  simLastT = now;
  simHour += (dt / SIM_DURATION_S) * N_HOURS;
  if (simHour >= N_HOURS) {
    simHour = N_HOURS - 0.001;
    simPlaying = false;
    document.getElementById('sim-play').innerHTML = '&#9654; Play';
    if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
  }
  applySimFrame(simHour);
  stepRain(dt);
  requestAnimationFrame(simTick);
}

// NOTE: overlay alpha is carried entirely by the per-vertex vec4 colour
// attribute set in riskColor(). overlayMaterial is a ShaderMaterial with no
// `opacity` uniform, so assigning overlayMaterial.opacity here would be a
// no-op -- deliberately omitted rather than left in as a dead line.
document.getElementById('sim-play').addEventListener('click', () => {
  simPlaying = !simPlaying;
  document.getElementById('sim-play').innerHTML = simPlaying ? '&#10074;&#10074; Pause' : '&#9654; Play';
  if (simPlaying) { simLastT = null; requestAnimationFrame(simTick); }
});
document.getElementById('sim-scrub').addEventListener('input', (e) => {
  simPlaying = false;
  document.getElementById('sim-play').innerHTML = '&#9654; Play';
  simHour = (parseFloat(e.target.value) / 1000) * N_HOURS;
  applySimFrame(simHour);
});
document.getElementById('sim-toggle').addEventListener('click', () => {
  const el = document.getElementById('sim-panel');
  const collapsed = el.classList.toggle('collapsed');
  document.getElementById('sim-toggle').innerHTML = collapsed ? '&#8249;' : '&#8250;';
});

// ---- video export via canvas.captureStream + MediaRecorder ----
let mediaRecorder = null;
let recordedChunks = [];
document.getElementById('sim-record').addEventListener('click', () => {
  const btn = document.getElementById('sim-record');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  if (!renderer.domElement.captureStream) {
    alert('This browser does not support canvas video capture. Try Chrome or Edge.');
    return;
  }
  const stream = renderer.domElement.captureStream(30);
  let mimeType = 'video/webm;codecs=vp9';
  if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/webm';
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    btn.classList.remove('recording');
    btn.innerHTML = '&#9679; Record 30s clip';
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'mandakini_simulation.webm';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  };
  mediaRecorder.start();
  btn.classList.add('recording');
  btn.innerHTML = '&#9679; Recording...';
  simHour = 0; simPlaying = true; simLastT = null;
  document.getElementById('sim-play').innerHTML = '&#10074;&#10074; Pause';
  requestAnimationFrame(simTick);
});

applySimFrame(0);
"""

NEW_JS = NEW_JS.replace("__SIM_JSON__", str({
    "n_hours": sim["n_hours"], "n_nodes": sim["n_nodes"],
    "failed_frac_per_hour": sim["failed_frac_per_hour"],
    "hourly_rain_mm": sim["hourly_rain_mm"],
    "fs_max_display": sim["fs_max_display"],
    "depth_max_display": sim["depth_max_display"],
    "erosion_max_display": sim["erosion_max_display"],
}).replace("'", '"'))
NEW_JS = NEW_JS.replace("__SIM_B64__", sim["b64"])
NEW_JS = NEW_JS.replace("__SIM_EROSION_B64__", sim["erosion_b64"])

# Inject into the LAST </script> only. The base viewer has two script tags
# (the three.js CDN tag and the main inline one); a plain str.replace() hits
# both, silently duplicating the entire multi-MB simulation payload into the
# CDN tag -- where it is inert (a <script src=...> ignores inline content)
# but still doubles the output file size.
_last_script = html.rindex("</script>")
html = html[:_last_script] + NEW_JS + "\n" + html[_last_script:]

# ---------------- full UI restyle (MUST be last) ----------------
# Replaces the base viewer's entire <style> block. Any earlier step that
# depended on the old CSS would be clobbered here, so keep this at the end.
FULL_CSS = r"""<style>
  :root {
    --bg: #1a1a1a; --panel: #e9e9e7; --panel-border: #999;
    --text: #111; --text-dim: #555; --accent: #2555a4;
    --mono: Consolas, "Courier New", monospace;
    --sans: Arial, Helvetica, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin:0; padding:0; width:100%; height:100%; background:var(--bg); overflow:hidden; font-family:var(--sans); }
  #canvas-wrap { position:absolute; inset:0; }
  canvas { display:block; }

  #panel {
    position:absolute; top:10px; left:10px; width:230px;
    background:var(--panel); border:1px solid var(--panel-border); border-radius:2px;
    padding:10px; color:var(--text); font-size:12px; overflow-y:auto;
  }
  #panel.collapsed { width:auto; max-height:none; padding:6px; }
  #panel.collapsed #panel-body, #panel.collapsed #panel-title { display:none; }
  #panel-header { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  #panel h1 { font-size:12.5px; font-weight:bold; margin:0 0 2px 0; }
  #panel .sub { font-size:10.5px; color:var(--text-dim); margin:0 0 10px 0; line-height:1.4; }

  #panel-toggle, #legend-toggle, #sim-toggle {
    flex-shrink:0; width:20px; height:20px; padding:0;
    background:#d4d4d2; border:1px solid var(--panel-border); border-radius:2px;
    color:var(--text); font-size:12px; line-height:1; cursor:pointer;
    display:flex; align-items:center; justify-content:center; font-family:var(--sans);
  }
  #panel-toggle:hover, #legend-toggle:hover, #sim-toggle:hover { background:#c8c8c6; }

  #legend-panel {
    position:absolute; top:10px; right:10px; width:230px;
    background:var(--panel); border:1px solid var(--panel-border); border-radius:2px;
    padding:10px; color:var(--text); font-size:12px; overflow-y:auto;
  }
  #legend-panel.collapsed { width:auto; max-height:none; padding:6px; }
  #legend-panel.collapsed #legend-body, #legend-panel.collapsed #legend-title { display:none; }
  #legend-header { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  #legend-panel h1 { font-size:12.5px; font-weight:bold; margin:0 0 8px 0; }
  #legend-layer-name { font-size:11.5px; font-weight:bold; color:var(--text); margin-bottom:5px; }
  #legend-desc { font-size:11px; color:#333; line-height:1.5; margin:0 0 10px 0; }
  #legend-bar-wrap { margin-bottom:6px; }
  #legend-bar-img { width:100%; height:14px; border:1px solid #aaa; display:block; }
  #legend-labels { display:flex; justify-content:space-between; font-family:var(--mono); font-size:10px; color:var(--text-dim); margin-top:4px; }
  #legend-labels .mid { position:absolute; left:50%; transform:translateX(-50%); }
  #legend-tick-row { position:relative; height:14px; margin-top:4px; }
  #legend-tick-row span { position:absolute; transform:translateX(-50%); font-family:var(--mono); font-size:10px; color:var(--text-dim); }
  .legend-approx-note { font-size:10px; color:#777; margin-top:6px; line-height:1.4; }
  .legend-swatch-row { display:flex; align-items:center; gap:6px; margin-bottom:5px; font-size:11px; }
  .legend-swatch { width:12px; height:12px; border:1px solid #888; flex-shrink:0; }

  .section-label { font-size:10.5px; color:var(--text-dim); margin:10px 0 5px 0; text-transform:uppercase; }
  .section-label:first-of-type { margin-top:0; }
  .layer-btn {
    display:block; width:100%; text-align:left; padding:5px 8px; margin-bottom:3px;
    background:#d4d4d2; border:1px solid #aaa; border-radius:2px;
    color:var(--text); font-size:11.5px; cursor:pointer; font-family:var(--sans);
  }
  .layer-btn:hover { background:#c8c8c6; }
  .layer-btn.active { border-color:var(--accent); background:#d3e0f2; font-weight:bold; }
  .row { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
  .row label { font-size:11px; color:var(--text-dim); }
  .row input[type=range] { width:110px; accent-color:var(--accent); }
  .row .val { font-family:var(--mono); font-size:11px; color:var(--text); width:32px; text-align:right; }
  .toggle-row { display:flex; align-items:center; gap:6px; margin-bottom:6px; font-size:11px; color:var(--text-dim); }
  .toggle-row input { accent-color:var(--accent); }
  #stats { margin-top:10px; padding-top:8px; border-top:1px solid var(--panel-border); font-family:var(--mono); font-size:10.5px; color:#333; line-height:1.7; }
  #stats b { color:var(--text); font-weight:bold; }
  #hint { position:absolute; bottom:10px; left:10px; font-size:10.5px; color:#999; font-family:var(--sans); pointer-events:none; }
  #loading { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:12px; background:var(--bg); z-index:10; }

  /* reserve vertical room so the side panels never run under the bottom bar */
  #panel, #legend-panel { max-height: calc(100vh - 20px - 150px) !important; }

  /* ---- timelapse / simulation bottom bar ---- */
  #sim-panel {
    position:absolute; bottom:10px; left:10px; right:10px;
    background:var(--panel); border:1px solid var(--panel-border); border-radius:2px;
    padding:10px 14px; color:var(--text); font-size:11.5px;
  }
  #sim-panel.collapsed #sim-body { display:none; }
  #sim-header { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  #sim-header h1 { font-size:12px; font-weight:bold; margin:0; }
  #sim-body { margin-top:8px; }
  #sim-controls { display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap; }
  #sim-play, #sim-record {
    background:#d4d4d2; border:1px solid #999; border-radius:2px; color:var(--text);
    padding:5px 12px; cursor:pointer; font-size:11.5px; font-family:inherit;
  }
  #sim-play:hover, #sim-record:hover { background:#c8c8c6; }
  #sim-record.recording { border-color:#a02020; color:#a02020; font-weight:bold; }
  #sim-scrub { flex:1; min-width:110px; accent-color:var(--accent); }
  #sim-readout { font-family:var(--mono); font-size:11px; color:#333; display:flex; gap:14px; flex-wrap:wrap; }
  #sim-readout b { color:var(--text); }
  #sim-caveat { font-size:10px; color:#777; margin-top:6px; line-height:1.5; }
  #sim-legend-row { display:flex; gap:14px; margin-top:6px; font-size:10.5px; color:var(--text-dim); flex-wrap:wrap; }
  .sim-swatch { display:inline-block; width:10px; height:10px; border:1px solid #888; margin-right:3px; vertical-align:middle; }
</style>"""

_style_start = html.index("<style>")
_style_end = html.index("</style>") + len("</style>")
html = html[:_style_start] + FULL_CSS + html[_style_end:]

with open(OUT_PATH, "w") as f:
    f.write(html)

print("wrote", OUT_PATH, len(html) / 1e6, "MB")
