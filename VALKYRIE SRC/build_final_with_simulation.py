import re
from pack_simulation_data import pack_simulation

BASE_HTML_PATH = "/home/claude/physics/base_viewer.html"
OUT_PATH = "/home/claude/physics/dem_3d_viewer_simulation.html"

sim = pack_simulation("/home/claude/physics/simulation_24h.npz")

html = open(BASE_HTML_PATH).read()

# ---------------- CSS ----------------
NEW_CSS = r"""
  #panel, #legend-panel {
    max-height: calc(100vh - 32px - 160px) !important;
  }
  #sim-panel {
    position:absolute; bottom:16px; left:16px; right:290px;
    background:rgba(28,31,36,0.92); backdrop-filter: blur(8px);
    border:1px solid #2a2d33; border-radius:10px;
    padding:14px 18px; color:#e8e5dd; font-size:12.5px;
  }
  #sim-panel.collapsed #sim-body { display:none; }
  #sim-header { display:flex; align-items:center; justify-content:space-between; gap:12px; }
  #sim-header h1 { font-size:13.5px; font-weight:600; margin:0; }
  #sim-toggle {
    width:22px; height:22px; padding:0; background:#22262c; border:1px solid #2a2d33;
    border-radius:6px; color:#e8e5dd; cursor:pointer; display:flex; align-items:center; justify-content:center;
  }
  #sim-body { margin-top:10px; }
  #sim-controls { display:flex; align-items:center; gap:10px; margin-bottom:8px; flex-wrap:wrap; }
  #sim-play, #sim-record {
    background:#22262c; border:1px solid #2a2d33; border-radius:6px; color:#e8e5dd;
    padding:6px 14px; cursor:pointer; font-size:12.5px; font-family:inherit;
  }
  #sim-play:hover, #sim-record:hover { border-color:#c8873f; }
  #sim-record.recording { border-color:#d9534f; color:#f0a8a5; }
  #sim-scrub { flex:1; min-width:120px; accent-color:#c8873f; }
  #sim-readout { font-family:'SF Mono',monospace; font-size:11.5px; color:#c3c6cb; display:flex; gap:16px; flex-wrap:wrap; }
  #sim-readout b { color:#f0d9b8; }
  #sim-caveat { font-size:10.5px; color:#77797e; margin-top:8px; line-height:1.5; font-style:italic; }
  #sim-legend-row { display:flex; gap:18px; margin-top:8px; font-size:11px; color:#9a9da4; flex-wrap:wrap; }
  .sim-swatch { display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:4px; vertical-align:middle; }
"""
html = html.replace("</style>", NEW_CSS + "</style>")

# ---------------- HTML panel ----------------
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

# ---------------- JS ----------------
NEW_JS = r"""
// =========================== 24h SIMULATION LAYER ===========================
const SIM = __SIM_JSON__;
const SIM_B64 = "__SIM_B64__";

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

function simByteAt(hourIdx, nodeIdx, channel) {
  return simPacked[(hourIdx * N_NODES + nodeIdx) * 2 + channel];
}

// overlay mesh sharing the SAME geometry as the terrain, so vertical
// exaggeration changes on the terrain automatically apply here too
const overlayColors = new Float32Array(N_NODES * 3);
if (!geometry.attributes.color) {
  geometry.setAttribute('color', new THREE.BufferAttribute(overlayColors, 3));
}
const overlayMaterial = new THREE.MeshBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.0,
  side: THREE.DoubleSide, depthWrite: false,
});
const overlayMesh = new THREE.Mesh(geometry, overlayMaterial);
overlayMesh.renderOrder = 1;
scene.add(overlayMesh);

function riskColor(fs01, depth01, out) {
  // fs01: 0..1 (0=very unstable, 1=very stable, already normalized)
  // 3-stop gradient: red -> yellow -> green
  let r, g, b;
  if (fs01 < 0.4) {
    const t = fs01 / 0.4;
    r = 0.753 + t * (0.831 - 0.753); g = 0.161 + t * (0.722 - 0.161); b = 0.169 + t * (0.247 - 0.169);
  } else {
    const t = Math.min(1, (fs01 - 0.4) / 0.6);
    r = 0.831 + t * (0.180 - 0.831); g = 0.722 + t * (0.490 - 0.722); b = 0.247 + t * (0.310 - 0.247);
  }
  if (depth01 > 0.02) {
    const dt = Math.min(1, depth01 * 2.0);
    r = r * (1 - dt) + 0.541 * dt;
    g = g * (1 - dt) + 0.353 * dt;
    b = b * (1 - dt) + 0.231 * dt;
  }
  out[0] = r; out[1] = g; out[2] = b;
}

let simPlaying = false;
let simHour = 0.0;   // fractional hour, 0..24
const SIM_DURATION_S = 30.0;
const tmpColor = [0, 0, 0];

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
    riskColor(fs01, depth01, tmpColor);
    colorAttr.setXYZ(i, tmpColor[0], tmpColor[1], tmpColor[2]);
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

document.getElementById('sim-play').addEventListener('click', () => {
  simPlaying = !simPlaying;
  document.getElementById('sim-play').innerHTML = simPlaying ? '&#10074;&#10074; Pause' : '&#9654; Play';
  overlayMaterial.opacity = 0.85;
  if (simPlaying) { simLastT = null; requestAnimationFrame(simTick); }
});
document.getElementById('sim-scrub').addEventListener('input', (e) => {
  simPlaying = false;
  document.getElementById('sim-play').innerHTML = '&#9654; Play';
  overlayMaterial.opacity = 0.85;
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
  simHour = 0; simPlaying = true; simLastT = null; overlayMaterial.opacity = 0.85;
  document.getElementById('sim-play').innerHTML = '&#10074;&#10074; Pause';
  requestAnimationFrame(simTick);
});

applySimFrame(0);
"""

NEW_JS = NEW_JS.replace("__SIM_JSON__", str({
    "n_hours": sim["n_hours"], "n_nodes": sim["n_nodes"],
    "failed_frac_per_hour": sim["failed_frac_per_hour"],
    "hourly_rain_mm": sim["hourly_rain_mm"],
}).replace("'", '"'))
NEW_JS = NEW_JS.replace("__SIM_B64__", sim["b64"])

html = html.replace("</script>", NEW_JS + "\n</script>")

with open(OUT_PATH, "w") as f:
    f.write(html)

print("wrote", OUT_PATH, len(html) / 1e6, "MB")
