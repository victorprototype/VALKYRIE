# VALKYRIE

**V**isualization **A**nalytics of **L**andslide **K**inetics and **Y**ield point with **R**isk **I**ndication and **E**valuation

> Watch a mountain basin's slopes weaken, fail and flow during a 24-hour storm, in 3D, in your browser, in 30 seconds.

---

## The idea in one minute

Heavy rain triggers landslides across the Himalaya. Rain soaks into the soil, water pressure builds up underground, the soil loses its grip, and a slope that stood for centuries lets go. It is hard to picture, and even harder to show to people who are not engineers.

VALKYRIE turns three public datasets (terrain, soil and satellite rainfall) into an **animated 3D replay of a single storm** over the **Mandakini river basin, Uttarakhand, India**, using rainfall from 17 June 2013 (the Uttarakhand / Kedarnath floods).

## What you see

Press **Play**. Twenty-four hours of storm play out in 30 seconds: rain falls, and the ground changes colour.

| Colour | Meaning |
|---|---|
| 🟢 Green | Stable (Factor of Safety above 2) |
| 🟡 Yellow | Marginal, close to the limit (around 1.2) |
| 🔴 Red | Failing (below 1) |
| 🟤 Brown | Moving debris from slopes that failed |

**Factor of Safety (FS)** is how many times stronger a slope is than the pull trying to move it. Above 1 it holds. Below 1 it slips.

You can also drag to orbit, switch map layers (slope, wetness, rainfall, soil), scrub through time, and record the replay as a video clip. The result is **one HTML file**: no server, nothing to install to view it (a modern browser such as Chrome or Edge is recommended).

## How it works

```mermaid
flowchart LR
  A["Terrain, soil and rainfall data"] --> B["TRIGRS: rain to water pressure to Factor of Safety, every hour"]
  B --> C["Voellmy: where does the failed soil flow?"]
  C --> D["Interactive 3D replay"]
```

Everything is computed at **169,420 points** across the basin, for each of 24 hours.

### TRIGRS: will the slope hold?

**In plain words.** As rain seeps down, water pressure inside the soil rises and pushes the grains apart, so the slope loses grip. TRIGRS (a USGS landslide model) calculates how that pressure grows hour by hour, then asks at every point: *is the slope still strong enough?*

**How we used it.** We implemented the published TRIGRS equations directly in Python (rain infiltration after Iverson 2000, and the infinite-slope Factor of Safety from the TRIGRS manual). We did not use the TRIGRS program itself. Inputs at each point are the local slope, estimated soil strength and permeability, and that day's rainfall.

**Nuances worth knowing**
- TRIGRS needs the **starting water-table depth**. We have no measurement, so we assumed 2 m (plus a small background seepage). TRIGRS' own manual warns that results are very sensitive to this and that, without field data, it should only be used for *hypothetical scenarios*. That is exactly what this is.
- **Soil strength and permeability are estimates**, derived from sand, clay and density maps using generic formulas, not lab tests.
- **Rain timing is synthetic.** The satellite data gives one daily total at roughly 5 km resolution. We spread it over 24 hours with a bell-shaped storm peaking mid-afternoon, because the real hour-by-hour pattern is unknown.
- Every failure is assumed to be a shallow slide of the same depth (1.5 m).

### Voellmy: where does the failed soil go?

**In plain words.** Once a slope fails, the soil does not simply stop. The Voellmy model, also used in professional avalanche and debris-flow software, treats moving debris like a sliding block with two brakes: **dry friction** (grip on the ground) and **turbulent drag** (churning mud, which grows with speed). Gravity pulls, the brakes resist, and the balance decides how fast and how far it travels.

**How we used it.** A simplified grid version. When a point first drops below FS 1, it releases 1.5 m of soil. Every 30 simulated seconds each pile speeds up or slows down according to the Voellmy balance and passes a share of its depth to its steepest downhill neighbour. Friction comes from the soil's friction angle. The total amount of soil is conserved exactly, and piles deeper than 5 m spill sideways.

**Nuances worth knowing**
- Flow follows a **single downhill path**, so it cannot fan out the way real debris does. The 5 m spill rule is a crude stand-in.
- The friction and drag values are **typical literature values, not calibrated** to this basin. Real saturated debris often behaves differently from static soil.
- Debris does not pick up extra soil on its way down.
- In flat valleys and pits, debris can still pile up deeper than the 5 m spill rule intends (in our run, a small share of debris cells exceed 10 m). The viewer caps the displayed depth at 10 m, so read debris depth as indicative, not as a prediction.
- It is an approximation, not a replacement for full 2D tools such as RAMMS or r.avaflow.

## Please read: honest limits

- This is a **demonstration of a method, not a hazard forecast.** It has not been calibrated or validated against recorded landslides.
- Every assumed value is flagged in the code and on the viewer's panel.
- Percentages shown refer to the whole map, which includes roughly a quarter that lies outside the mapped watershed (default soil values are used there).
- A few constants (grid size, map coordinates) are specific to the Mandakini study area.

## Data

| What | Source | Resolution |
|---|---|---|
| Terrain (DEM) and derived slope, aspect, flow and wetness maps | Copernicus GLO-30 | about 30 m |
| Soil texture and density | SoilGrids 2.0 | 250 m |
| Rainfall | CHIRPS v3.0, daily, 17 June 2013 | about 5.5 km |

The prepared input files (about 170 MB) are too large for git, so they are attached to this repository as a [GitHub Release](https://github.com/victorprototype/VALKYRIE/releases/tag/v1.0.0-dem-sim-data). One command downloads and unpacks them into `data/`. See [`data/README.md`](data/README.md) for the file list.

## What is in this repository

Scripts are grouped by the stage they belong to. Each file starts with a short header saying what it reads, writes and who uses it.

| Stage | File | Role |
|---|---|---|
| **Setup** | `download_data.py` | Downloads the input data from the Release into `data/` |
| | `paths.py` | The one place that defines the `data/` and `outputs/` folders |
| **A. Data** | `lzw_decode.py`, `read_soilgrids.py` | Read the compressed SoilGrids file without extra libraries |
| | `resample_to_grid.py` | Line soil and rainfall maps up with the terrain grid |
| | `soil_ptf.py` | Turn sand / clay / density into soil strength and permeability |
| **B. Graph** | `build_graph.py`, `build_graph_v2.py` | Terrain (and optionally soil and rain) as a graph of points and links |
| **C. Physics** | `trigrs_physics.py` | TRIGRS: water pressure and Factor of Safety |
| | `voellmy_runout.py` | Voellmy: debris flow after failure |
| | `run_simulation.py` | Runs the 24-hour simulation end to end |
| **D. Viewer** | `build_dem_3d_viewer.py` | Builds the base 3D terrain viewer |
| | `pack_simulation_data.py` | Compresses the simulation for the browser |
| | `build_final_with_simulation.py` | Combines both into the final animated viewer |

Folders: `src/` code, `data/` inputs (downloaded), `outputs/` everything the scripts generate.

## Run it (for developers)

Runs on any machine; no cloud account or notebook platform needed.

```bash
git clone https://github.com/victorprototype/VALKYRIE.git
cd VALKYRIE
pip install -r requirements.txt

python src/download_data.py                 # 1. fetch the input data (about 170 MB) into data/
python src/build_dem_3d_viewer.py           # 2. base 3D viewer
python src/run_simulation.py                # 3. TRIGRS + Voellmy simulation
python src/build_final_with_simulation.py   # 4. final viewer
```

Open **`outputs/dem_3d_viewer_simulation.html`** in your browser. Steps 2 to 4 take about a minute in our tests.

- Scripts find their folders on their own, so you can run them from any directory.
- To use other folders, set `VALKYRIE_DATA_DIR` and/or `VALKYRIE_OUTPUT_DIR` first (see `src/paths.py`).
- Optional: `python src/build_graph_v2.py` exports the full terrain + soil + rainfall graph.
- Each physics module has a built-in demo that needs no data: `python src/trigrs_physics.py` and `python src/voellmy_runout.py`.

## References

- Iverson, R. M. (2000). Landslide triggering by rain infiltration. *Water Resources Research*, 36(7).
- Baum, R. L., Savage, W. Z., Godt, J. W. (2008). *TRIGRS v2.0 manual.* USGS Open-File Report 2008-1159.
- Voellmy, A. (1955). Über die Zerstörungskraft von Lawinen. *Schweizerische Bauzeitung*, 73.
- Funk, C. et al. (2015). CHIRPS. *Scientific Data*, 2.
- Poggio, L. et al. (2021). SoilGrids 2.0. *SOIL*, 7.
