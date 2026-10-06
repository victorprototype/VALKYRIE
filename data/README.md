# Input data

The input rasters (about 170 MB) are **not stored in git**. They are attached to this
repository as a GitHub Release, and one command fetches and unpacks them here:

```bash
python src/download_data.py
```

Manual alternative: download
[`dem-sim-data.zip`](https://github.com/victorprototype/VALKYRIE/releases/download/v1.0.0-dem-sim-data/dem-sim-data.zip)
and unzip it so the `.tif` files sit directly inside this `data/` folder.

## What is in the zip

| File(s) | What it is |
|---|---|
| `output_hh.tif` | Elevation (DEM) of the study area |
| `viz.hh_slope.tif`, `viz.hh_aspect.tif`, `viz.hh_roughness.tif` | Slope, aspect and roughness derived from the DEM |
| `viz.hh_hillshade.tif`, `viz.hh_hillshade-color.tif`, `viz.hh_color-relief.tif` | Shaded-relief and colour images used to texture the 3D viewer |
| `TWI.tif`, `D8_Flow_Accumulation.tif`, `DInf_Specific_Catchment_Area.tif` | Hydrology: wetness index, flow accumulation, catchment area |
| `D8_Flow_Direction.tif`, `DInf_Flow_Direction.tif`, `Hydro_Conditioned_DEM.tif` | Optional extra layers for the viewer |
| `soilgrids_mandakini.tif` | SoilGrids 2.0 sand, clay and bulk density (0-5 and 5-15 cm) |
| `chirps-v3.0.sat.2013.06.17.tif` | CHIRPS v3.0 daily rainfall, 17 June 2013 |
| `streams.*` | Stream network shapefile (not used by the current scripts) |

Sources: Copernicus GLO-30 DEM, SoilGrids 2.0 (ISRIC), CHIRPS v3.0 (UCSB Climate Hazards Center).

## Using a different folder

Set `VALKYRIE_DATA_DIR` (and optionally `VALKYRIE_OUTPUT_DIR`) before running any script;
see `src/paths.py`.
