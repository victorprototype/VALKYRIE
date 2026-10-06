# =============================================================================
# VALKYRIE | download_data.py
# Stage   : Setup (run once, before everything else)
# Purpose : Downloads the project's input data (about 170 MB) from the GitHub
#           Release and unpacks it into the data/ folder.
# Reads   : the Release asset dem-sim-data.zip (over the internet)
# Writes  : data/*.tif (and a few small helper files)
# Used by : you, once. Every other script then reads from data/.
# =============================================================================
"""Download and unpack the VALKYRIE input data.

The rasters are too large to keep inside the git repository, so they are
published as a zip file attached to a GitHub Release. This script fetches that
zip, unpacks it into the data folder (see paths.py) and checks that every file
the pipeline needs is present.

Usage:
    python src/download_data.py            # download only if something is missing
    python src/download_data.py --force    # download again and overwrite

Uses only the Python standard library, so there is nothing extra to install.
"""
import os
import sys
import zipfile
import urllib.request

from paths import DATA_DIR, data_file

DATA_URL = ("https://github.com/victorprototype/VALKYRIE/releases/download/"
            "v1.0.0-dem-sim-data/dem-sim-data.zip")

# Files the pipeline cannot run without. The zip also contains a few optional
# extras (flow-direction rasters, streams shapefile) that are unpacked as well.
REQUIRED_FILES = [
    # terrain + derived rasters (build_graph.py and the 3D viewer)
    "output_hh.tif", "viz.hh_slope.tif", "viz.hh_aspect.tif", "viz.hh_roughness.tif",
    "viz.hh_hillshade.tif", "viz.hh_hillshade-color.tif", "viz.hh_color-relief.tif",
    # hydrology rasters (graph features)
    "TWI.tif", "D8_Flow_Accumulation.tif", "DInf_Specific_Catchment_Area.tif",
    # soil and rainfall inputs of the simulation
    "soilgrids_mandakini.tif", "chirps-v3.0.sat.2013.06.17.tif",
]


def missing_files():
    """Names of required files that are not yet in the data folder."""
    return [f for f in REQUIRED_FILES if not os.path.isfile(data_file(f))]


def _progress(block_count, block_size, total_size):
    """Print a simple one-line download progress indicator."""
    if total_size > 0:
        done = min(block_count * block_size, total_size)
        sys.stdout.write(f"\r  downloading... {done / 1e6:6.1f} / {total_size / 1e6:.1f} MB")
        sys.stdout.flush()


def download_and_unpack():
    """Fetch the zip, unpack it into DATA_DIR, then delete the zip."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "dem-sim-data.zip")
    print(f"Downloading input data to {DATA_DIR}")
    urllib.request.urlretrieve(DATA_URL, zip_path, _progress)
    print()
    print("  unpacking...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)


def main():
    """Download only if needed (or if --force is given), then verify the result."""
    force = "--force" in sys.argv
    if not force and not missing_files():
        print(f"All required data files are already in {DATA_DIR} - nothing to do.")
        return 0

    download_and_unpack()

    still_missing = missing_files()
    if still_missing:
        print("ERROR: these required files are still missing after unpacking:")
        for f in still_missing:
            print("  -", f)
        return 1
    print(f"Done. All {len(REQUIRED_FILES)} required files are in {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
