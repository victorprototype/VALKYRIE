# =============================================================================
# VALKYRIE | paths.py
# Stage   : Setup (shared by every script)
# Purpose : The ONE place that says where input data is read from and where
#           results are written, so no script contains a hard-coded path.
# =============================================================================
"""Locations of the input data and of the generated outputs.

Defaults are relative to the repository root, so everything works no matter
which folder you run Python from:

    data/      input rasters (see data/README.md for the file list)
    outputs/   everything the scripts generate

To use other folders (for example a different disk or a notebook platform's
input directory), set the environment variables VALKYRIE_DATA_DIR and/or
VALKYRIE_OUTPUT_DIR before running.
"""
import os

# Repository root = the folder that contains src/ (this file lives in src/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("VALKYRIE_DATA_DIR", os.path.join(REPO_ROOT, "data"))
OUTPUT_DIR = os.environ.get("VALKYRIE_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))

# Make sure the output folder exists (git does not keep empty folders).
os.makedirs(OUTPUT_DIR, exist_ok=True)


def data_file(name):
    """Full path of an input file inside the data folder."""
    return os.path.join(DATA_DIR, name)


def output_file(name):
    """Full path of a generated file inside the outputs folder."""
    return os.path.join(OUTPUT_DIR, name)
