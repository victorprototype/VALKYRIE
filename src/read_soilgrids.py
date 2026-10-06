# =============================================================================
# VALKYRIE | read_soilgrids.py
# Stage   : A. Data readers (low level)
# Purpose : Reads the SoilGrids GeoTIFF (tiled, LZW-compressed, 6 bands) into a
#           NumPy array using our own decoder.
# Reads   : soilgrids_mandakini.tif
# Writes  : array of shape (rows, cols, 6)
# Used by : resample_to_grid.py
# =============================================================================
"""Reader for the tiled, LZW-compressed SoilGrids GeoTIFF.

Band order in the file: sand 0-5cm, sand 5-15cm, clay 0-5cm, clay 5-15cm,
bulk density 0-5cm, bulk density 5-15cm.
"""

import numpy as np
import tifffile
from lzw_decode import lzw_decode


def read_tiled_lzw_tiff(path):
    """Decode a tiled LZW GeoTIFF into a (rows, cols, bands) array.

    A tiled TIFF stores the picture as many small squares (tiles), each
    compressed separately. We decompress every tile and paste it into place.
    Assumes no TIFF 'predictor' (none was present in this file).
    """
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        # Image size (rows, columns, bands) and the size of one tile.
        H, W, S = page.shape
        tw, tl = page.tags["TileWidth"].value, page.tags["TileLength"].value
        dtype = page.dtype
        bytes_per_px = dtype.itemsize * S

        tiles_across = -(-W // tw)  # ceil
        tiles_down = -(-H // tl)

        # Empty full-size image; decoded tiles are pasted into it below.
        out = np.zeros((H, W, S), dtype=dtype)
        fh = page.parent.filehandle
        offsets = page.tags["TileOffsets"].value
        counts = page.tags["TileByteCounts"].value

        # Visit tiles left-to-right, top-to-bottom (the order TIFF stores them in).
        idx = 0
        for ty in range(tiles_down):
            for tx in range(tiles_across):
                fh.seek(offsets[idx])
                raw = fh.read(counts[idx])
                # Decompress this tile, then reshape the flat bytes into (tile rows, tile cols, bands).
                decoded = lzw_decode(raw)
                tile = np.frombuffer(decoded[:tw * tl * bytes_per_px], dtype=dtype).reshape(tl, tw, S)
                # Tiles on the right/bottom edge can overhang the image, so crop them to fit.
                row0, col0 = ty * tl, tx * tw
                row1, col1 = min(row0 + tl, H), min(col0 + tw, W)
                out[row0:row1, col0:col1, :] = tile[: row1 - row0, : col1 - col0, :]
                idx += 1
        return out


# Manual check: prints min/max/mean of each band and how many pixels are no-data.
# Uses data/soilgrids_mandakini.tif.
if __name__ == "__main__":
    from paths import data_file
    arr = read_tiled_lzw_tiff(data_file("soilgrids_mandakini.tif"))
    print("shape", arr.shape, arr.dtype)
    names = ["sand_0-5cm", "sand_5-15cm", "clay_0-5cm", "clay_5-15cm", "bdod_0-5cm", "bdod_5-15cm"]
    for b, name in enumerate(names):
        band = arr[:, :, b]
        nodata = (band == -32768).sum()
        valid = band[band != -32768]
        print(f"{name:14s} min={valid.min():6d} max={valid.max():6d} mean={valid.mean():8.1f} nodata={nodata}")
