import numpy as np
import tifffile
from lzw_decode import lzw_decode


def read_tiled_lzw_tiff(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        H, W, S = page.shape
        tw, tl = page.tags["TileWidth"].value, page.tags["TileLength"].value
        dtype = page.dtype
        bytes_per_px = dtype.itemsize * S

        tiles_across = -(-W // tw)  # ceil
        tiles_down = -(-H // tl)

        out = np.zeros((H, W, S), dtype=dtype)
        fh = page.parent.filehandle
        offsets = page.tags["TileOffsets"].value
        counts = page.tags["TileByteCounts"].value

        idx = 0
        for ty in range(tiles_down):
            for tx in range(tiles_across):
                fh.seek(offsets[idx])
                raw = fh.read(counts[idx])
                decoded = lzw_decode(raw)
                tile = np.frombuffer(decoded[:tw * tl * bytes_per_px], dtype=dtype).reshape(tl, tw, S)
                row0, col0 = ty * tl, tx * tw
                row1, col1 = min(row0 + tl, H), min(col0 + tw, W)
                out[row0:row1, col0:col1, :] = tile[: row1 - row0, : col1 - col0, :]
                idx += 1
        return out


if __name__ == "__main__":
    arr = read_tiled_lzw_tiff("/mnt/user-data/uploads/soilgrids_mandakini.tif")
    print("shape", arr.shape, arr.dtype)
    names = ["sand_0-5cm", "sand_5-15cm", "clay_0-5cm", "clay_5-15cm", "bdod_0-5cm", "bdod_5-15cm"]
    for b, name in enumerate(names):
        band = arr[:, :, b]
        nodata = (band == -32768).sum()
        valid = band[band != -32768]
        print(f"{name:14s} min={valid.min():6d} max={valid.max():6d} mean={valid.mean():8.1f} nodata={nodata}")
