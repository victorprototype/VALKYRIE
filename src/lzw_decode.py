# =============================================================================
# VALKYRIE | lzw_decode.py
# Stage   : A. Data readers (low level)
# Purpose : Pure-Python decoder for LZW-compressed TIFF data, so the SoilGrids file
#           can be read without the 'imagecodecs' library.
# Reads   : raw compressed bytes of one TIFF strip/tile
# Writes  : decompressed bytes
# Used by : read_soilgrids.py
# =============================================================================

"""
Minimal pure-Python TIFF-variant LZW decoder (no imagecodecs dependency).
Implements the classic LZW algorithm with TIFF's "early change" quirk
(TIFF Revision 6.0 spec, Section 13): code width increases one code
early relative to standard LZW/GIF.
"""
import struct


def lzw_decode(data: bytes) -> bytes:
    """Decompress one block of TIFF-flavoured LZW data.

    LZW works by building a dictionary of byte sequences while reading, so the
    compressed stream only has to say 'use dictionary entry N'. The TIFF variant
    grows its code width (9 -> 12 bits) one code earlier than GIF-style LZW.
    """
    # Two reserved codes: 256 = 'reset the dictionary', 257 = 'end of data'.
    CLEAR = 256
    EOI = 257
    out = bytearray()

    def reset_table():
        """Fresh dictionary: codes 0-255 are the single bytes; 256 and 257 are reserved."""
        table = [bytes([i]) for i in range(256)] + [b"", b""]  # 256=clear,257=eoi placeholders
        return table

    # Decoder state: the dictionary, the current code width in bits (starts at 9),
    # and the number of the next free dictionary slot.
    table = reset_table()
    code_size = 9
    next_code = 258

    bitpos = 0
    nbits = len(data) * 8

    def read_code(width):
        """Read the next `width`-bit code (most significant bit first). Returns None when the data runs out."""
        nonlocal bitpos
        if bitpos + width > nbits:
            return None
        val = 0
        for _ in range(width):
            byte = data[bitpos // 8]
            bit = (byte >> (7 - (bitpos % 8))) & 1
            val = (val << 1) | bit
            bitpos += 1
        return val

    # Main loop. Read a code -> look up its bytes -> write them out -> add
    # (previous bytes + first byte of this entry) to the dictionary as a new entry.
    prev = None
    while True:
        code = read_code(code_size)
        if code is None or code == EOI:
            break
        # CLEAR code: forget everything learned so far and start over.
        if code == CLEAR:
            table = reset_table()
            code_size = 9
            next_code = 258
            prev = None
            continue
        # Normal case: the code is already in the dictionary.
        if code < len(table):
            entry = table[code]
        # Special case: the code refers to the entry being created right now
        # (previous bytes + their own first byte).
        elif code == next_code and prev is not None:
            entry = prev + prev[:1]
        else:
            raise ValueError(f"bad LZW code {code} at bitpos {bitpos}")
        out += entry
        if prev is not None:
            table.append(prev + entry[:1])
            next_code += 1
            # TIFF early-change: bump width one code sooner than standard LZW
            if next_code == 511:
                code_size = 10
            elif next_code == 1023:
                code_size = 11
            elif next_code == 2047:
                code_size = 12
        prev = entry
    return bytes(out)


# Manual self-test only (not part of the pipeline): decodes every strip/tile of a real
# file and prints the total size. Uses data/soilgrids_mandakini.tif. For this tiled
# file the decoded total is larger than rows x cols x bands x 2 bytes, because edge
# tiles are padded; read_soilgrids.py crops that padding off.
if __name__ == "__main__":
    import tifffile
    import numpy as np

    from paths import data_file
    path = data_file("soilgrids_mandakini.tif")
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        print("shape", page.shape, "dtype", page.dtype)
        print("offsets", page.dataoffsets[:3], "... n_strips", len(page.dataoffsets))
        print("bytecounts", page.databytecounts[:3])
        print("predictor", page.predictor)
        print("rowsperstrip", page.rowsperstrip)

        fh = page.parent.filehandle
        strips = []
        for offset, count in zip(page.dataoffsets, page.databytecounts):
            fh.seek(offset)
            raw = fh.read(count)
            decoded = lzw_decode(raw)
            strips.append(decoded)
        all_bytes = b"".join(strips)
        print("total decoded bytes:", len(all_bytes), " expected:", page.shape[0]*page.shape[1]*page.shape[2]*2)
