"""
Minimal pure-Python TIFF-variant LZW decoder (no imagecodecs dependency).
Implements the classic LZW algorithm with TIFF's "early change" quirk
(TIFF Revision 6.0 spec, Section 13): code width increases one code
early relative to standard LZW/GIF.
"""
import struct


def lzw_decode(data: bytes) -> bytes:
    CLEAR = 256
    EOI = 257
    out = bytearray()

    def reset_table():
        table = [bytes([i]) for i in range(256)] + [b"", b""]  # 256=clear,257=eoi placeholders
        return table

    table = reset_table()
    code_size = 9
    next_code = 258

    bitpos = 0
    nbits = len(data) * 8

    def read_code(width):
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

    prev = None
    while True:
        code = read_code(code_size)
        if code is None or code == EOI:
            break
        if code == CLEAR:
            table = reset_table()
            code_size = 9
            next_code = 258
            prev = None
            continue
        if code < len(table):
            entry = table[code]
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


if __name__ == "__main__":
    import tifffile
    import numpy as np

    path = "/mnt/user-data/uploads/soilgrids_mandakini.tif"
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
