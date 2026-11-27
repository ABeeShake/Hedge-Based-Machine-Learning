"""tools/parquet_pure.py -- Dependency-free reader for flat, Snappy- or un-compressed Parquet files.

Fallback for machines without pyarrow/fastparquet (used by prepare_parquet_cohorts.py --reader pure).
Supports the subset of Parquet written by pandas/pyarrow for flat tables: data page v1, PLAIN and
RLE_DICTIONARY encodings, optional (definition level 0/1) columns, physical types BOOLEAN, INT32,
INT64, FLOAT, DOUBLE and BYTE_ARRAY (UTF-8). Anything else raises NotImplementedError.
"""
import struct


class R:
    def __init__(s,b): s.b=b; s.i=0
    def byte(s): v=s.b[s.i]; s.i+=1; return v
    def varint(s):
        r=sh=0
        while True:
            x=s.byte(); r|=(x&0x7f)<<sh; sh+=7
            if not x&0x80: return r
    def zz(s): n=s.varint(); return (n>>1)^-(n&1)
    def binary(s): n=s.varint(); v=s.b[s.i:s.i+n]; s.i+=n; return v
    def val(s,t):
        if t==1: return True
        if t==2: return False
        if t==3: return s.byte()
        if t in (4,5,6): return s.zz()
        if t==7: v=struct.unpack('<d',s.b[s.i:s.i+8])[0]; s.i+=8; return v
        if t==8: return s.binary()
        if t in (9,10):
            h=s.byte(); n=h>>4; et=h&0xf
            if n==15: n=s.varint()
            return [s.val(et) for _ in range(n)]
        if t==12: return s.struct()
        raise ValueError(t)
    def struct(s):
        d={}; last=0
        while True:
            h=s.byte()
            if h==0: return d
            t=h&0xf; delta=h>>4
            fid = last+delta if delta else s.zz()
            last=fid
            d[fid]= (t==1) if t in (1,2) else s.val(t)


def _snappy(b):
    i = n = sh = 0
    while True:
        x = b[i]; i += 1; n |= (x & 0x7f) << sh; sh += 7
        if not x & 0x80:
            break
    out = bytearray()
    while i < len(b):
        t = b[i]; i += 1; k = t & 3
        if k == 0:
            ln = t >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(b[i:i + nb], "little"); i += nb
            ln += 1; out += b[i:i + ln]; i += ln
            continue
        if k == 1:
            ln = ((t >> 2) & 7) + 4; off = ((t >> 5) << 8) | b[i]; i += 1
        elif k == 2:
            ln = (t >> 2) + 1; off = int.from_bytes(b[i:i + 2], "little"); i += 2
        else:
            ln = (t >> 2) + 1; off = int.from_bytes(b[i:i + 4], "little"); i += 4
        s = len(out) - off
        if off >= ln:
            out += out[s:s + ln]
        else:
            for j in range(ln):
                out.append(out[s + j])
    if len(out) != n:
        raise ValueError("snappy: length mismatch")
    return bytes(out)


def _uvar(b, i):
    r = sh = 0
    while True:
        x = b[i]; i += 1; r |= (x & 0x7f) << sh; sh += 7
        if not x & 0x80:
            return r, i


def _rle_bp(b, i, end, bw, count):
    out = []
    while i < end and len(out) < count:
        h, i = _uvar(b, i)
        if h & 1:
            ng = h >> 1; nbytes = ng * bw
            bits = int.from_bytes(b[i:i + nbytes], "little"); i += nbytes; mask = (1 << bw) - 1
            out.extend((bits >> (j * bw)) & mask for j in range(ng * 8))
        else:
            w = (bw + 7) // 8; v = int.from_bytes(b[i:i + w], "little"); i += w
            out.extend([v] * (h >> 1))
    return out[:count], i


_TYPES = {0: "BOOLEAN", 1: "INT32", 2: "INT64", 4: "FLOAT", 5: "DOUBLE", 6: "BYTE_ARRAY"}


def _plain(b, i, n, typ):
    if typ == "DOUBLE":
        return list(struct.unpack_from("<%dd" % n, b, i))
    if typ == "FLOAT":
        return list(struct.unpack_from("<%df" % n, b, i))
    if typ == "INT64":
        return list(struct.unpack_from("<%dq" % n, b, i))
    if typ == "INT32":
        return list(struct.unpack_from("<%di" % n, b, i))
    if typ == "BOOLEAN":
        bits = int.from_bytes(b[i:i + (n + 7) // 8], "little")
        return [bool((bits >> j) & 1) for j in range(n)]
    if typ == "BYTE_ARRAY":
        out = []
        for _ in range(n):
            ln = struct.unpack_from("<I", b, i)[0]; i += 4
            out.append(b[i:i + ln].decode("utf-8")); i += ln
        return out
    raise NotImplementedError(typ)


def read_table(path, columns=None):
    """Return {column_name: list} for the requested top-level columns (None -> all)."""
    with open(path, "rb") as fh:
        f = fh.read()
    if f[:4] != b"PAR1" or f[-4:] != b"PAR1":
        raise ValueError(f"{path} is not a Parquet file")
    n = struct.unpack("<I", f[-8:-4])[0]
    md = R(f[-8 - n:-8]).struct()
    schema = md[2][1:]
    names = [(e[4].decode(), _TYPES.get(e.get(1)), e.get(3, 1)) for e in schema]
    want = [nm for nm, _, _ in names] if columns is None else list(columns)
    out = {}
    for ci, (nm, typ, rep) in enumerate(names):
        if nm not in want:
            continue
        vals = []
        for rg in md[4]:
            cc = rg[1][ci][3]
            if cc[4] not in (0, 1):
                raise NotImplementedError(f"codec {cc[4]}")
            pos = cc[11] if cc.get(11) is not None and cc[11] < cc[9] else cc[9]
            end = pos + cc[7]; nvals = cc[5]; got = 0; dictionary = None
            while pos < end and got < nvals:
                r = R(f[pos:pos + 4096]); ph = r.struct(); hl = r.i
                raw = f[pos + hl:pos + hl + ph[3]]; pos += hl + ph[3]
                data = _snappy(raw) if cc[4] == 1 else raw
                if ph[1] == 2:
                    dictionary = _plain(data, 0, ph[7][1], typ); continue
                if ph[1] != 0:
                    raise NotImplementedError("only data page v1 is supported")
                dph = ph[5]; nv = dph[1]; enc = dph[2]; i = 0
                if rep == 1:  # OPTIONAL
                    ln = struct.unpack("<I", data[0:4])[0]
                    defs, _ = _rle_bp(data, 4, 4 + ln, 1, nv); i = 4 + ln
                else:
                    defs = [1] * nv
                nn = sum(defs)
                if enc in (2, 8):
                    bw = data[i]; i += 1
                    idx, _ = _rle_bp(data, i, len(data), bw, nn) if bw else ([0] * nn, i)
                    vv = [dictionary[k] for k in idx]
                elif enc == 0:
                    vv = _plain(data, i, nn, typ)
                else:
                    raise NotImplementedError(f"encoding {enc}")
                it = iter(vv)
                vals.extend(next(it) if d else None for d in defs)
                got += nv
        out[nm] = vals
    missing = set(want) - set(out)
    if missing:
        raise KeyError(f"columns not found: {sorted(missing)}")
    return out
