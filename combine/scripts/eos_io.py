"""
EOS access that does not trust the FUSE mount's directory listings (2026-09-25).

On lxplus929 and lxplus971 the eosxd FUSE mount served stale, EMPTY listings
for directories whose files `eos ls` / xrootd showed fine, and could not open
those files by name either. glob/os.listdir raise nothing, so the datacard
builders silently read a subset (the 09-24 decorr card lost 21% of
Other_Higgs CMS_res_m Up; the 09-25 decorr build lost whole sumw dirs).

Listings here come from the xrootd server (authoritative); reads try FUSE
first and fall back to xrootd. Every function raises on failure.
"""
import io
import os
import json

XRD_URL = "root://eosuser.cern.ch"
_fs = None
N_FALLBACK = {"listdir": 0, "read": 0}


def _filesystem():
    global _fs
    if _fs is None:
        from XRootD import client
        _fs = client.FileSystem(XRD_URL)
    return _fs


def listdir(path):
    """Entry names of an EOS directory from the server (no dot entries such as
    the .sys.v#.* version dirs). Returns [] if the directory does not exist."""
    from XRootD.client.flags import DirListFlags
    st, lst = _filesystem().dirlist(path, DirListFlags.STAT)
    if not st.ok:
        if st.errno == 3011 or "No such file" in st.message or "not found" in st.message.lower():
            return []
        raise OSError(f"xrootd dirlist failed for {path}: {st.message}")
    names = sorted(e.name for e in lst if not e.name.startswith("."))
    try:
        fuse = set(os.listdir(path))
    except OSError:
        fuse = set()
    if fuse != set(names):
        N_FALLBACK["listdir"] += 1
        if N_FALLBACK["listdir"] <= 20:
            print(f"  NOTE eos_io: FUSE listing of {path} differs from server "
                  f"({len(fuse)} vs {len(names)} entries); using server listing", flush=True)
    return names


def isdir(path):
    st, info = _filesystem().stat(path)
    if not st.ok:
        return False
    from XRootD.client.flags import StatInfoFlags
    return bool(info.flags & StatInfoFlags.IS_DIR)


def files(dirpath, suffix):
    """Full paths of the files in dirpath ending with suffix (server listing)."""
    return [os.path.join(dirpath, n) for n in listdir(dirpath) if n.endswith(suffix)]


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        st, info = _filesystem().stat(path)
        if not st.ok:
            raise OSError(f"xrootd stat failed for {path}: {st.message}")
        return float(info.modtime)


def read_bytes(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        from XRootD import client
        N_FALLBACK["read"] += 1
        with client.File() as f:
            st, _ = f.open(f"{XRD_URL}/{path}")
            if not st.ok:
                raise OSError(f"xrootd open failed for {path}: {st.message}")
            st, data = f.read()
            if not st.ok:
                raise OSError(f"xrootd read failed for {path}: {st.message}")
        return data


def read_json(path):
    return json.loads(read_bytes(path))


def _parquet_source(path):
    if os.path.exists(path):
        return path
    return io.BytesIO(read_bytes(path))


def read_parquet(path, columns=None):
    """pyarrow Table; columns missing from the file are silently dropped from
    the request (same as the builders' previous 'available columns' fallback)."""
    import pyarrow.parquet as pq
    src = _parquet_source(path)
    pf = pq.ParquetFile(src)
    if columns is not None:
        avail = set(pf.schema_arrow.names)
        columns = [c for c in columns if c in avail]
    return pf.read(columns=columns)


def parquet_columns(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(_parquet_source(path)).schema_arrow.names
