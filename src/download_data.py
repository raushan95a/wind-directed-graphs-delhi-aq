"""Step 0: download Delhi rows of the CPCB archive (Vonter/india-cpcb-aqi, ODbL-1.0) for 2019-2024.

Reads only the Delhi row groups and needed columns from the remote yearly parquet files via HTTP range
requests (~75 MB total). Falls back to downloading the full yearly file if range reads fail.
Usage: python download_data.py <out_dir> 2019 2020 ... 2024
"""
import sys, os, subprocess
import fsspec, pyarrow.parquet as pq, pyarrow.compute as pc

outdir = sys.argv[1]; years = [int(y) for y in sys.argv[2:]]; os.makedirs(outdir, exist_ok=True)
WANT = ("Station ID", "State", "City", "Station Name", "Timestamp", "PM2.5", "PM10", "AT", "RH", "WS", "WD")

# Keep only the row groups whose 'State' range can contain Delhi, read just the needed
# columns, then filter to Delhi rows and drop the unit suffixes from column names.
def extract(pf):
    names = pf.schema_arrow.names
    want = [n for n in names if n.split(" (")[0] in WANT]
    si = names.index("State"); md = pf.metadata
    rgs = [i for i in range(md.num_row_groups)
           if (lambda st: st is None or not st.has_min_max or (st.min <= "Delhi" <= st.max))(md.row_group(i).column(si).statistics)]
    t = pf.read_row_groups(rgs, columns=want)
    t = t.filter(pc.equal(t["State"], "Delhi"))
    return t.rename_columns([n.split(" (")[0] for n in t.column_names])

# One remote parquet file per year. Range reads avoid downloading the full ~0.3-0.8 GB file;
# if the server refuses range requests we fall back to a full download.
for y in years:
    dst = os.path.join(outdir, f"delhi_{y}.parquet")
    if os.path.exists(dst): print(y, "already present"); continue
    url = f"https://github.com/Vonter/india-cpcb-aqi/releases/download/{y}/cpcb-air-quality-{y}.parquet"
    try:
        fs = fsspec.filesystem("http")
        with fs.open(url, "rb", block_size=4 * 2 ** 20) as f:
            t = extract(pq.ParquetFile(f))
    except Exception as e:
        print(y, "range read failed (", e, ") -> full download", flush=True)
        tmp = f"/tmp/cpcb_{y}.parquet"
        subprocess.run(["wget", "-q", "-O", tmp, url], check=True)
        t = extract(pq.ParquetFile(tmp)); os.remove(tmp)
    pq.write_table(t, dst)
    print(y, "rows", t.num_rows, "stations", len(pc.unique(t["Station ID"])), flush=True)
