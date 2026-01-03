"""chrom_io.py – Creating AnnDataModule from bigwigs."""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pybigtools
from tqdm import tqdm
import xarray as xr
import zarr
import re

from .grandata import GRAnData
from collections import defaultdict
try:
    import sparse
except:
    print('no sparse')

# ——————————————————————————————————————————————————————————
# low-level utilities
# ——————————————————————————————————————————————————————————

def _make_temp_bed(peaks: pd.DataFrame, width: int) -> str:
    # assume peaks has 'chrom','start','end'
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".bed")
    half = (width - (peaks.end - peaks.start)) // 2
    starts = (peaks.start - half).clip(lower=0)
    ends   = starts + width
    for c,s,e in zip(peaks.chrom, starts, ends):
        tmp.write(f"{c}\t{int(s)}\t{int(e)}\n")
    tmp.close()
    return tmp.name

def _extract_bw(
    bw_path: Path,
    bed_path: str | None,
    bin_stat: str,
    n_bins: int,
    fill_value: float = 0.0,
    intervals: list[tuple[str, int, int]] | None = None,
) -> np.ndarray:
    with pybigtools.open(str(bw_path), "r") as bw:
        if intervals is None:
            if bed_path is None:
                raise ValueError("bed_path or intervals must be provided.")
            lines = open(bed_path).read().splitlines()
            intervals = [(c, int(s), int(e)) for c, s, e in (l.split("\t") for l in lines)]
        n = len(intervals)
        arr = np.full((n, n_bins), fill_value, dtype="float32")
        for i, (c, s, e) in enumerate(intervals):
            try:
                vals = bw.values(c, int(s), int(e), missing=fill_value,
                                 exact=False, bins=n_bins, summary=bin_stat)
                arr[i] = vals
            except KeyError:
                pass
        return arr.astype("float32")

def grandata_from_bigwigs(
    region_table: pd.DataFrame,
    bigwig_dir: Path|str,
    backed_path: Path|str,
    target_region_width: int,
    *,
    array_name: str = 'X',
    obs_dim: str = 'obs',
    var_dim: str = 'var',
    seq_dim: str = 'seq_bins',
    bin_stat: str = "mean",
    chunk_size: int = 512,
    n_bins: int = 1,
    backend: str = "zarr",
    tile_size: int = 5000,
    fill_value: float = 0.,
    obs_chunk_size: int | None = None,
    n_workers: int = 1,
) -> GRAnData:
    """
    Create a new GRAnData from region_table and a folder of BigWigs.
    The resulting store is written to `backed_path`, and a first array
    named `array_name` is streamed in along dims (obs_dim, var_dim, seq_dim).
    """
    bigwig_dir = Path(bigwig_dir)

    # 1) scan BigWigs and collect chromosomes
    bw_files = []
    chroms = set()
    for p in bigwig_dir.iterdir():
        try:
            with pybigtools.open(str(p), "r") as bw:
                chroms |= set(bw.chroms().keys())
            bw_files.append(p)
        except Exception:
            continue
    if not bw_files:
        raise FileNotFoundError(f"No BigWigs in {bigwig_dir!r}")
    bw_files.sort()

    # 2) filter peaks to those chroms
    peaks = region_table.query("chrom in @chroms").reset_index(drop=True)
    peaks["region"] = (
        peaks.chrom.astype(str)
        + ":"
        + peaks.start.astype(int).astype(str)
        + "-"
        + peaks.end.astype(int).astype(str)
    )

    # 3) build obs- and var- coordinate arrays
    obs_idx = [p.stem.replace(".", "_") for p in bw_files]
    var_idx = peaks.region.values

    extra = {}
    extra[f"{obs_dim}-_-index"] = xr.DataArray(obs_idx, dims=[obs_dim])
    extra[f"{var_dim}-_-index"] = xr.DataArray(var_idx, dims=[var_dim])

    # var‐metadata too
    for col in ("chrom","start","end","region"):
        extra[f"{var_dim}-_-{col}"] = (
            xr.DataArray(peaks[col].values, dims=[var_dim])
              .chunk({var_dim: chunk_size})
        )

    adata = GRAnData(**extra)
    adata.attrs['chunk_size'] = chunk_size
    adata.attrs['target_region_width'] = target_region_width
    adata.attrs['bin_stat'] = bin_stat
    adata.attrs['n_bins'] = n_bins
    adata.attrs['obs_dim'] = obs_dim
    adata.attrs['var_dim'] = var_dim
    adata.attrs['seq_dim'] = seq_dim

    # 4) initialize the on‐disk store
    if backend == "icechunk":
        adata.to_icechunk(str(backed_path), mode="w")
    else:
        adata.to_zarr(str(backed_path), mode="w")
        adata = GRAnData.open_zarr(str(backed_path))

    # 5) stream in the very first array
    return add_bigwig_array(
        adata,
        region_table=peaks,
        bigwig_dir=bw_files,
        array_name=array_name,
        obs_dim=obs_dim,
        var_dim=var_dim,
        seq_dim=seq_dim,
        bin_stat=bin_stat,
        target_region_width=target_region_width,
        chunk_size=chunk_size,
        n_bins=n_bins,
        backend=backend,
        tile_size=tile_size,
        fill_value=fill_value,
        obs_chunk_size=obs_chunk_size,
        n_workers=n_workers,
    )

# ——————————————————————————————————————————————————————————
# public API #2: append a new array from bigwigs into an existing GRAnData
# ——————————————————————————————————————————————————————————

def add_bigwig_array(
    adata: GRAnData,
    region_table: pd.DataFrame,
    bigwig_dir: Path|str|list[Path],
    *,
    array_name: str,
    obs_dim: str,
    var_dim: str,
    seq_dim: str,
    target_region_width: int,
    bin_stat: str = "mean",
    chunk_size: int = 512,
    n_bins: int = 1,
    backend: str = "zarr",
    tile_size: int = 5000,
    fill_value = np.nan,
    obs_chunk_size: int | None = None,
    n_workers: int = 1,
) -> GRAnData:
    # allow passing either a folder or pre-collected list
    if not isinstance(bigwig_dir, (list,tuple)):
        bigwig_dir = list(Path(bigwig_dir).iterdir())
    bw_files = sorted(Path(p) for p in bigwig_dir)

    n_obs = len(adata[f"{obs_dim}-_-index"].values)#len(bw_files)
    n_var = len(region_table)

    obs_names = adata[f"{obs_dim}-_-index"].values.astype(str).tolist()
    # if your obs‐index was created from p.stem.replace('.','_'), do the same here
    file_map = {
        p.stem.replace('.', '_'): p
        for p in bw_files
    }
    
    # 1) determine seq_len directly
    seq_len = n_bins
    if obs_chunk_size is None:
        obs_chunk_size = min(16, n_obs)

    # 2) create empty Zarr/Icechunk backing without allocating full array in RAM
    if backend=="icechunk":
        adata.session = adata.repo.writable_session("main")
        grp = zarr.open_group(adata.session.store, mode="a")
        if array_name not in grp:
            arr = grp.create(
                name=array_name,
                shape=(n_obs, n_var, seq_len),
                chunks=(obs_chunk_size, chunk_size, seq_len),
                dtype="float32",
                fill_value=fill_value
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = [obs_dim, var_dim, seq_dim]
        else:
            arr = grp[array_name]
    else:
        path = adata.encoding["source"]
        adata.to_zarr(path, mode="a")
        store = zarr.open(path, mode="a")
        if array_name not in store:
            arr = store.create_dataset(
                array_name,
                shape=(n_obs, n_var, seq_len),
                chunks=(obs_chunk_size, chunk_size, seq_len),
                dtype="float32",
                fill_value=fill_value
            )
        else:
            arr = store[array_name]

    # 3) tile through var-dimension
    for v0 in tqdm(range(0, n_var, tile_size), desc=f"writing {array_name}"):
        v1 = min(v0 + tile_size, n_var)
        tile = region_table.iloc[v0:v1]
        tmpb = _make_temp_bed(tile, target_region_width)

        with open(tmpb) as bed_f:
            intervals = [(c, int(s), int(e)) for c, s, e in (l.split("\t") for l in bed_f)]
        # for each obs slot, see if we have a matching bigwig
        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {}
                for i, name in enumerate(obs_names):
                    bw_path = file_map.get(name)
                    if bw_path is None:
                        continue
                    futures[ex.submit(
                        _extract_bw, bw_path, None, bin_stat, n_bins,
                        fill_value=fill_value, intervals=intervals
                    )] = i
                for fut in as_completed(futures):
                    i = futures[fut]
                    out = fut.result()
                    if out.ndim == 1:
                        out = out.reshape((v1 - v0, seq_len))
                    arr[i, v0:v1, :] = out
        else:
            for i, name in enumerate(obs_names):
                bw_path = file_map.get(name)
                if bw_path is None:
                    # no file for this obs → leave as fill_value
                    continue
                out = _extract_bw(
                    bw_path, None, bin_stat, n_bins,
                    fill_value=fill_value, intervals=intervals
                )
                if out.ndim == 1:
                    out = out.reshape((v1 - v0, seq_len))
                arr[i, v0:v1, :] = out
        os.remove(tmpb)

        if backend=="icechunk":
            # commit each tile (optional)
            # adata.to_icechunk(mode="a", commit_name=f"{array_name}_{v0}_{v1}")
            pass

    if backend=="icechunk":
        adata.session = adata.repo.readonly_session("main")
    else:
        # reopen to refresh xarray coords / dims
        adata = GRAnData.open_zarr(path)

    return adata

# -----------------------
# Additional utility functions
# -----------------------
def prepare_intervals(
    adata: GRAnData,
    var_dim: str,
    *,
    chrom_suffix: str = "chrom",
    start_suffix: str = "start",
    end_suffix:   str = "end",
    index_suffix: str = "index",
) -> dict[str, list[tuple[int,int,str]]]:
    """
    From adata, pull out the columns
      {var_dim}-_-{chrom,start,end,index}
    and return {chrom: [(start,end,var_name),…] sorted by start}.
    """
    chrom_col = f"{var_dim}-_-{chrom_suffix}"
    start_col = f"{var_dim}-_-{start_suffix}"
    end_col   = f"{var_dim}-_-{end_suffix}"
    idx_col   = f"{var_dim}-_-{index_suffix}"

    df = pd.DataFrame({
        "chrom": adata[chrom_col].values.astype(str),
        "start": adata[start_col].values.astype(int),
        "end":   adata[end_col].values.astype(int),
    }, index=adata[idx_col].values)
    df = df.sort_values(["chrom","start"])

    out: dict[str, list[tuple[int,int,str]]] = defaultdict(list)
    for var_name, row in df.iterrows():
        out[row["chrom"]].append((row["start"], row["end"], var_name))
    return dict(out)


# ——————————————————————————————————————————————————————————
# 2) overlap‐finding on sorted intervals (unchanged)
# ——————————————————————————————————————————————————————————

def _find_overlaps_in_sorted_bed(bed_df, chrom_intervals):
    row_to_overlaps = defaultdict(list)
    chrom_pos = defaultdict(int)
    for _, row in bed_df.iterrows():
        chrom     = row["chrom"]
        start_q   = row["start"]
        end_q     = row["end"]
        row_idx   = row["row_idx"]
        intervals = chrom_intervals.get(chrom, [])
        pos       = chrom_pos.get(chrom, 0)
        # skip intervals ending before query
        while pos < len(intervals) and intervals[pos][1] < start_q:
            pos += 1
        scan = pos
        # collect overlaps
        while scan < len(intervals) and intervals[scan][0] < end_q:
            row_to_overlaps[row_idx].append(intervals[scan][2])
            scan += 1
        chrom_pos[chrom] = scan
    return row_to_overlaps

def _find_overlaps_for_bedp(bedp_df, chrom_intervals, coord_col_prefix):
    # identical to before — can stay unchanged
    df = bedp_df[[
        f"{coord_col_prefix}", 
        f"start{coord_col_prefix[-1]}",
        f"end{coord_col_prefix[-1]}", 
        "row_idx"
    ]].copy().rename(columns={
        f"{coord_col_prefix}": "chrom",
        f"start{coord_col_prefix[-1]}": "start",
        f"end{coord_col_prefix[-1]}": "end"
    }).sort_values(["chrom","start"]).reset_index(drop=True)
    return _find_overlaps_in_sorted_bed(df, chrom_intervals)


# ——————————————————————————————————————————————————————————
# 3) generic “add contact strengths” API
# ——————————————————————————————————————————————————————————

def add_contact_strengths_to_varp(
    adata: GRAnData,
    bedp_files: list[Path]|Path,
    *,
    array_name: str,
    var_dim: str,
    var_target_dim: str,
    obs_dim: str,
    chrom_suffix: str = "chrom",
    start_suffix: str = "start",
    end_suffix:   str = "end",
    index_suffix: str = "index",
) -> GRAnData:
    """
    For each BEDP in bedp_files, find overlaps against adata’s var_dim intervals,
    accumulate (i,j,obs,score) tuples, and assemble a sparse COO of shape
      (n_var, n_var, n_obs),
    then wrap in an xarray.DataArray with dims [var_dim, var_target_dim, obs_dim]
    under adata.data_vars[array_name].
    """
    # 1) build chrom→intervals
    chrom_intervals = prepare_intervals(
        adata, var_dim,
        chrom_suffix=chrom_suffix,
        start_suffix=start_suffix,
        end_suffix=end_suffix,
        index_suffix=index_suffix,
    )

    # 2) precompute sizes and index maps
    n_var       = adata.sizes[var_dim]
    bedp_list   = list(map(Path, bedp_files))
    n_obs       = len(bedp_list)
    var_index   = adata[f"{var_dim}-_-{index_suffix}"].values.astype(str)
    var_to_idx  = {v:i for i,v in enumerate(var_index)}

    rows = []
    cols = []
    obs  = []
    dat  = []

    # 3) loop BEDP files
    for obs_idx, path in enumerate(bedp_list):
        df = pd.read_csv(
            path, sep="\t", header=None,
            names=["chr1","start1","end1","chr2","start2","end2","score"]
        )
        df["row_idx"] = df.index

        # find overlaps for each end
        overlaps1 = _find_overlaps_for_bedp(df, chrom_intervals, "chr1")
        overlaps2 = _find_overlaps_for_bedp(df, chrom_intervals, "chr2")

        for _, row in df.iterrows():
            o1 = overlaps1.get(int(row["row_idx"]), [])
            o2 = overlaps2.get(int(row["row_idx"]), [])
            if not o1 or not o2:
                continue
            sc = float(row["score"])
            for v1 in o1:
                i1 = var_to_idx.get(v1)
                if i1 is None: continue
                for v2 in o2:
                    i2 = var_to_idx.get(v2)
                    if i2 is None: continue
                    rows.append(i1)
                    cols.append(i2)
                    obs.append(obs_idx)
                    dat.append(sc)

    # 4) build sparse COO (even if no contacts, produce a dummy zero)
    if len(rows)==0:
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        obs  = np.array([0], dtype=int)
        dat  = np.array([0.0], dtype=float)
    else:
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        obs  = np.array(obs,  dtype=int)
        dat  = np.array(dat,  dtype=float)

    shape = (n_var, n_var, n_obs)
    coords = {
        var_dim:        var_index,
        var_target_dim: var_index,
        obs_dim:        [p.stem for p in bedp_list],
    }
    dims = [var_dim, var_target_dim, obs_dim]

    coo = sparse.COO( np.vstack([rows,cols,obs]), dat, shape=shape )
    da  = xr.DataArray(coo, dims=dims, coords=coords)

    # 5) insert into the GRAnData
    adata[array_name] = da
    return adata
