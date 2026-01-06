"""chrom_io.py – Create GRAnData stores and arrays from BigWig inputs."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pybigtools
import xarray as xr
from tqdm import tqdm

from .grandata import GRAnData
from collections import defaultdict
try:
    import sparse
except:
    print('no sparse')

# ——————————————————————————————————————————————————————————
# low-level utilities
# ——————————————————————————————————————————————————————————

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

    Note: this uses a dask-first pipeline; `backend` must be "zarr".
    The parameters `tile_size` and `n_workers` are retained for compatibility
    but are ignored.
    """
    if backend != "zarr":
        raise ValueError("Only backend='zarr' is supported for grandata_from_bigwigs.")

    bigwig_dir = Path(bigwig_dir)

    # 1) scan BigWigs and collect chromosomes
    bw_files = []
    chroms = set()
    for p in tqdm(list(bigwig_dir.iterdir()), desc="Scanning BigWigs"):
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
    try:
        from dask.diagnostics import ProgressBar
    except Exception:
        ProgressBar = None
    if ProgressBar is None:
        adata.to_zarr(str(backed_path), mode="w")
    else:
        with ProgressBar():
            adata.to_zarr(str(backed_path), mode="w")
    adata = GRAnData.open_zarr(str(backed_path), consolidated=False)

    # 5) stream in the very first array
    return add_bigwig_array_dask(
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
        fill_value=fill_value,
        obs_chunk_size=obs_chunk_size,
    )


def grandata_from_bigwigs_dask(
    region_table: pd.DataFrame,
    bigwig_dir: Path | str,
    backed_path: Path | str,
    target_region_width: int,
    *,
    array_name: str = "X",
    obs_dim: str = "obs",
    var_dim: str = "var",
    seq_dim: str = "seq_bins",
    bin_stat: str = "mean",
    chunk_size: int = 512,
    n_bins: int = 1,
    fill_value: float = 0.0,
    obs_chunk_size: int | None = None,
) -> GRAnData:
    """
    Dask-first version of grandata_from_bigwigs.

    Builds a lazy dask array for the BigWig extraction and writes via xarray.to_zarr.
    Avoids manual zarr writes and uses stable xarray/dask APIs.
    """
    try:
        import dask.array as da
    except ImportError as exc:
        raise ImportError("grandata_from_bigwigs_dask requires dask.") from exc

    bigwig_dir = Path(bigwig_dir)
    bw_files = []
    chroms = set()
    for p in tqdm(list(bigwig_dir.iterdir()), desc="Scanning BigWigs"):
        try:
            with pybigtools.open(str(p), "r") as bw:
                chroms |= set(bw.chroms().keys())
            bw_files.append(p)
        except Exception:
            continue
    if not bw_files:
        raise FileNotFoundError(f"No BigWigs in {bigwig_dir!r}")
    bw_files.sort()

    peaks = region_table.query("chrom in @chroms").reset_index(drop=True)
    peaks["region"] = (
        peaks.chrom.astype(str)
        + ":"
        + peaks.start.astype(int).astype(str)
        + "-"
        + peaks.end.astype(int).astype(str)
    )

    obs_idx = [p.stem.replace(".", "_") for p in bw_files]
    var_idx = peaks.region.values

    extra = {}
    extra[f"{obs_dim}-_-index"] = xr.DataArray(obs_idx, dims=[obs_dim])
    extra[f"{var_dim}-_-index"] = xr.DataArray(var_idx, dims=[var_dim])
    for col in ("chrom", "start", "end", "region"):
        extra[f"{var_dim}-_-{col}"] = (
            xr.DataArray(peaks[col].values, dims=[var_dim]).chunk({var_dim: chunk_size})
        )

    adata = GRAnData(**extra)
    adata.attrs["chunk_size"] = chunk_size
    adata.attrs["target_region_width"] = target_region_width
    adata.attrs["bin_stat"] = bin_stat
    adata.attrs["n_bins"] = n_bins
    adata.attrs["obs_dim"] = obs_dim
    adata.attrs["var_dim"] = var_dim
    adata.attrs["seq_dim"] = seq_dim

    if obs_chunk_size is None:
        obs_chunk_size = min(16, len(obs_idx))

    widths = peaks.end - peaks.start
    half = (target_region_width - widths) // 2
    starts = (peaks.start - half).clip(lower=0)
    ends = starts + target_region_width
    intervals = list(
        zip(peaks.chrom.astype(str), starts.astype(int), ends.astype(int))
    )

    n_obs = len(obs_idx)
    n_var = len(var_idx)

    def _load_block(_block, block_info=None):
        info = block_info[0]["array-location"]
        obs_loc = info[0]
        var_loc = info[1]
        obs_slice = obs_loc if isinstance(obs_loc, slice) else slice(obs_loc[0], obs_loc[1])
        var_slice = var_loc if isinstance(var_loc, slice) else slice(var_loc[0], var_loc[1])
        return _extract_bw_chunk(
            bw_files=bw_files,
            intervals=intervals,
            obs_slice=obs_slice,
            var_slice=var_slice,
            n_bins=n_bins,
            bin_stat=bin_stat,
            fill_value=fill_value,
        )

    template = da.zeros(
        (n_obs, n_var, n_bins),
        chunks=(obs_chunk_size, chunk_size, n_bins),
        dtype="float32",
    )
    data = da.map_blocks(_load_block, template, dtype="float32")

    adata[array_name] = xr.DataArray(
        data,
        dims=(obs_dim, var_dim, seq_dim),
        coords={
            obs_dim: obs_idx,
            var_dim: var_idx,
            seq_dim: np.arange(n_bins),
        },
    )

    try:
        from dask.diagnostics import ProgressBar
    except Exception:
        ProgressBar = None
    if ProgressBar is None:
        adata.to_zarr(str(backed_path), mode="w")
    else:
        with ProgressBar():
            adata.to_zarr(str(backed_path), mode="w")
    return GRAnData.open_zarr(str(backed_path), consolidated=False)


def _extract_bw_chunk(
    *,
    bw_files: list[Path | None],
    intervals: list[tuple[str, int, int]],
    obs_slice: slice,
    var_slice: slice,
    n_bins: int,
    bin_stat: str,
    fill_value: float,
) -> np.ndarray:
    obs_idx = np.arange(len(bw_files))[obs_slice]
    var_idx = np.arange(len(intervals))[var_slice]
    out = np.full((len(obs_idx), len(var_idx), n_bins), fill_value, dtype="float32")

    for o_i, obs_i in enumerate(obs_idx):
        bw_path = bw_files[obs_i]
        if bw_path is None:
            continue
        with pybigtools.open(str(bw_path), "r") as bw:
            for v_j, v_i in enumerate(var_idx):
                c, s, e = intervals[v_i]
                try:
                    vals = bw.values(
                        c,
                        int(s),
                        int(e),
                        missing=fill_value,
                        exact=False,
                        bins=n_bins,
                        summary=bin_stat,
                    )
                    out[o_i, v_j, :] = vals
                except KeyError:
                    continue
    return out

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
    """Compatibility wrapper around add_bigwig_array_dask (dask-first)."""
    return add_bigwig_array_dask(
        adata,
        region_table=region_table,
        bigwig_dir=bigwig_dir,
        array_name=array_name,
        obs_dim=obs_dim,
        var_dim=var_dim,
        seq_dim=seq_dim,
        target_region_width=target_region_width,
        bin_stat=bin_stat,
        chunk_size=chunk_size,
        n_bins=n_bins,
        fill_value=fill_value,
        obs_chunk_size=obs_chunk_size,
    )


def add_bigwig_array_dask(
    adata: GRAnData,
    region_table: pd.DataFrame,
    bigwig_dir: Path | str | list[Path],
    *,
    array_name: str,
    obs_dim: str,
    var_dim: str,
    seq_dim: str,
    target_region_width: int,
    bin_stat: str = "mean",
    chunk_size: int = 512,
    n_bins: int = 1,
    fill_value: float = np.nan,
    obs_chunk_size: int | None = None,
) -> GRAnData:
    try:
        import dask.array as da
    except ImportError as exc:
        raise ImportError("add_bigwig_array_dask requires dask.") from exc

    if "source" not in getattr(adata, "encoding", {}):
        raise ValueError("add_bigwig_array_dask requires a zarr-backed GRAnData.")

    if not isinstance(bigwig_dir, (list, tuple)):
        bw_files = sorted(Path(p) for p in Path(bigwig_dir).iterdir())
    else:
        bw_files = sorted(Path(p) for p in bigwig_dir)

    n_obs = len(adata[f"{obs_dim}-_-index"].values)
    n_var = len(region_table)

    obs_names = adata[f"{obs_dim}-_-index"].values.astype(str).tolist()
    file_map = {p.stem.replace(".", "_"): p for p in bw_files}
    bw_paths = [file_map.get(name) for name in obs_names]

    if obs_chunk_size is None:
        obs_chunk_size = min(16, n_obs)

    widths = region_table.end - region_table.start
    half = (target_region_width - widths) // 2
    starts = (region_table.start - half).clip(lower=0)
    ends = starts + target_region_width
    intervals = list(
        zip(region_table.chrom.astype(str), starts.astype(int), ends.astype(int))
    )

    def _load_block(_block, block_info=None):
        info = block_info[0]["array-location"]
        obs_loc = info[0]
        var_loc = info[1]
        obs_slice = obs_loc if isinstance(obs_loc, slice) else slice(obs_loc[0], obs_loc[1])
        var_slice = var_loc if isinstance(var_loc, slice) else slice(var_loc[0], var_loc[1])
        return _extract_bw_chunk(
            bw_files=bw_paths,
            intervals=intervals,
            obs_slice=obs_slice,
            var_slice=var_slice,
            n_bins=n_bins,
            bin_stat=bin_stat,
            fill_value=fill_value,
        )

    template = da.zeros(
        (n_obs, n_var, n_bins),
        chunks=(obs_chunk_size, chunk_size, n_bins),
        dtype="float32",
    )
    data = da.map_blocks(_load_block, template, dtype="float32")

    obs_labels = adata[f"{obs_dim}-_-index"].values.astype(str)
    if f"{var_dim}-_-index" in adata:
        var_labels = adata[f"{var_dim}-_-index"].values.astype(str)
    else:
        var_labels = np.arange(n_var)

    da_out = xr.DataArray(
        data,
        dims=(obs_dim, var_dim, seq_dim),
        coords={
            obs_dim: obs_labels,
            var_dim: var_labels,
            seq_dim: np.arange(n_bins),
        },
    )

    # Align to the existing store dims to avoid size-mismatch errors when adata is a subset.
    store_ds = xr.open_zarr(adata.encoding["source"], consolidated=False)
    if f"{obs_dim}-_-index" in store_ds:
        store_obs = store_ds[f"{obs_dim}-_-index"].values.astype(str)
    elif obs_dim in store_ds.coords:
        store_obs = store_ds.coords[obs_dim].values
    else:
        store_obs = np.arange(store_ds.sizes.get(obs_dim, len(obs_labels)))

    if f"{var_dim}-_-index" in store_ds:
        store_var = store_ds[f"{var_dim}-_-index"].values.astype(str)
    elif var_dim in store_ds.coords:
        store_var = store_ds.coords[var_dim].values
    else:
        store_var = np.arange(store_ds.sizes.get(var_dim, n_var))

    if len(store_obs) != len(obs_labels) or len(store_var) != len(var_labels):
        da_out = da_out.reindex(
            {obs_dim: store_obs, var_dim: store_var},
            fill_value=fill_value,
        )

    # Ensure uniform chunks after reindexing to satisfy zarr chunk constraints.
    if obs_chunk_size is None:
        obs_chunk_size = min(16, len(store_obs))
    da_out = da_out.chunk(
        {
            obs_dim: obs_chunk_size,
            var_dim: chunk_size,
            seq_dim: n_bins,
        }
    )

    try:
        from dask.diagnostics import ProgressBar
    except Exception:
        ProgressBar = None
    if ProgressBar is None:
        xr.Dataset({array_name: da_out}).to_zarr(adata.encoding["source"], mode="a")
    else:
        with ProgressBar():
            xr.Dataset({array_name: da_out}).to_zarr(adata.encoding["source"], mode="a")
    return GRAnData.open_zarr(adata.encoding["source"], consolidated=False)

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
