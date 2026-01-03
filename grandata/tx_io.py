import h5py
import numpy as np
import pandas as pd
import xarray as xr
import sparse
import pybigtools
from tqdm import tqdm
from scipy.sparse import csr_matrix
from pathlib import Path
from itertools import product
import re
from typing import Union, Literal, List, Tuple, Dict

from . import GRAnData
# from gtfparse import read_gtf


def read_gtf(gtf: str) -> pd.DataFrame:
    df = pd.read_csv(
        gtf,
        sep="\t",
        header=None,
        comment="#",
        dtype=str,        # keep everything as string
        names=[
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )

    fields = [
        "gene_id",
        "transcript_id",
        "exon_number",
        "gene",
        "gene_name",
        "gene_source",
        "gene_biotype",
        "transcript_name",
        "transcript_source",
        "transcript_biotype",
        "protein_id",
        "exon_id",
        "tag",
    ]

    for field in fields:
        pattern = rf"{field}\s+\"([^\"]+)\""
        df[field] = df["attribute"].str.extract(pattern)

    df.drop(columns="attribute", inplace=True)

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    return df

def read_h5ad_selective_to_grandata(
    filename: Union[str, Path],
    mode: Literal["r", "r+"] = "r",
    selected_fields: List[str] = None,
) -> GRAnData:
    """
    Based on similar function from ANTIPODE.
    Read just the specified top‐level AnnData fields (e.g. "X","obs","var","layers", etc.)
    from an .h5ad file via h5py, reconstruct sparse/categorical if needed,
    and return a GRAnData (xarray.Dataset). This version unpacks obs/var
    into -_- columns so we never pass a DataFrame into GRAnData.__init__.
    """
    selected_fields = selected_fields or ["X", "obs", "var"]

    def h5_tree(g):
        out = {}
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                out[k] = h5_tree(v)
            else:
                try:
                    out[k] = len(v)
                except TypeError:
                    out[k] = "scalar"
        return out

    def prune_tree(tree_dict, keep_keys):
        """
        Return a pruned version of `tree_dict` that includes only the top‐level
        keys in `keep_keys` (if present), along with their entire nested structure.
        """
        pruned = {}
        for k in keep_keys:
            if k in tree_dict:
                pruned[k] = tree_dict[k]
        return pruned

    def read_h5_to_dict(group, subtree):
        def helper(grp, sub):
            out = {}
            for k, v in sub.items():
                if isinstance(v, dict):
                    out[k] = helper(grp[k], v) if (k in grp and isinstance(grp[k], h5py.Group)) else None
                else:
                    if k in grp and isinstance(grp[k], h5py.Dataset):
                        ds = grp[k]
                        if ds.shape == ():
                            out[k] = ds[()]
                        else:
                            arr = ds[...]
                            if arr.dtype.kind == "S":
                                # decode raw bytes to Unicode
                                arr = arr.astype("U")
                            out[k] = arr
                    else:
                        out[k] = None
            return out
        return helper(group, subtree)

    def convert_to_dataframe(d: dict) -> pd.DataFrame:
        # infer length from first non‐dict value
        length = next((len(v) for v in d.values() if not isinstance(v, dict)), None)
        if length is None:
            raise ValueError("Cannot infer obs/var length")
        cols = {}
        for k, v in d.items():
            if isinstance(v, dict) and {"categories", "codes"} <= set(v):
                codes = np.asarray(v["codes"], int)
                cats = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in v["categories"]
                ]
                if len(codes) == length:
                    cols[k] = pd.Categorical.from_codes(codes, cats)
            elif isinstance(v, dict) and {"data", "indices", "indptr"} <= set(v):
                max_ind = max(v["indices"]) + 1 if len(v["indices"]) > 0 else 0
                shape = tuple(v.get("shape", (length, max_ind)))
                cols[k] = csr_matrix(
                    (v["data"], v["indices"], v["indptr"]), shape=shape
                )
            elif not isinstance(v, dict):
                arr = np.asarray(v)
                if arr.ndim == 1 and arr.shape[0] == length:
                    if arr.dtype.kind == "S":
                        # decode raw bytes to Unicode
                        arr = arr.astype("U")
                    cols[k] = arr
        return pd.DataFrame(cols)

    # ————— Read HDF5 and prune ——————————————————————————————————

    with h5py.File(filename, mode) as f:
        full_tree = h5_tree(f)
        pruned = prune_tree(full_tree, selected_fields)
        raw = read_h5_to_dict(f, pruned)

    data_vars = {}
    coords = {}

    # — obs: unpack into coords + obs-_-col ——————————————————————————————
    if "obs" in raw:
        od = raw["obs"]
        idx = od.pop("_index", None)
        obs_df = convert_to_dataframe(od)
        if idx is not None:
            decoded_idx = []
            for x in idx:
                if isinstance(x, (bytes, np.bytes_)):
                    decoded_idx.append(x.decode("utf-8"))
                else:
                    decoded_idx.append(str(x))
            obs_df.index = decoded_idx
        coords["obs"] = obs_df.index.to_numpy()
    
        # unpack columns…
        for col in obs_df.columns:
            data_vars[f"obs-_-{col}"] = xr.DataArray(
                obs_df[col].values,
                dims=("obs",),
                coords={"obs": coords["obs"]},
            )
        data_vars["obs-_-index"] = xr.DataArray(coords["obs"], dims=("obs",))

    # — var: same pattern ——————————————————————————————————————————————
    if "var" in raw:
        vd = raw["var"]
        idx = vd.pop("_index", None)
        var_df = convert_to_dataframe(vd)
        if idx is not None:
            decoded_idx = []
            for x in idx:
                if isinstance(x, (bytes, np.bytes_)):
                    decoded_idx.append(x.decode("utf-8"))
                else:
                    decoded_idx.append(str(x))
            var_df.index = decoded_idx
        coords["var"] = var_df.index.to_numpy()
    
        for col in var_df.columns:
            data_vars[f"var-_-{col}"] = xr.DataArray(
                var_df[col].values,
                dims=("var",),
                coords={"var": coords["var"]},
            )
        data_vars["var-_-index"] = xr.DataArray(coords["var"], dims=("var",))

    # — X matrix ——————————————————————————————————————————————————
    if "X" in raw:
        xraw = raw["X"]
        if isinstance(xraw, dict) and {"data", "indices", "indptr"} <= set(xraw):
            csr_mat = csr_matrix((xraw["data"], xraw["indices"], xraw["indptr"]))
            arr = sparse.COO.from_scipy_sparse(csr_mat)
        else:
            arr = np.asarray(xraw)
        data_vars["X"] = xr.DataArray(arr, dims=("obs", "var"), coords=coords)

    # — layers/obsm/varm/obsp ——————————————————————————————————————————
    for grp in ("layers", "obsm", "varm", "obsp"):
        if grp in raw:
            for name, val in raw[grp].items():
                if val is None:
                    continue
                if isinstance(val, dict) and {"data", "indices", "indptr"} <= set(val):
                    csr_mat = csr_matrix((val["data"], val["indices"], val["indptr"]))
                    arr = sparse.COO.from_scipy_sparse(csr_mat)
                else:
                    arr = np.asarray(val)

                if grp == "layers":
                    dims, c = ("obs", "var"), coords
                elif grp == "obsm":
                    d2 = f"obsm_{name}"
                    dims, c = ("obs", d2), {"obs": coords["obs"], d2: np.arange(arr.shape[1])}
                elif grp == "varm":
                    d2 = f"varm_{name}"
                    dims, c = ("var", d2), {"var": coords["var"], d2: np.arange(arr.shape[1])}
                else:  # obsp
                    d2 = f"obsp_{name}"
                    dims, c = ("obs", d2), {"obs": coords["obs"], d2: coords["obs"]}

                data_vars[f"{grp}-_-{name}"] = xr.DataArray(arr, dims=dims, coords=c)

    # ——— Finally, build and return GRAnData ——————————————————————
    return GRAnData(data_vars=data_vars, coords=coords)

def write_tss_bigwigs(
    matrix: np.ndarray | xr.DataArray,
    var_names: list[str] | None,
    obs_names: list[str] | None,
    gtf_file: str,
    target_dir: str,
    gtf_gene_field: str = 'gene',
    n_bases: int = 1000,
    chromsizes: dict[str, int] = None,
    gene_replace_dict = None
):
    """
    Write signed TSS-aligned transcription bigWig files (1 per obs),
    merging any overlapping TSS intervals by summing their values.

    Parameters
    ----------
    matrix : np.ndarray | xr.DataArray
        Shape (n_obs, n_var), transcription values. If DataArray, obs/var names
        can be inferred from its coords.
    var_names : list[str] | None
        Names of genes, in the same order as matrix columns.
    obs_names : list[str] | None
        Names for each observation (e.g. clusters, pseudobulk sets).
    gtf_file : str
        Path to a gene annotation GTF.
    target_dir : str
        Folder where output .bw files are written.
    n_bases : int
        Number of bases downstream of the TSS to represent.
    chromsizes : dict[str, int], optional
        Chromosome sizes. If not provided, inferred from GTF.
    gene_replace_dict : dict
        Dictionary to convert GTF gene names to new names
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = None
    var_dim = None
    if isinstance(matrix, xr.DataArray):
        obs_dim, var_dim = matrix.dims[:2]
        if obs_names is None:
            obs_names = matrix.coords[obs_dim].astype(str).tolist()
        if var_names is None:
            var_names = matrix.coords[var_dim].astype(str).tolist()
    if obs_names is None or var_names is None:
        raise ValueError("obs_names and var_names must be provided for ndarray inputs.")
    
    # --- Load GTF and restrict to gene-level features ---
    gtf = read_gtf(gtf_file)
    gtf = gtf.loc[gtf["feature"] == "gene"]

    if gene_replace_dict is not None:
        gtf[gtf_gene_field] = [gene_replace_dict.get(x,x) for x in gtf[gtf_gene_field]]
    gtf = gtf.loc[~gtf[gtf_gene_field].isna()]
    
    shared_var_names = list(
        set(gtf[gtf_gene_field].unique()) & set(var_names)
    )
    gtf = gtf.loc[gtf[gtf_gene_field].isin(shared_var_names)]
    gtf = gtf.dropna(subset=["seqname", "start", "end", "strand", gtf_gene_field])
    
    # Re‐align matrix to only those var_names that exist in GTF:
    keep_mask = np.array([x in shared_var_names for x in var_names])
    matrix = matrix[:, keep_mask]

    # Now re‐index GTF so that the row order matches shared_var_names exactly:
    gtf_by_name = gtf.set_index(gtf_gene_field).loc[shared_var_names]

    # Infer chromsizes if needed:
    if chromsizes is None:
        chromsizes = (
            gtf_by_name[["seqname", "end"]]
            .groupby("seqname")["end"]
            .max()
            .astype(int)
            .to_dict()
        )

    # Extract per‐gene chromosome, TSS position, and strand:
    chroms = gtf_by_name["seqname"].tolist()
    starts = gtf_by_name["start"].astype(int).tolist()
    ends = gtf_by_name["end"].astype(int).tolist()
    strands = gtf_by_name["strand"].tolist()
    # TSS is 'start' if '+' strand, else 'end' for '-'
    tss_list = [s if strand == "+" else e for s, e, strand in zip(starts, ends, strands)]

    # --- Now write one BigWig per observation, merging overlaps ---
    for obs_idx, obs_name in enumerate(obs_names):
        obs_name = re.sub('/','-',obs_name) #just in case people (like me) have silly names
        obs_name = re.sub(' ','_',obs_name)
        
        print('writing',obs_name)
        path = target_dir / f"{obs_name}.bw"

        # 1) Build the raw interval list: (chrom, start, end, signed_value)
        raw_intervals: list[tuple[str, int, int, float]] = []
        if isinstance(matrix, xr.DataArray):
            row = matrix.isel({obs_dim: obs_idx}).data
            row_vals = np.asarray(row).ravel()
        else:
            row_vals = matrix[obs_idx]
        for i, value in tqdm(list(enumerate(row_vals)), desc=f"Building intervals for {obs_name}", leave=False):
            chrom = chroms[i]
            tss = tss_list[i]
            strand = strands[i]

            # Define the 0-based interval [tss, tss + n_bases)
            start = tss
            end = tss + n_bases
            if start >= chromsizes.get(chrom, 0):
                continue
            if end > chromsizes[chrom]:
                end = chromsizes[chrom]

            signed_value = float(value) * (1.0 if strand == "+" else -1.0)
            raw_intervals.append((chrom, start, end, signed_value))

        # 2) Group intervals by chromosome
        chrom_to_intervals: dict[str, list[tuple[int, int, float]]] = {}
        for chrom, s, e, v in raw_intervals:
            chrom_to_intervals.setdefault(chrom, []).append((s, e, v))

        # 3) For each chromosome, merge overlapping intervals via sweep‐line
        merged_values: list[tuple[str, int, int, float]] = []
        for chrom, iv_list in chrom_to_intervals.items():
            # Build event list: (position, delta_value). We'll store both +v at start, −v at end.
            events: list[tuple[int, float]] = []
            for s, e, v in iv_list:
                # Only consider non‐empty intervals
                if e <= s:
                    continue
                events.append((s, +v))
                events.append((e, -v))

            # Sort by position. If two events share the same position, ensure positive deltas come first
            # so that we do not accidentally drop a segment where a start and end coincide.
            events.sort(key=lambda x: (x[0], -x[1]))

            current_sum = 0.0
            prev_pos = None
            idx = 0
            n_events = len(events)

            while idx < n_events:
                pos = events[idx][0]
                # Before we add the deltas at 'pos', if there's a previous segment running
                if prev_pos is not None and pos > prev_pos and current_sum != 0.0:
                    # Emit the merged interval [prev_pos, pos) with the running sum
                    merged_values.append((chrom, prev_pos, pos, current_sum))

                # Now consume all events with this same 'pos'
                while idx < n_events and events[idx][0] == pos:
                    current_sum += events[idx][1]
                    idx += 1

                prev_pos = pos

            # No need to handle trailing segment: by definition, once current_sum returns to zero,
            # no further intervals remain. If it never returns to zero, we've handled until the last event.

        # 4) Finally, sort merged_values by chromosome + start (just in case)
        merged_values.sort(key=lambda x: (x[0], x[1]))

        # 5) Write out the merged intervals to BigWig
        writer = pybigtools.open(str(path), mode='w')
        pybigtools.BigWigWrite.write(writer, chroms=chromsizes, vals=merged_values)
        writer.close()

        
def group_aggr_xr(
    ds: xr.Dataset,
    array_name: str,
    categories: Union[str, List[str]],
    agg_func=np.mean,
    normalize: bool = False,
    materialize: bool = True,
) -> xr.DataArray:
    """
    Group–aggregate an xarray.Dataset along 'obs' by one or more categorical
    obs columns, using xarray.groupby on the specified data array, and return
    a DataArray whose dimensions correspond to each category plus the var dimension.

    Parameters
    ----------
    ds
        An xarray.Dataset containing:
          - a DataArray `ds[array_name]` with dims ("obs","var") or similar,
          - one or more obs columns named "obs-_-<category>".
    array_name
        Name of the DataArray in `ds` to aggregate (e.g. "X", "layers-_-counts", "obsp-_-contacts").
    categories
        Single category name or list of names like obs-_-<category>).
    agg_func
        Aggregation function (e.g. np.mean, np.median, np.std).
    normalize
        If True, each observation is normalized by its row-sum before grouping.
    materialize
        If False, return the grouped DataArray without densifying or reshaping.

    Returns
    -------
    xr.DataArray
        A DataArray with dimensions:
          - one dimension per category (named exactly as in `categories`),
          - plus the var dimension (same name as in `ds[array_name]`).
        The coords along each category axis are the observed levels of that category
        (in first-appearance order), and the coord along the var axis is carried
        over from the original DataArray.
    """
    # — normalize categories list —
    if isinstance(categories, str):
        categories = [categories]
    if not categories:
        raise ValueError("Must supply at least one category name")

    # — pick the DataArray and its dims —
    da = ds[array_name]
    obs_dim, var_dim = da.dims[:2]
    # capture the original var-axis coordinate
    var_coord = da.coords[var_dim]

    # — collect category arrays & orders —
    category_orders = {}
    cat_arrs = []
    for cat in categories:
        arr = ds[cat].values.astype(str)
        # preserve first-appearance order
        seen = dict.fromkeys(arr.tolist())
        category_orders[cat] = list(seen.keys())
        cat_arrs.append(arr)

    # — build a combined grouping key (string) for xarray.groupby —
    if len(categories) == 1:
        group_key = categories[0]
        grouping = xr.DataArray(cat_arrs[0],
                                dims=obs_dim,
                                coords={obs_dim: ds.coords[obs_dim]})
    else:
        sep = "____"
        combo = cat_arrs[0].astype('<U0')
        for arr in cat_arrs[1:]:
            combo = np.char.add(np.char.add(combo, sep), arr)
        group_key = sep.join(categories)
        grouping = xr.DataArray(combo,
                                dims=obs_dim,
                                coords={obs_dim: ds.coords[obs_dim]})

    # assign the grouping coordinate (internally) so we can group by it
    da = da.assign_coords(**{group_key: grouping})

    # — optional normalize each row by its sum —
    if normalize:
        da = da / da.sum(dim=var_dim, keep_attrs=True)

    # — groupby & reduce over obs_dim —
    grouped = da.groupby(group_key).reduce(agg_func, dim=obs_dim)
    if not materialize:
        return grouped

    # — pull out the raw numpy array, densify if sparse-backed —
    raw = grouped.data
    if hasattr(raw, "todense"):
        arr = raw.todense()
    elif hasattr(raw, "toarray"):
        arr = raw.toarray()
    else:
        arr = np.asarray(raw)

    # — reorder and reshape into (*category_sizes, n_vars) —
    n_vars = da.sizes[var_dim]
    if len(categories) == 1:
        cat = categories[0]
        levels = category_orders[cat]
        # the grouped index gives the observed levels in the grouping order
        observed = grouped[ group_key ].values.astype(str).tolist()
        idx = [levels.index(v) for v in observed]
        result = arr[idx, :]
        # dims and coords for the single‐category case
        dims = [cat, var_dim]
        coords = {
            cat: levels,
            var_dim: var_coord
        }
    else:
        # build the full cartesian product of category levels
        lists_of_levels = [category_orders[c] for c in categories]
        all_combos = list(product(*lists_of_levels))
        combo_strs = [sep.join(c) for c in all_combos]
        observed = grouped[group_key].values.astype(str).tolist()
        idx = [combo_strs.index(v) for v in observed]
        reshaped = arr[idx, :]
        sizes = [len(category_orders[c]) for c in categories]
        result = reshaped.reshape(*sizes, n_vars)
        # dims and coords for the multi‐category case
        dims = categories + [var_dim]
        coords = {var_dim: var_coord}
        for cat in categories:
            coords[cat] = category_orders[cat]

    # — construct and return the aggregated DataArray —
    return xr.DataArray(data=result, dims=dims, coords=coords)
