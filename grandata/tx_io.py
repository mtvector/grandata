import h5py
import numpy as np
import pandas as pd
import xarray as xr
import sparse
import pybigtools

from scipy.sparse import csr_matrix
from pathlib import Path
from itertools import product

from typing import Union, Literal, List, Tuple, Dict

from grandata import GRAnData
from gtfparse import read_gtf


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

    # ————— Helpers (same as before) ——————————————————————————————————

    def h5_tree(g):
        out = {}
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                out[k] = h5_tree(v)
            else:
                try: out[k] = len(v)
                except TypeError: out[k] = "scalar"
        return out

    def dict_to_ete3_tree(d, parent=None):
        from ete3 import Tree
        if parent is None: parent = Tree(name="root")
        for k, v in d.items():
            c = parent.add_child(name=k)
            if isinstance(v, dict):
                dict_to_ete3_tree(v, c)
        return parent

    def ete3_tree_to_dict(t):
        def helper(n):
            if n.is_leaf(): return n.name
            return {c.name: helper(c) for c in n.get_children()}
        return {c.name: helper(c) for c in t.get_children()}

    def prune_tree(tree, keep_keys):
        t = dict_to_ete3_tree(tree)
        keep = set()
        for key in keep_keys:
            for node in t.search_nodes(name=key):
                keep.update(node.iter_ancestors())
                keep.update(node.iter_descendants())
                keep.add(node)
        for n in t.traverse("postorder"):
            if n not in keep and n.up:
                n.detach()
        return ete3_tree_to_dict(t)

    def read_h5_to_dict(group, subtree):
        def helper(grp, sub):
            out = {}
            for k, v in sub.items():
                if isinstance(v, dict):
                    out[k] = helper(grp[k], v) if k in grp else None
                else:
                    if k in grp and isinstance(grp[k], h5py.Dataset):
                        ds = grp[k]
                        if ds.shape == ():
                            out[k] = ds[()]
                        else:
                            arr = ds[...]
                            if arr.dtype.kind == "S":
                                arr = arr.astype(str)
                            out[k] = arr
                    else:
                        out[k] = None
            return out
        return helper(group, subtree)

    def convert_to_dataframe(d: dict) -> pd.DataFrame:
        # infer length
        length = next((len(v) for v in d.values() if not isinstance(v, dict)), None)
        if length is None:
            raise ValueError("Cannot infer obs/var length")
        cols = {}
        for k, v in d.items():
            if isinstance(v, dict) and {"categories","codes"} <= set(v):
                codes = np.asarray(v["codes"], int)
                cats  = [c.decode() if isinstance(c, bytes) else c for c in v["categories"]]
                if len(codes)==length:
                    cols[k] = pd.Categorical.from_codes(codes, cats)
            elif isinstance(v, dict) and {"data","indices","indptr"} <= set(v):
                shape = tuple(v.get("shape",(length, max(v["indices"])+1)))
                cols[k] = csr_matrix((v["data"], v["indices"], v["indptr"]), shape=shape)
            elif not isinstance(v, dict):
                arr = np.asarray(v)
                if arr.ndim==1 and arr.shape[0]==length:
                    if arr.dtype.kind=="S":
                        arr = arr.astype(str)
                    cols[k] = arr
        return pd.DataFrame(cols)

    # ————— Read HDF5 and prune ——————————————————————————————————

    with h5py.File(filename, mode) as f:
        full_tree = h5_tree(f)
        pruned    = prune_tree(full_tree, selected_fields)
        raw       = read_h5_to_dict(f, pruned)

    data_vars = {}
    coords     = {}

    # — obs: unpack into coords + obs-_-col ——————————————————————————————
    if "obs" in raw:
        od = raw["obs"]
        idx = od.pop("_index", None)
        obs_df = convert_to_dataframe(od)
        if idx is not None:
            obs_df.index = [str(x) for x in idx]
        coords["obs"] = obs_df.index.to_numpy()

        # now unpack columns
        for col in obs_df.columns:
            data_vars[f"obs-_-{col}"] = xr.DataArray(
                obs_df[col].values,
                dims=("obs",),
                coords={"obs": coords["obs"]}
            )
        # also store index
        data_vars["obs-_-index"] = xr.DataArray(coords["obs"], dims=("obs",))

    # — var: same pattern ——————————————————————————————————————————————
    if "var" in raw:
        vd = raw["var"]
        idx = vd.pop("_index", None)
        var_df = convert_to_dataframe(vd)
        if idx is not None:
            var_df.index = [str(x) for x in idx]
        coords["var"] = var_df.index.to_numpy()

        for col in var_df.columns:
            data_vars[f"var-_-{col}"] = xr.DataArray(
                var_df[col].values,
                dims=("var",),
                coords={"var": coords["var"]}
            )
        data_vars["var-_-index"] = xr.DataArray(coords["var"], dims=("var",))

    # — X matrix ——————————————————————————————————————————————————
    if "X" in raw:
        xraw = raw["X"]
        print(xraw)
        if isinstance(xraw, dict) and {"data","indices","indptr"} <= set(xraw):
            csr_mat = csr_matrix((xraw["data"], xraw["indices"], xraw["indptr"]))
                                  #shape=tuple(xraw["shape"]))
            arr = sparse.COO.from_scipy_sparse(csr_mat)
        else:
            arr = np.asarray(xraw)
        data_vars["X"] = xr.DataArray(arr, dims=("obs","var"), coords=coords)

    # — layers/obsm/varm/obsp ——————————————————————————————————————————
    for grp in ("layers","obsm","varm","obsp"):
        if grp in raw:
            for name, val in raw[grp].items():
                if val is None:
                    continue
                if isinstance(val, dict) and {"data","indices","indptr"} <= set(val):
                    csr_mat = csr_matrix((val["data"], val["indices"], val["indptr"]))
                                          #shape=tuple(val.get("shape",arr.shape)))
                    arr = sparse.COO.from_scipy_sparse(csr_mat)
                else:
                    arr = np.asarray(val)

                if grp=="layers":
                    dims, c = ("obs","var"), coords
                elif grp=="obsm":
                    d2 = f"obsm_{name}"
                    dims, c = ("obs",d2), {"obs":coords["obs"],d2:np.arange(arr.shape[1])}
                elif grp=="varm":
                    d2 = f"varm_{name}"
                    dims, c = ("var",d2), {"var":coords["var"],d2:np.arange(arr.shape[1])}
                else:  # obsp
                    d2 = f"obsp_{name}"
                    dims, c = ("obs",d2), {"obs":coords["obs"],d2:coords["obs"]}

                data_vars[f"{grp}-_-{name}"] = xr.DataArray(arr, dims=dims, coords=c)

    # ——— Finally, build and return GRAnData ——————————————————————
    return GRAnData(data_vars=data_vars, coords=coords)

def write_tss_bigwigs_pybigtools(
    matrix: np.ndarray,
    var_names: list[str],
    obs_names: list[str],
    gtf_file: str,
    target_dir: str,
    gtf_gene_field: str = 'gene',
    n_bases: int = 1000,
    chromsizes: dict[str, int] = None,
):
    """
    Write signed TSS-aligned transcription bigWig files (1 per obs)
    using pybigtools.BigWigWrite.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (n_obs, n_var), transcription values.
    var_names : list[str]
        Names of genes, in the same order as matrix columns.
    obs_names : list[str]
        Names for each observation (e.g. clusters, pseudobulk sets).
    gtf_file : str
        Path to a gene annotation GTF.
    target_dir : str
        Folder where output .bw files are written.
    n_bases : int
        Number of bases downstream of the TSS to represent.
    chromsizes : dict[str, int], optional
        Chromosome sizes. If not provided, inferred from GTF.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Load GTF and restrict to gene-level features
    gtf = read_gtf(gtf_file)
    gtf = gtf[(gtf["feature"] == "gene") & (gtf[gtf_gene_field].isin(var_names))]
    gtf = gtf.dropna(subset=["seqname", "start", "end", "strand", gtf_gene_field])

    # Align with var_names order
    gtf_by_name = gtf.set_index(gtf_gene_field).loc[var_names]

    # Infer chromsizes if needed
    if chromsizes is None:
        chromsizes = (
            gtf_by_name[["seqname", "end"]]
            .groupby("seqname")["end"]
            .max()
            .astype(int)
            .to_dict()
        )

    # Extract TSS positions and strand per gene
    chroms   = gtf_by_name["seqname"].tolist()
    starts   = gtf_by_name["start"].astype(int).tolist()
    ends     = gtf_by_name["end"].astype(int).tolist()
    strands  = gtf_by_name["strand"].tolist()
    tss_list = [s if strand == "+" else e for s, e, strand in zip(starts, ends, strands)]

    # Write one BigWig file per observation
    for obs_idx, obs_name in enumerate(obs_names):
        path = target_dir / f"{obs_name}.bw"
        values_to_write = []

        for i, value in enumerate(matrix[obs_idx]):
            chrom = chroms[i]
            tss   = tss_list[i]
            strand = strands[i]

            start = tss
            end   = tss + n_bases
            if start >= chromsizes.get(chrom, 0):
                continue
            if end > chromsizes[chrom]:
                end = chromsizes[chrom]

            signed_value = float(value) * (1 if strand == "+" else -1)
            values_to_write.append((chrom, start, end, signed_value))

        # Write with pybigtools
        with pybigtools.BigWigWrite(str(path)) as writer:
            writer.write(chroms=chromsizes, vals=values_to_write)

def group_aggr_xr(
    ds: xr.Dataset,
    array_name: str,
    categories: Union[str, List[str]],
    agg_func=np.mean,
    normalize: bool = False,
) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """
    Group–aggregate an xarray.Dataset along 'obs' by one or more categorical
    obs columns, using xarray.groupby on the specified data array.

    Parameters
    ----------
    ds
        An xarray.Dataset (e.g. CrAnData) containing:
          - a DataArray `ds[array_name]` with dims ("obs","var") or similar,
          - one or more obs columns named "obs-_-<cat>".
    array_name
        Name of the DataArray in `ds` to aggregate (e.g. "X", "layers-_-counts", "obsp-_-contacts").
    categories
        Single category name or list of names (the <cat> in "obs-_-<cat>").
    agg_func
        Aggregation function (e.g. np.mean, np.median, np.std).
    normalize
        If True, each observation is normalized by its row‑sum before grouping.

    Returns
    -------
    result : np.ndarray
        Aggregated values, shape (*category_sizes, num_vars).
    category_orders : dict
        Maps each category name → list of its observed levels (in first‑appearance order).
    """
    # — normalize categories list —
    if isinstance(categories, str):
        categories = [categories]
    if not categories:
        raise ValueError("Must supply at least one category name")

    # — pick the DataArray and its dims —
    da = ds[array_name]
    obs_dim, var_dim = da.dims[:2]
    n_vars = da.sizes[var_dim]

    # — collect category arrays & orders —
    category_orders: Dict[str, List[str]] = {}
    cat_arrs: List[np.ndarray] = []
    for cat in categories:
        arr = ds[f"obs-_-{cat}"].values.astype(str)
        # preserve first‑appearance order
        seen = dict.fromkeys(arr.tolist())
        category_orders[cat] = list(seen.keys())
        cat_arrs.append(arr)

    # — build grouping coordinate —
    if len(categories) == 1:
        group_dim = categories[0]
        grouping = xr.DataArray(cat_arrs[0], dims=obs_dim, coords={obs_dim: ds.coords[obs_dim]})
    else:
        sep = "__"
        combo = cat_arrs[0]
        for arr in cat_arrs[1:]:
            combo = np.char.add(np.char.add(combo, sep), arr)
        group_dim = sep.join(categories)
        grouping = xr.DataArray(combo, dims=obs_dim, coords={obs_dim: ds.coords[obs_dim]})

    da = da.assign_coords(**{group_dim: grouping})

    # — optional normalize each row by its sum —
    if normalize:
        da = da / da.sum(dim=var_dim, keepdims=True)

    # — groupby & reduce —
    grouped = da.groupby(group_dim).reduce(agg_func, dim=obs_dim)

    # — extract the raw data, densifying if needed —
    raw = grouped.data
    if hasattr(raw, "todense"):
        arr = raw.todense()
    elif hasattr(raw, "toarray"):
        arr = raw.toarray()
    else:
        arr = np.asarray(raw)

    # — reorder and reshape into (*category_sizes, n_vars) —
    if len(categories) == 1:
        cats = category_orders[categories[0]]
        # ensure our output follows the same order
        idx = [cats.index(v) for v in grouped[ group_dim ].values.astype(str)]
        result = arr[idx, :]
    else:
        lists = [category_orders[c] for c in categories]
        combos = list(product(*lists))
        combo_strs = [sep.join(c) for c in combos]
        idx = [combo_strs.index(v) for v in grouped[group_dim].values.astype(str)]
        reshaped = arr[idx, :]
        sizes = [len(category_orders[c]) for c in categories]
        result = reshaped.reshape(*sizes, n_vars)

    return result, category_orders

