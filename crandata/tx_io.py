import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
import sparse
from pathlib import Path
from typing import Union, Literal, List
from crandata import CrAnData

def read_h5ad_selective_to_crandata(
    filename: Union[str, Path],
    mode: Literal["r", "r+"] = "r",
    selected_fields: List[str] = None,
) -> CrAnData:
    """
    Read just the specified top‐level AnnData fields (e.g. "X","obs","var","layers", etc.)
    from an .h5ad file via h5py, reconstruct sparse/categorical if needed,
    and return a CrAnData (xarray.Dataset).  This version unpacks obs/var
    into -_- columns so we never pass a DataFrame into CrAnData.__init__.
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

    # ——— Finally, build and return CrAnData ——————————————————————
    return CrAnData(data_vars=data_vars, coords=coords)


