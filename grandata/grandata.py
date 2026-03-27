import xarray as xr
import pandas as pd
import numpy as np
import os
import json
import zarr
from collections.abc import Sequence

try:
    import sparse  # for sparse multidimensional arrays
except ImportError:
    print("no sparse")
    # consider backed sparse: https://github.com/scverse/anndata/blob/main/src/anndata/_core/sparse_dataset.py
    sparse = None

class GRAnData(xr.Dataset):
    __slots__ = ("__dict__",)  # remove always_convert_df from slots

    def __init__(self, 
                 data_vars=None,
                 coords=None,
                 **kwargs):
        """
        Create a GRAnData object as a thin wrapper of xarray.Dataset.
        You can use get_dataframe('var') or 'obs' etc to get hierarchical variables
        grouped into dataframe (the separator is -_- ...idk).
        Also automatically stores and reloads sparse arrays.
        """
        if data_vars is None:
            data_vars = {}
        # Merge any additional kwargs.
        data_vars = dict(data_vars)
        data_vars.update(kwargs)
        
        # Unpack DataFrames for hierarchical grouping.
        for key in data_vars.keys():
            if key in data_vars and isinstance(data_vars[key], pd.DataFrame):
                df = data_vars.pop(key)
                for col in df.columns:
                    data_vars[f"{key}-_-{col}"] = xr.DataArray(df[col].values, dims=(key,))
                # Optionally, you can store the original index as a coordinate.
                data_vars[f"{key}-_-index"] = xr.DataArray(df.index.values, dims=(key,))
        
        # Ensure every variable is an xr.DataArray.
        for key, var in data_vars.items():
            if not isinstance(var, xr.DataArray):
                data_vars[key] = xr.DataArray(var)
        if coords is None:
            coords = {}
        
        super().__init__(data_vars=data_vars, coords=coords)
        
        # Make sure keys don't contain '/'.
        for key in self.data_vars:
            if "/" in key:
                safe_name = key.replace("/", "_")
                setattr(self, safe_name, self.data_vars[key])
                
    @property
    def array_names(self):
        return list(self.data_vars.keys())
        
    def get_dataframe(self, top):
        cols = {}
        expected = None
        # Gather all keys that start with "top-_-"
        for key in list(self.data_vars.keys()):
            if not key.startswith(top + "-_-"):
                continue
            da = super().__getitem__(key)
            if expected is None:
                expected = da.shape[0]
            elif da.shape[0] != expected:
                continue  # skip keys with mismatched length
            col_name = key.split("-_-", 1)[1]
            cols[col_name] = da.values
    
        if not cols:
            return None
    
        # Look for any column name containing "index"
        index_col = next((name for name in cols if name in ("index", "_index")), None)
        if index_col is not None:
            idx = cols.pop(index_col)
            return pd.DataFrame(cols, index=idx)
        else:
            return pd.DataFrame(cols, index=np.arange(expected))
    
    # === Sparse encoding/decoding methods ===
    @staticmethod
    def _encode_sparse(var):
        sp = var.data  # assume sparse.COO
        return {
            "sparse": True,
            "data": sp.data.tolist(),
            "coords": sp.coords.tolist(),
            "shape": list(sp.shape),
            "dtype": str(sp.dtype),
            "dims": var.dims  # Preserve the dimension names
        }
    
    @staticmethod
    def _decode_sparse(encoded):
        data = np.array(encoded["data"])
        coords = np.array(encoded["coords"])
        shape = tuple(encoded["shape"])
        return sparse.COO(coords, data, shape=shape)
    
    def sparse_serialized(self):
        if sparse is None:
            return self
        encoded_vars = {}
        keys_to_drop = []
        for key, var in list(self.data_vars.items()):
            try:
                data_attr = var.data
            except Exception:
                continue
            if hasattr(data_attr, "todense") and isinstance(data_attr, sparse.COO):
                encoded_vars[key] = json.dumps(self._encode_sparse(var))
                keys_to_drop.append(key)
        new_ds = self.drop_vars(keys_to_drop)
        for key, encoded_str in encoded_vars.items():
            new_ds = new_ds.assign({ "encoded_" + key: xr.DataArray(encoded_str, dims=()) })
        if encoded_vars:
            new_ds.attrs["sparse_encoded_keys"] = json.dumps(list(encoded_vars.keys()))
        return new_ds
    
    @classmethod
    def _decode_sparse_from_vars(cls, obj):
        new_vars = dict(obj.data_vars)
        for key in list(new_vars):
            if key.startswith("encoded_"):
                orig_key = key[len("encoded_"):]
                encoded_str = new_vars[key].values.item()
                enc = json.loads(encoded_str)
                decoded = cls._decode_sparse(enc)
                dims = enc.get("dims", None)  # Retrieve stored dims
                new_vars[orig_key] = xr.DataArray(decoded, dims=dims)
                del new_vars[key]
        return cls(data_vars=new_vars, coords=obj.coords)

    @classmethod
    def open_zarr(cls, store, **kwargs):
        ds = xr.open_zarr(store, **kwargs)
        encoding = ds.encoding
        obj = cls(data_vars=ds.data_vars, coords=ds.coords)
        obj = cls._decode_sparse_from_vars(obj)
        obj.encoding = encoding
        obj.attrs = ds.attrs
        return obj
        
    def append_zarr(self, **kwargs):
        """to_zarr but use the stored source path"""
        self.to_zarr(store=self.encoding['source'],**kwargs)
        
    #Use Xarray DataSet's built in to_zarr

    #TODO Implement open_s3_zarr if I want this later

    def unify_convert_chunks(self,out_path):
        '''Run xarray.DataSet.unify_chunks() and apply new chunks to each var'''
        new_self = self.unify_chunks()
        for k in new_self.keys():
            new_self[k] = new_self[k].chunk({k:new_self.chunks[k] for k in new_self[k].dims})
        new_self.to_zarr(out_path,mode='w')


def concat_grandata(
    adatas: Sequence[GRAnData],
    out_path,
    concat_dim: str = "var",
    join: str = "outer",
    add_key: str | None = None,
    add_values: Sequence[object] | None = None,
    shuffle: bool = False,
    random_state: int | None = None,
    consolidated: bool = False,
    mode: str = "w",
    **to_zarr_kwargs,
) -> GRAnData:
    """
    Concatenate one or more GRAnData objects, optionally annotate the concatenated
    axis with per-input labels, optionally shuffle that axis, write the merged
    zarr store, and reopen it as a backed GRAnData object.

    Parameters
    ----------
    adatas
        Input GRAnData objects. These may be backed objects from ``GRAnData.open_zarr``.
    out_path
        Output zarr store path.
    concat_dim
        Dimension along which to concatenate, e.g. ``"var"``.
    join
        Xarray alignment mode for non-concatenated dimensions. Must be ``"inner"``
        or ``"outer"``.
    add_key
        Optional variable name to add along ``concat_dim`` with one constant value
        per input dataset, e.g. ``"var-_-species"``.
    add_values
        Per-input values used for ``add_key``. Required when ``add_key`` is set.
    shuffle
        If True, shuffle the concatenated axis before writing.
    random_state
        Optional seed used when ``shuffle=True``.
    consolidated
        Passed through to the reopened ``GRAnData.open_zarr`` call.
    mode
        Write mode passed to ``to_zarr``.
    **to_zarr_kwargs
        Additional keyword arguments forwarded to ``to_zarr``.

    Returns
    -------
    GRAnData
        Reopened merged object backed by the written zarr store.
    """
    if not adatas:
        raise ValueError("adatas must contain at least one GRAnData object.")
    if join not in ("inner", "outer"):
        raise ValueError("join must be 'inner' or 'outer'.")
    if add_key is None and add_values is not None:
        raise ValueError("add_values may only be provided when add_key is set.")
    if add_key is not None:
        if add_values is None:
            raise ValueError("add_values is required when add_key is set.")
        if len(add_values) != len(adatas):
            raise ValueError("add_values must match the number of input GRAnData objects.")

    prepared = []
    for idx, adata in enumerate(adatas):
        if concat_dim not in adata.dims:
            raise ValueError(f"concat_dim {concat_dim!r} not found in input dataset {idx}.")

        ds = adata.copy()
        if add_key is not None:
            label = np.full(adata.sizes[concat_dim], add_values[idx], dtype=object)
            ds[add_key] = xr.DataArray(label, dims=(concat_dim,), coords={concat_dim: adata.coords[concat_dim]})
        prepared.append(ds)

    merged = xr.concat(prepared, dim=concat_dim, join=join)

    if shuffle:
        rng = np.random.default_rng(random_state)
        permutation = rng.permutation(merged.sizes[concat_dim])
        merged = merged.isel({concat_dim: permutation})

    merged_attrs = dict(merged.attrs)
    merged = GRAnData(data_vars=merged.data_vars, coords=merged.coords)
    merged.attrs = merged_attrs
    merged.to_zarr(store=out_path, mode=mode, **to_zarr_kwargs)
    return GRAnData.open_zarr(out_path, consolidated=consolidated)
