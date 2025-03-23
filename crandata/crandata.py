import xarray as xr
import pandas as pd
import numpy as np
import os
import json

try:
    import sparse  # for sparse multidimensional arrays
except ImportError:
    print("no sparse")
    sparse = None

class CrAnData(xr.Dataset):
    __slots__ = ("__dict__",)  # remove always_convert_df from slots

    def __init__(self, 
                 data_vars=None,
                 coords=None,
                 **kwargs):
        """
        Create a CrAnData object as a thin wrapper of xarray.Dataset.
        You can use get_dataframe('var') or 'obs' etc to get hierarchical variables
        as a grouped dataframe (the separator is -_- ...idk).
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
                # For each column in the DataFrame, create a new data variable using key "obs-_-<col>"
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
        # Gather all keys that start with "top-_-".
        for key in list(self.data_vars.keys()):
            if key.startswith(top + "-_-"):
                da = super().__getitem__(key)
                if expected is None:
                    expected = da.shape[0]
                elif da.shape[0] != expected:
                    continue  # skip keys with mismatched length
                col_name = key.split("-_-", 1)[1]
                cols[col_name] = da.values
        if cols:
            return pd.DataFrame(cols, index=np.arange(expected))
        else:
            return None
        
    def __repr__(self):
        rep = f"CrAnData object\nArray names: {self.array_names}\n"
        rep += f"Coordinates: {list(self.coords.keys())}\n"
        return rep
        
    def _repr_html_(self):
        return self.__repr__()
    
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
    def open_dataset(cls, path, **kwargs):
        ds = xr.open_dataset(path, **kwargs)
        encoding = ds.encoding
        obj = cls(data_vars=ds.data_vars, coords=ds.coords, always_convert_df=always_convert_df)
        obj = cls._decode_sparse_from_vars(obj)
        obj.encoding = encoding
        return obj

    @classmethod
    def open_zarr(cls, store, **kwargs):
        ds = xr.open_zarr(store, **kwargs)
        encoding = ds.encoding
        obj = cls(data_vars=ds.data_vars, coords=ds.coords)
        obj = cls._decode_sparse_from_vars(obj)
        obj.encoding = encoding
        return obj
