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
                 # always_convert_df=np.array(['']),  # list of top-level keys to be grouped into a DataFrame on access
                 **kwargs):
        """
        Create a CrAnData object as a subclass of xarray.Dataset.
        Instead of storing always_convert_df as a separate attribute,
        we store it in the data variables.
        If a DataFrame is provided for a key like "obs" or "var", we unpack it
        into separate hierarchical variables (e.g. "obs-_-col1", "obs-_-col2", etc.)
        so that later accessing adata.obs returns a grouped DataFrame.
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
        
        # Store always_convert_df as a data variable (wrapped in a 1D DataArray).
        # data_vars["always_convert_df"] = xr.DataArray(always_convert_df, dims=("item",))
        
        super().__init__(data_vars=data_vars, coords=coords)
        
        # Create shortcut attributes for keys that contain '/'.
        for key in self.data_vars:
            if "/" in key:
                safe_name = key.replace("/", "_")
                setattr(self, safe_name, self.data_vars[key])
                
        # If "var" (or "obs") is among the grouping keys, create the grouped DataFrame.
        # for group_key in always_convert_df.data:
        #     grouped = self.get_dataframe(group_key)
        #     if grouped is not None:
        #         setattr(self, group_key, grouped)

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
    
    # def __getitem__(self, key):
    #     # Bypass custom logic for the "always_convert_df" key.
    #     if key == "always_convert_df":
    #         return object.__getattribute__(self, "_variables")["always_convert_df"]
    #     if isinstance(key, str):
    #         if key in self.always_convert_df:
    #             df = self.get_dataframe(key)
    #             if df is not None:
    #                 return df
    #         # For hierarchical keys (like "obs-_-col") group them.
    #         if "-_-" in key:
    #             top, sub = key.split("-_-", 1)
    #             if top in self.always_convert_df:
    #                 df = self.get_dataframe(top)
    #                 if df is None:
    #                     raise KeyError(f"No grouped data found for key '{top}'")
    #                 return df[sub]
    #     return super().__getitem__(key)
    
    # def __getattr__(self, attr):
    #     # First, if attr is in always_convert_df, return its grouped DataFrame.
    #     acdf = object.__getattribute__(self, "always_convert_df")
    #     if attr in acdf:
    #         print(attr)
    #         df = self.get_dataframe(attr)
    #         if df is not None:
    #             return df
    #     # Otherwise, check if any data variable (with "-_-" replaced by "_") matches.
    #     dvars = object.__getattribute__(self, "data_vars")
    #     for key in dvars:
    #         if key.replace("/", "_") == attr:
    #             return dvars[key]
    #     raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")
    
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
        # # Preserve the stored always_convert_df using assign.
        # new_ds = new_ds.assign(always_convert_df=self.data_vars["always_convert_df"])
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
        # if "always_convert_df" in new_vars:
        #     arr = new_vars["always_convert_df"].data
        #     try:
        #         always_convert = arr.tolist()
        #     except AttributeError:
        #         always_convert = arr
        # else:
        #     always_convert = []
        return cls(data_vars=new_vars, coords=obj.coords)#, always_convert_df=always_convert)

    # @property
    # def always_convert_df(self):
    #     # Bypass __getitem__ by directly accessing the underlying _variables dict.
    #     d = object.__getattribute__(self, "_variables")
    #     arr = d["always_convert_df"]
    #     try:
    #         return arr.data.tolist()
    #     except AttributeError:
    #         return arr.data

    # @always_convert_df.setter
    # def always_convert_df(self, new_value):
    #     """
    #     Update the always_convert_df setting.
        
    #     This setter expects new_value to be a list. It creates a new DataArray
    #     for the provided list, updates the underlying storage, and then refreshes
    #     any external attributes (like adata.obs or adata.var) to reflect the new list.
    #     """
    #     new_list = list(new_value)
    #     new_da = xr.DataArray(new_list, dims=("item",))
    #     d = object.__getattribute__(self, "_variables")
    #     d["always_convert_df"] = new_da

    #     # Remove any previously set external attributes that are not in the new list.
    #     # (They were set during __init__ or via a previous call.)
    #     old_keys = set(self.__dict__.keys()).intersection(set(["obs", "var"]))
    #     for key in old_keys:
    #         if key not in new_list:
    #             try:
    #                 delattr(self, key)
    #             except Exception:
    #                 pass
    #     # Set (or update) attributes for each key in the new always_convert_df list.
    #     for key in new_list:
    #         grouped = self.get_dataframe(key)
    #         if grouped is not None:
    #             setattr(self, key, grouped)
                
    @classmethod
    def open_dataset(cls, path, **kwargs):
        ds = xr.open_dataset(path, **kwargs)
        encoding = ds.encoding
        # if "always_convert_df" in ds.data_vars:
        #     arr = ds.data_vars["always_convert_df"].data
        #     try:
        #         always_convert_df = arr.tolist()
        #     except AttributeError:
        #         always_convert_df = arr
        # else:
        #     always_convert_df = []
        obj = cls(data_vars=ds.data_vars, coords=ds.coords, always_convert_df=always_convert_df)
        obj = cls._decode_sparse_from_vars(obj)
        obj.encoding = encoding
        return obj

    @classmethod
    def open_zarr(cls, store, **kwargs):
        ds = xr.open_zarr(store, **kwargs)
        encoding = ds.encoding
        # if "always_convert_df" in ds.data_vars:
        #     arr = ds.data_vars["always_convert_df"].data
        #     try:
        #         always_convert_df = arr.tolist()
        #     except AttributeError:
        #         always_convert_df = arr
        # else:
        #     always_convert_df = []
        obj = cls(data_vars=ds.data_vars, coords=ds.coords)#, always_convert_df=always_convert_df)
        obj = cls._decode_sparse_from_vars(obj)
        obj.encoding = encoding
        return obj
