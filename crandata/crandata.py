import xarray as xr
import pandas as pd
import numpy as np
import os
import json
import zarr
import icechunk as ic
from icechunk.xarray import to_icechunk

try:
    import sparse  # for sparse multidimensional arrays
except ImportError:
    print("no sparse")
    sparse = None

class CrAnData(xr.Dataset):
    __slots__ = ("__dict__","session","repo")  # remove always_convert_df from slots

    def __init__(self, 
                 data_vars=None,
                 coords=None,
                 session=None,
                 **kwargs):
        """
        Create a CrAnData object as a thin wrapper of xarray.Dataset.
        You can use get_dataframe('var') or 'obs' etc to get hierarchical variables
        grouped into dataframe (the separator is -_- ...idk).
        Also automatically stores and reloads sparse arrays.
        """
        self.session = None
        self.repo = None
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
        attrs = ds.attrs
        obj = cls(data_vars=ds.data_vars, coords=ds.coords)
        obj = cls._decode_sparse_from_vars(obj)
        obj.encoding = encoding
        obj.attrs = attrs
        return obj
        
    def append_zarr(self, **kwargs):
        """to_zarr but use the stored source path"""
        self.to_zarr(store=self.encoding['source'],**kwargs)
        
    #Use Xarray DataSet's built in to_zarr

    @classmethod
    def open_icechunk(cls, store, cache_config={'num_bytes_chunks':int(1e9)}, **kwargs):
        '''store must be either path to zarr store or icechunk repo'''
        if isinstance(store, (str, os.PathLike)):
            store_path = store
            storage_config = ic.local_filesystem_storage(store_path)
            config = ic.RepositoryConfig.default()
            config.caching = ic.CachingConfig(**cache_config)
            if not ic.Repository.exists(storage_config):
                repo = ic.Repository.create(storage_config,config)
            else:
                repo = ic.Repository.open(storage_config, config)
        else:
            repo = store
        session = repo.readonly_session("main")
            # store_path = session.store.attrs["source"] #TODO test this = it doesn't work
        ds = xr.open_zarr(session.store, consolidated=False, **kwargs)
        # Save the store path in the attributes or as a separate property.
        # ds.attrs["source"] = store_path
        obj = cls(data_vars=ds.data_vars, coords=ds.coords)
        obj.encoding = ds.encoding
        obj.attrs = ds.attrs
        obj.session = session
        obj.repo = repo
        return obj

    def make_write_session(self):
        self.session = self.repo.writable_session("main")
    
    def to_icechunk(self,store=None,commit_name="commit_", cache_config={}, **kwargs):
        '''store must be either path to zarr store or icechunk repo, not technically an icechunk store (ik, confusing)'''
        if store is not None:
            if isinstance(store, (str, os.PathLike)):
                store_path = store
                storage_config = ic.local_filesystem_storage(store_path)
                config = ic.RepositoryConfig.default()
                config.caching = ic.CachingConfig(**cache_config)
                if not ic.Repository.exists(storage_config):
                    self.repo = ic.Repository.create(storage_config,config)
                else:
                    self.repo = ic.Repository.open(storage_config, config)
            else:
                self.repo = store
        write_session = self.repo.writable_session("main")
        to_icechunk(self,write_session,**kwargs)
        write_session.commit(commit_name)
        self.session = self.repo.readonly_session("main")
    #TODO Implement open_s3_zarr if I want this later

    def unify_convert_chunks(self,out_path):
        '''Run xarray.DataSet.unify_chunks() and apply new chunks to each var'''
        new_self = self.unify_chunks()
        for k in new_self.keys():
            new_self[k] = new_self[k].chunk({k:new_self.chunks[k] for k in new_self[k].dims})
        if '.icechunk' in out_path:
            new_self.to_icechunk(out_path,mode='w')
        else:
            new_self.to_zarr(out_path,mode='w')
