import numpy as np
import torch
import h5py
import xbatcher
import xarray as xr
import os
import zarr
from torchdata.nodes import BaseNode, Loader, MultiNodeWeightedSampler

def _reindex_array(array, local_obs, global_obs):
    """
    Given an array whose first dimension corresponds to local observations,
    create a new array with first dimension equal to len(global_obs). For observations
    present in local_obs, copy the values; for missing ones, fill with NaN.
    """
    new_shape = (len(global_obs),) + array.shape[1:]
    new_array = np.full(new_shape, np.nan, dtype=array.dtype)
    for i, obs in enumerate(local_obs):
        idx = np.where(global_obs == obs)[0]
        if idx.size:
            new_array[idx[0]] = array[i]
    return new_array

# ---------------------------------------------------------------------------
# New Node that wraps a CrAnDataModule for use with torchdata.nodes.
# ---------------------------------------------------------------------------
class CrAnDataNode(BaseNode):
    """
    A torchdata.nodes.BaseNode that iterates over batches produced by a CrAnDataModule.
    
    Parameters:
      module: A CrAnDataModule instance that has been set up for a given state.
      state: A string indicating the state ("train", "val", "test", or "predict").
    """
    def __init__(self, module, state="train"):
        self.module = module
        self.state = state
        if state not in self.module._generators:
            raise ValueError(f"Generator for state '{state}' not set. Call setup('{state}') first.")
        self.generator = self.module._generators[state]
        self.index = 0
        self.length = len(self.generator)

    def reset(self, initial_state=None):
        self.index = 0

    def __next__(self):
        # Wrap-around if we reach the end.
        if self.index >= self.length:
            self.index = 0
        batch = self.module.get_batch(self.state, self.index)
        self.index += 1
        return batch

    def get_state(self):
        return {"index": self.index}

# ---------------------------------------------------------------------------
# New CrAnDataModule using torchdata.nodes for dataloading.
# ---------------------------------------------------------------------------
class CrAnDataModule:
    """
    A module wrapping a CrAnData object (e.g. an xarray.Dataset with genomic data)
    that uses xbatcher for sampling. A DNATransform (if provided) is applied to the
    "sequences" key when batches are retrieved. Dataloaders are provided via torchdata.nodes.
    
    Attributes:
      adata: The CrAnData object.
      batch_size: Number of slices to sample along the "var" dimension.
      shuffle_dims: List of dimension names to be shuffled uniformly in each batch.
      dnatransform: Optional callable applied to the "sequences" variable.
    """
    def __init__(self, adata, batch_size=256, shuffle=True, dnatransform=None, shuffle_dims=None,cache_dir = './batch_cache',load_variables={'sequences':'sequences'}):
        self.adata = adata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []  # e.g. ['obs']
        self._generators = {}  # Holds xbatcher generators for each state
        # self.cache_dir = str(cache_dir)

    def setup(self, state="train"):
        """
        Set up the xbatcher generator for a given state.
        (It is assumed that the CrAnData object already has appropriate sample probabilities.)
        """
        dim_dict = dict(self.adata.dims)
        dim_dict['var'] = self.batch_size
        dim_dict.pop('item', None)
        # os.makedirs(self.cache_dir,exist_ok=True)
        # cache_store = zarr.storage.DirectoryStore(self.cache_dir)

        self._generators[state] = xbatcher.BatchGenerator(
            ds=self.adata,
            input_dims=dim_dict,
            batch_dims={'var': self.batch_size},
            cache=None #For now, idk might be useful later
        )

    def get_batch(self, state, index):
        if state not in self._generators:
            raise ValueError(f"Generator for state '{state}' not set. Call setup('{state}') first.")
        batch = self._generators[state][index]
        if self.dnatransform is not None and 'sequences' in batch.data_vars:
            start, end, rc_flags = self.dnatransform.get_window_indices_and_rc(batch['sequences'])
            subset_seq = batch['sequences'].isel(seq_len=slice(start, end))
            batch = batch.isel(seq_len=slice(start, end))
            transformed = self.dnatransform.apply_rc(batch, rc_flags)
    
        if self.shuffle and self.shuffle_dims:
            for dim in self.shuffle_dims:
                example = next((da for da in batch.data_vars.values() if dim in da.dims), None)
                if example is not None:
                    axis = example.get_axis_num(dim)
                    perm = np.random.permutation(example.shape[axis])
                    for var_name, da in batch.data_vars.items():
                        if dim in da.dims:
                            batch.data_vars[var_name] = da.isel({dim: perm})
        return batch

    # -------------------------------------------------------------------------
    # Dataloader properties returning torchdata Loaders.
    # -------------------------------------------------------------------------
    @property
    def train_dataloader(self):
        node = CrAnDataNode(self, state="train")
        return Loader(node)

    @property
    def val_dataloader(self):
        node = CrAnDataNode(self, state="val")
        return Loader(node)

    @property
    def test_dataloader(self):
        node = CrAnDataNode(self, state="test")
        return Loader(node)

    @property
    def predict_dataloader(self):
        node = CrAnDataNode(self, state="predict")
        return Loader(node)

    def __repr__(self):
        return (f"CrAnDataModule(batch_size={self.batch_size}, "
                f"shuffle_dims={self.shuffle_dims}, dnatransform={self.dnatransform})")

# ---------------------------------------------------------------------------
# New MetaCrAnDataModule using torchdata.nodes.MultiNodeWeightedSampler.
# ---------------------------------------------------------------------------
class MetaCrAnDataModule:
    """
    Combines multiple CrAnData objects (each wrapped in a CrAnDataModule) into a single meta–module.
    For a given state the dataloader uses MultiNodeWeightedSampler to sample from each module
    with a probability proportional to its file weight.
    
    Attributes:
      modules: List of CrAnDataModule instances.
      file_probs: Per–module sampling probabilities (default: uniform).
      epoch_size: Number of batches per epoch (for reference).
    """
    def __init__(self, adatas, batch_size=256, shuffle=True, dnatransform=None, shuffle_dims=None, epoch_size=100000,file_probs = None):
        self.modules = [
            CrAnDataModule(adata, batch_size=batch_size, shuffle=shuffle,
                           dnatransform=dnatransform, shuffle_dims=shuffle_dims)
            for adata in adatas
        ]
        self.file_probs = np.ones(len(self.modules)) / len(self.modules) if file_probs is None else file_probs
        self.epoch_size = epoch_size

    @property
    def train_dataloader(self):
        nodes = {f"module_{i}": CrAnDataNode(mod, state="train")
                 for i, mod in enumerate(self.modules)}
        weights = {f"module_{i}": self.file_probs[i] for i in range(len(self.modules))}
        multi_node = MultiNodeWeightedSampler(nodes, weights)
        return Loader(multi_node)

    @property
    def val_dataloader(self):
        nodes = {f"module_{i}": CrAnDataNode(mod, state="val")
                 for i, mod in enumerate(self.modules)}
        weights = {f"module_{i}": self.file_probs[i] for i in range(len(self.modules))}
        multi_node = MultiNodeWeightedSampler(nodes, weights)
        return Loader(multi_node)

    @property
    def test_dataloader(self):
        nodes = {f"module_{i}": CrAnDataNode(mod, state="test")
                 for i, mod in enumerate(self.modules)}
        weights = {f"module_{i}": self.file_probs[i] for i in range(len(self.modules))}
        multi_node = MultiNodeWeightedSampler(nodes, weights)
        return Loader(multi_node)

    @property
    def predict_dataloader(self):
        nodes = {f"module_{i}": CrAnDataNode(mod, state="predict")
                 for i, mod in enumerate(self.modules)}
        weights = {f"module_{i}": self.file_probs[i] for i in range(len(self.modules))}
        multi_node = MultiNodeWeightedSampler(nodes, weights)
        return Loader(multi_node)

    def __repr__(self):
        return (f"MetaCrAnDataModule(num_modules={len(self.modules)}, "
                f"epoch_size={self.epoch_size})")
