import numpy as np
import torch
import h5py
import xbatcher
import xarray as xr
import os
import zarr
from .crandata import CrAnData
from torchdata.nodes import BaseNode, Loader, IterableWrapper, ParallelMapper

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

class RoundRobinNode(BaseNode):
    def __init__(self, nodes, concat_dim='var', join='inner', num_workers=1):
        super().__init__()
        self.nodes = nodes
        self.concat_dim = concat_dim
        self.join = join
        self.num_workers = num_workers

    def reset(self, initial_state=None):
        super().reset(initial_state)
        for node in self.nodes:
            node.reset(initial_state)

    def get_state(self):
        return {i: node.get_state() for i, node in enumerate(self.nodes)}

    def _get_next_batch(self, node):
        # Attempt to fetch the next batch; if exhausted, reset and try again.
        try:
            return next(node)
        except StopIteration:
            node.reset()
            return next(node)

    def next(self):
        # Use ParallelMapper to apply _get_next_batch to each node concurrently.
        mapper = ParallelMapper(
            source=IterableWrapper(self.nodes),
            map_fn=self._get_next_batch,
            num_workers=self.num_workers,
            method="thread"
        )
        batches = list(mapper)
        return xr.concat(*batches, dim=self.concat_dim, join=self.join)

class SequentialNode(BaseNode):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.current = 0

    def reset(self, initial_state=None):
        super().reset(initial_state)
        for node in self.nodes:
            node.reset(initial_state)
        self.current = 0

    def get_state(self):
        return {
            "current": self.current,
            "states": [node.get_state() for node in self.nodes]
        }

    def next(self):
        # Loop until we find a node with data or we run out of nodes.
        while self.current < len(self.nodes):
            try:
                # Attempt to fetch the next batch from the current node.
                return next(self.nodes[self.current])
            except StopIteration:
                # If exhausted, move to the next node.
                self.current += 1
        raise StopIteration

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
    def __init__(self, adata, batch_size=32, shuffle=True, dnatransform=None, shuffle_dims=None,cache_dir = './batch_cache',data_sources={'sequences':'sequences'},split='var-_-split',batch_dim='var'):
        self.adata = adata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []  # e.g. ['obs']
        self._generators = {}  # Holds xbatcher generators for each state
        self.split = split
        self.batch_dim = batch_dim
        # self.cache_dir = str(cache_dir)
        self.state_instructions = {
            "train": {"apply_rc": True, "shuffle": True},
            "val": {"apply_rc": True, "shuffle": True},
            "test": {"apply_rc": False, "shuffle": False},
            "predict": {"apply_rc": False, "shuffle": False},
        }

    def setup(self, state="train"):
        """
        Set up the xbatcher generator for a given state.
        (It is assumed that the CrAnData object already has appropriate sample probabilities.)
        """
        dim_dict = dict(self.adata.dims)
        dim_dict[self.batch_dim] = self.batch_size
        dim_dict.pop('item', None)
        # os.makedirs(self.cache_dir,exist_ok=True)
        # cache_store = zarr.storage.LocalStore(self.cache_dir)
        self.adata = CrAnData.open_zarr(self.adata.encoding['source'])#Reload from disk for more robustness
        state_idx = self.adata[self.split].compute()==state
        print(state_idx)
        self._generators[state] = xbatcher.BatchGenerator(
            ds=self.adata.isel({self.batch_dim:state_idx}) if state != 'predict' else self.adata,
            input_dims=dim_dict,
            batch_dims={self.batch_dim: self.batch_size},
            cache=None #For now, idk might be useful later
        )

    def get_batch(self, state, index, instructions=None):
        if state not in self._generators:
            raise ValueError(f"Generator for state '{state}' not set. Call setup('{state}') first.")
        instructions = instructions or self.state_instructions.get(state, {})
        batch = self._generators[state][index]

        # Handle reverse complementing if needed.
        if instructions.get("apply_rc") and self.dnatransform is not None and 'sequences' in batch.data_vars:
            start, end, rc_flags = self.dnatransform.get_window_indices_and_rc(batch['sequences'])
            batch = batch.isel(seq_len=slice(start, end))
            batch = self.dnatransform.apply_rc(batch, rc_flags)

        # Handle shuffling if needed.
        if instructions.get("shuffle") and self.shuffle_dims:
            for dim in self.shuffle_dims:
                example = next((da for da in batch.data_vars.values() if dim in da.dims), None)
                if example is not None:
                    axis = example.get_axis_num(dim)
                    perm = np.random.permutation(example.shape[axis])
                    for var_name, da in batch.data_vars.items():
                        if dim in da.dims:
                            batch.data_vars[var_name] = da.isel({dim: perm})
        return batch


    @property
    def train_dataloader(self):
        return self._get_dataloader("train")

    @property
    def val_dataloader(self):
        return self._get_dataloader("val")

    @property
    def test_dataloader(self):
        return self._get_dataloader("test")

    @property
    def predict_dataloader(self):
        return self._get_dataloader("predict")

    @property
    def train_dataloader(self):
        node = CrAnDataNode(self, state="train")
        return Loader(node)
        
    def _get_dataloader(self, state: str):
        node = CrAnDataNode(self, state=state)
        return Loader(node)

    def __repr__(self):
        return (f"CrAnDataModule(batch_size={self.batch_size}, "
                f"shuffle_dims={self.shuffle_dims}, dnatransform={self.dnatransform})")

class MetaCrAnDataModule:
    """
    Combines multiple CrAnData objects (each wrapped in a CrAnDataModule) into a single meta–module.
    For a given state the dataloader uses MultiNodeWeightedSampler to sample from each module
    with a probability proportional to its file weight.
    
    Attributes:
      modules: List of CrAnDataModule instances.
      epoch_size: Number of batches per epoch (for reference).
    """
    def __init__(self, adatas, batch_size=256, shuffle=True, dnatransform=None, shuffle_dims=None, epoch_size=100000,data_sources={'sequences':'sequences'},batch_dim='var',num_workers=1,join='inner'):
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.modules = [
            CrAnDataModule(adata, batch_size=batch_size, shuffle=shuffle,
                           dnatransform=dnatransform, shuffle_dims=shuffle_dims, data_sources=data_sources)
            for adata in adatas
        ]
        self.epoch_size = epoch_size
        self.join = join
        self.num_workers = num_workers
        self.coords = xr.concat(adatas,batch_dim,join=self.join).coords

    def setup(self, state="train"):
        for mod in self.modules:
            mod.setup(state)
    
    @property
    def train_dataloader(self):
        nodes = [IterableWrapper(mod) for mod in self.modules]
        multi_node = RoundRobinNode(
            nodes, concat_dim=self.batch_dim, join=self.join, num_workers=self.num_workers
        )
        multi_node = CrAnDataNode(multi_node, state='train')
        return Loader(multi_node)

    @property
    def val_dataloader(self):
        nodes = [IterableWrapper(mod) for mod in self.modules]
        multi_node = SequentialNode(
            nodes,
        )
        multi_node = CrAnDataNode(multi_node, state='val')
        return Loader(multi_node)

    @property
    def test_dataloader(self):
        nodes = [IterableWrapper(mod) for mod in self.modules]
        multi_node = SequentialNode(
            nodes,
        )
        multi_node = CrAnDataNode(multi_node, state='test')
        return Loader(multi_node)

    @property
    def predict_dataloader(self):
        nodes = [IterableWrapper(mod) for mod in self.modules]
        multi_node = SequentialNode(
            nodes,
        )
        multi_node = CrAnDataNode(multi_node, state='predict')
        return Loader(multi_node)

    def __repr__(self):
        return (f"MetaCrAnDataModule(num_modules={len(self.modules)}, "
                f"epoch_size={self.epoch_size})")
