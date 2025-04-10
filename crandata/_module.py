import numpy as np
import xbatcher
import xarray as xr
import os
import zarr
from .crandata import CrAnData
from torchdata.nodes import BaseNode, Loader, IterableWrapper, ParallelMapper#,Prefetcher


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

class TensorConversionNode(BaseNode):
    """
    A node that converts an upstream xarray.Dataset batch into a dictionary of tensors.
    
    This node renames the keys of the dataset's data variables according to a provided mapping 
    (load_keys) and converts each variable into either a torch.Tensor or a numpy array.
    This is useful for interfacing with neural networks that expect specific keyword argument names.
    
    Parameters
    ----------
    upstream : BaseNode
        The upstream node producing xarray.Dataset batches.
    load_keys : dict
        Mapping of original data variable names to new names.
    as_numpy : bool, optional
        If True, converts variables to numpy arrays; otherwise converts to torch.Tensor.
        (Default is False.)
    """
    def __init__(self, upstream, load_keys, as_numpy=True):
        super().__init__()
        self.upstream = upstream
        self.load_keys = load_keys
        self.as_numpy = as_numpy
        if not as_numpy:
            import torch

    def reset(self, initial_state=None):
        self.upstream.reset(initial_state)

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.upstream)
        out = {}
        for key, val in self.load_keys.items():
            arr = batch[key].values
            out[val] = arr if self.as_numpy else torch.as_tensor(arr)
        return out

    def get_state(self):
        return self.upstream.get_state()

class CrAnDataNode(BaseNode):
    """
    A wrapper node that applies state-specific postprocessing to batches
    produced by an upstream node. The upstream node is expected to be a torchdata.nodes.BaseNode 
    (or wrapped via IterableWrapper) that yields raw batches. CrAnDataNode applies the processing
    (e.g. reverse complementing and shuffling) based on the given state instructions.
    """
    def __init__(self, upstream, state="train", instructions=None, dnatransform=None, shuffle_dims=None):
        super().__init__()
        # upstream must be a node; if you have a module, wrap it with IterableWrapper first.
        self.upstream = upstream  
        self.state = state
        # instructions (a dict) controls state-specific behavior
        self.instructions = instructions or {}
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []

    def reset(self, initial_state=None):
        self.upstream.reset(initial_state)

    def __iter__(self):
        return self

    def __next__(self):
        # Get a batch from the upstream node
        batch = next(self.upstream)

        # Apply reverse complement processing if needed.
        if self.instructions.get("apply_rc") and self.dnatransform is not None and 'sequences' in batch.data_vars:
            start, end, rc_flags = self.dnatransform.get_window_indices_and_rc(batch['sequences'])
            batch = batch.isel(seq_len=slice(start, end))
            batch = self.dnatransform.apply_rc(batch, rc_flags)

        # Apply shuffling if needed.
        if self.instructions.get("shuffle") and self.shuffle_dims:
            for dim in self.shuffle_dims:
                example = next((da for da in batch.data_vars.values() if dim in da.dims), None)
                if example is not None:
                    axis = example.get_axis_num(dim)
                    perm = np.random.permutation(example.shape[axis])
                    for var_name, da in batch.data_vars.items():
                        if dim in da.dims:
                            batch.data_vars[var_name] = da.isel({dim: perm})
        return batch

    def get_state(self):
        return self.upstream.get_state()

class RoundRobinNode(BaseNode):
    def __init__(self, nodes, concat_dim='var', join='inner', num_workers=0):
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
        return xr.concat(batches, dim=self.concat_dim, join=self.join)

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
    def __init__(self, adata, batch_size=32, load_keys={'sequences':'sequences'}, shuffle=False, dnatransform=None, shuffle_dims=None,cache_dir = './batch_cache',split='var-_-split',batch_dim='var'):
        self.adata = adata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []  # e.g. ['obs']
        self.load_keys = load_keys
        self._generators = {}  # Holds xbatcher generators for each state
        self.split = split
        self.batch_dim = batch_dim
        # self.cache_dir = str(cache_dir)
        self.instructions = {
            "train": {"apply_rc": True, "shuffle": shuffle},
            "val": {"apply_rc": True, "shuffle": shuffle},
            "test": {"apply_rc": False, "shuffle": False},
            "predict": {"apply_rc": False, "shuffle": False},
        }

    def setup(self, state="train"):
        """
        Set up the xbatcher generator for a given state.
        (It is assumed that the CrAnData object already has appropriate sample probabilities.)
        """
        print(self.adata,dir(self.adata))
        # self.adata = CrAnData.open_zarr(self.adata.repo)#Reload from disk for sync
        state_idx = self.adata[self.split].compute()==state
        loading_adata = self.adata.isel({self.batch_dim:state_idx}) if state != 'predict' else self.adata
        cur_keys = list(self.load_keys.keys())
        loading_adata = loading_adata[cur_keys]
        dim_dict = dict(loading_adata.sizes) #for futurewarning
        dim_dict[self.batch_dim] = self.batch_size
        # os.makedirs(self.cache_dir,exist_ok=True)
        self._generators[state] = xbatcher.BatchGenerator(
            ds=loading_adata,
            input_dims=dim_dict,
            batch_dims={self.batch_dim: self.batch_size},
            cache=None #For now, idk might be useful later
        )

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

    def load(self):
        """load module's dataset to memory"""
        self.adata.load()
        
    def _get_dataloader(self, state: str):
        node = CrAnDataNode(self._generators[state], state=state,
                                 instructions=self.instructions, dnatransform=self.dnatransform, 
                                 shuffle_dims=self.shuffle_dims)
        node = TensorConversionNode(node, self.load_keys)
        # node = Prefetcher(node, prefetch_factor=4)
        return Loader(node)

    def __repr__(self):
        return (f"CrAnDataModule(batch_size={self.batch_size}, "
                f"shuffle_dims={self.shuffle_dims}, dnatransform={self.dnatransform})")

class MetaCrAnDataModule:
    """
    Combines multiple CrAnData objects into a unified meta-module that aggregates data from
    several CrAnDataModule instances. This class creates composite dataloaders using torchdata.nodes,
    allowing for coordinated sampling and batch processing across multiple datasets.

    Parameters
    ----------
    adatas : list
        A list of CrAnData objects to be wrapped into individual CrAnDataModule instances.
    batch_size : list
        Batch size from each dataset (they get concatenated together per batch).
    shuffle : bool, optional
        Whether to shuffle the data (default is False).
    dnatransform : callable, optional
        Applies reverse complement transformations on sequence data.
    shuffle_dims : list, optional
        List of dimension names along which to perform shuffling (e.g. ['obs']).
    epoch_size : int, optional
        Number of batches per epoch for reference (default is 100000).
    load_keys : dict, optional
        Mapping of logical data source names to variable names (default is {'sequences': 'sequences'}).
    batch_dim : str, optional
        The name of the dimension along which batches are concatenated (default is 'var').
    num_workers : int, optional
        Number of worker threads for parallel batch processing (default is 1).
    join : str, optional
        The method to join data arrays from different modules (e.g. "inner"; default is 'inner').

    Attributes
    ----------
    modules : list
        List of CrAnDataModule instances, one for each provided CrAnData object.
    coords : xarray.core.coords.Coordinates
        Coordinates obtained by concatenating the input adatas along the specified batch dimension.
    instructions : dict
        A dictionary of state-specific processing instructions that control whether to apply
        reverse complementing ('apply_rc') and shuffling ('shuffle') for each state.

    Methods
    -------
    setup(state: str)
        Initializes the xbatcher generators for each submodule for the given state.
    train_dataloader, val_dataloader, test_dataloader, predict_dataloader : property
    """
    def __init__(self, adatas, batch_size, load_keys={'sequences':'sequences'}, shuffle=False, dnatransform=None, shuffle_dims=None, epoch_size=100000,batch_dim='var',num_workers=0,join='inner'):
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []  # e.g. ['obs']
        self.load_keys = load_keys
        self.modules = [
            CrAnDataModule(adata, batch_size=batch_size[i], shuffle=shuffle,
                           dnatransform=dnatransform, shuffle_dims=shuffle_dims, load_keys=load_keys)
            for i,adata in enumerate(adatas)
        ]
        self.epoch_size = epoch_size
        self.join = join
        self.num_workers = num_workers
        self.coords = xr.concat(adatas,batch_dim,join=self.join).coords
        self.instructions = {
            "train": {"apply_rc": True, "shuffle": shuffle},
            "val": {"apply_rc": True, "shuffle": shuffle},
            "test": {"apply_rc": False, "shuffle": False},
            "predict": {"apply_rc": False, "shuffle": False},
        }


    def setup(self, state="train"):
        for mod in self.modules:
            mod.setup(state)
    
    def _get_dataloader(self, state: str, node_type):
        # Create a list of nodes (wrapped generators) for the given state.
        nodes = [IterableWrapper(mod._generators[state]) for mod in self.modules]
        # Depending on the type of aggregation, create the multi-node.
        if node_type is RoundRobinNode:
            multi_node = RoundRobinNode(
                nodes, concat_dim=self.batch_dim, join=self.join, num_workers=self.num_workers
            )
        else:
            multi_node = node_type(nodes)
        # Wrap with the CrAnDataNode that applies state–dependent processing.
        multi_node = CrAnDataNode(
            multi_node, state=state,
            instructions=self.instructions, dnatransform=self.dnatransform,
            shuffle_dims=self.shuffle_dims
        )
        multi_node = TensorConversionNode(multi_node, self.load_keys)
        return Loader(multi_node)

    def load(self):
        """load all modules' datasets to memory"""
        for m in self.modules:
            m.adata.load()
    
    @property
    def train_dataloader(self):
        return self._get_dataloader("train", RoundRobinNode)

    @property
    def val_dataloader(self):
        return self._get_dataloader("val", SequentialNode)

    @property
    def test_dataloader(self):
        return self._get_dataloader("test", SequentialNode)

    @property
    def predict_dataloader(self):
        return self._get_dataloader("predict", SequentialNode)

    def __repr__(self):
        return (f"MetaCrAnDataModule(num_modules={len(self.modules)}, "
                f"epoch_size={self.epoch_size})")
