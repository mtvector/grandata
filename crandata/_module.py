import numpy as np
import xbatcher
import xarray as xr
import os
import zarr
from .crandata import CrAnData
from torchdata.nodes import BaseNode, Loader, Mapper, IterableWrapper, ParallelMapper,Prefetcher,PinMemory,MultiNodeWeightedSampler

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

class TensorConverter:
    """
    
    Parameters
    ----------
    load_keys : dict
        Mapping of original data variable names to new names.
    as_numpy : bool, optional
        If True, converts variables to numpy arrays; otherwise converts to torch.Tensor.
        (Default is False.)
    """
    def __init__(self, load_keys, as_numpy=True):
        self.load_keys = load_keys
        self.as_numpy = as_numpy
        if not as_numpy:
            import torch

    def __call__(self,batch):
        out = {}
        for key, val in self.load_keys.items():
            arr = batch[key].values
            out[val] = arr if self.as_numpy else torch.as_tensor(arr)
        return out

class DNATransformApplicator:
    """
    A wrapper node that applies state-specific postprocessing to batches
    produced by an upstream node. The upstream node is expected to be a torchdata.nodes.BaseNode 
    (or wrapped via IterableWrapper) that yields raw batches. CrAnDataNode applies the processing
    (e.g. reverse complementing and shuffling) based on the given state instructions.
    """
    def __init__(self, state="train", instructions=None, dnatransform=None, shuffle_dims=None):
        self.state = state
        self.instructions = instructions or {}
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []

    def __call__(self,batch):
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

class SequentialNode(BaseNode):
    """
    Sequentially iterates over a list of nodes. Moves to the next node once the current is exhausted.
    """
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
        while self.current < len(self.nodes):
            try:
                return next(self.nodes[self.current])
            except StopIteration:
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
        # self.adata = CrAnData.open_zarr(self.adata.repo)#Do this outside
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
        converter = TensorConverter(self.load_keys)
        node = Mapper(node,map_fn=converter)
        # node = Prefetcher(node, prefetch_factor=4)
        return Loader(node)

    def __repr__(self):
        return (f"CrAnDataModule(batch_size={self.batch_size}, "
                f"shuffle_dims={self.shuffle_dims}, dnatransform={self.dnatransform})")


class MetaCrAnDataModule:
    def __init__(
        self,
        adatas,
        batch_size,
        weights=None,
        load_keys={'sequences': 'sequences'},
        shuffle=False,
        dnatransform=None,
        shuffle_dims=None,
        epoch_size=100000,
        batch_dim='var',
        join='inner',
        num_workers=0,
        pin_memory=None,
    ):
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []
        self.load_keys = load_keys
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Wrap each CrAnData in its own CrAnDataModule
        self.modules = [
            CrAnDataModule(
                adata,
                batch_size=batch_size,
                shuffle=shuffle,
                dnatransform=dnatransform,
                shuffle_dims=shuffle_dims,
                load_keys=load_keys
            )
            for i, adata in enumerate(adatas)
        ]

        # --- 1. Prepare weights for WeightedSampler ---
        if weights is None:
            # equal probability if not specified
            w = 1.0 / len(self.modules)
            weights = [w] * len(self.modules)
        if len(weights) != len(self.modules):
            raise ValueError("`weights` must have same length as `adatas`")
        # map integer index → weight
        self.weight_map = {i: float(weights[i]) for i in range(len(weights))}

        # --- 2. Compute unioned coordinates across all adatas ---
        # gather all dimension names
        if join == 'inner':
            all_dims = set().intersection(*(set(ds.coords) for ds in adatas))
        else:
            all_dims = set().union(*(set(ds.coords) for ds in adatas))
        self.global_coords = {}
        for dim in all_dims:
            if dim != batch_dim:
                # collect only those adatas that actually have this dim
                arrays = [ds.coords[dim].values for ds in adatas if dim in ds.coords]
                # take unique union
                union_vals = np.unique(np.concatenate(arrays))
                self.global_coords[dim] = union_vals
            else:
                print(batch_dim)
    
        # State‐specific reverse‐complement / shuffle instructions:
        self.instructions = {
            "train": {"apply_rc": True,  "shuffle": shuffle},
            "val":   {"apply_rc": True,  "shuffle": shuffle},
            "test":  {"apply_rc": False, "shuffle": False},
            "predict":{"apply_rc": False,"shuffle": False},
        }

    def setup(self, state="train"):
        for mod in self.modules:
            mod.setup(state)

    def _get_weighted_dataloader(self, state: str):
        """
        Uses MultiNodeWeightedSampler to draw batches from each submodule
        according to self.weight_map, then reindexes to global coords,
        applies DNATransform, and finally converts to tensors.
        """
        node_map = {
            i: IterableWrapper(mod._generators[state])
            for i, mod in enumerate(self.modules)
        }
        sampler = MultiNodeWeightedSampler(node_map, self.weight_map)

        reindex_fn = lambda ds: ds.reindex(self.global_coords, fill_value=np.nan)
        node = Mapper(sampler, map_fn=reindex_fn)

        transform_applicator = DNATransformApplicator(
            state=state,
            instructions=self.instructions,
            dnatransform=self.dnatransform,
            shuffle_dims=self.shuffle_dims
        )
        node = Mapper(node, map_fn=transform_applicator)

        converter = TensorConverter(self.load_keys)
        node = Mapper(node, map_fn=converter)
        # node = Prefetcher(node,prefetch_factor=4)
        if self.pin_memory is not None:
            PinMemory(node,self.pin_memory)
        return Loader(node)

    @property
    def train_dataloader(self):
        return self._get_weighted_dataloader("train")

    @property
    def val_dataloader(self):
        # unchanged: sequential
        return self._get_sequential_dataloader("val")

    @property
    def test_dataloader(self):
        return self._get_sequential_dataloader("test")

    @property
    def predict_dataloader(self):
        return self._get_sequential_dataloader("predict")

    def _get_sequential_dataloader(self, state: str):
        # fallback for val/test/predict using simple round‐robin or sequential logic
        nodes = [IterableWrapper(mod._generators[state]) for mod in self.modules]
        multi_node = SequentialNode(nodes)
        # then same transform + conversion
        transform_applicator = DNATransformApplicator(
            state=state,
            instructions=self.instructions,
            dnatransform=self.dnatransform,
            shuffle_dims=self.shuffle_dims
        )
        node = Mapper(multi_node, map_fn=transform_applicator)
        converter = TensorConverter(self.load_keys)
        node = Mapper(node, map_fn=converter)
        return Loader(node)

    def __repr__(self):
        return (f"MetaCrAnDataModule(num_modules={len(self.modules)}, "
                f"batch_dim={self.batch_dim})")
