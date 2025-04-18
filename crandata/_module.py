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
    Applies state‐specific postprocessing (reverse‐complement + shuffling) to xarray.Dataset batches.
    """
    def __init__(self, state="train", instructions=None, dnatransform=None, shuffle_dims=None,sequence_vars=None):
        self.state = state
        self.instructions = instructions or {}
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []
        self.sequence_vars = sequence_vars or ["sequences"]

    def __call__(self, batch: xr.Dataset) -> xr.Dataset:
        # 1) Reverse‐complement windowing
        if self.instructions.get("apply_rc") and self.dnatransform is not None:
            # find first available sequence var to compute window+flags
            for key in self.sequence_vars:
                if key in batch.data_vars:
                    start, end, rc_flags = self.dnatransform.get_window_indices_and_rc(batch[key])
                    # slice all
                    batch = batch.isel(seq_len=slice(start, end))
                    # apply RC individually to each seq‐var present
                    for k in self.sequence_vars:
                        if k in batch.data_vars:
                            batch[k] = self.dnatransform.apply_rc(batch[k], rc_flags)
                    break

        # 2) Shuffle entire dataset along each shuffle_dim in one shot
        if self.instructions.get("shuffle") and self.shuffle_dims:
            indexers = {}
            for dim in self.shuffle_dims:
                if dim in batch.dims:
                    size = batch.sizes[dim]
                    indexers[dim] = np.random.permutation(size)
            if indexers:
                # xarray will apply each indexer to all arrays that share that dim
                batch = batch.isel(indexers)

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
    def __init__(
        self,
        adata,
        batch_size: int = 32,
        load_keys: dict[str,str] = {'sequences':'sequences'},
        shuffle: bool = False,
        dnatransform=None,
        shuffle_dims: list[str] | None = None,
        split: str = 'var-_-split',
        batch_dim: str = 'var',
        sequence_vars: list[str] = ['sequences'],
        prefetch_factor: int = 2,
        pin_memory: bool | None = None,
    ):
        self.adata = adata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []
        self.load_keys = load_keys
        self.split = split
        self.batch_dim = batch_dim
        self.sequence_vars = sequence_vars
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        self.instructions = {
            "train":  {"apply_rc": True,  "shuffle": shuffle},
            "val":    {"apply_rc": True,  "shuffle": shuffle},
            "test":   {"apply_rc": False, "shuffle": False},
            "predict":{"apply_rc": False, "shuffle": False},
        }

        self._generators: dict[str, xbatcher.BatchGenerator] = {}

    def setup(self, state="train"):
        # same as before…
        state_idx = self.adata[self.split].compute() == state
        ds = self.adata.isel({self.batch_dim: state_idx}) if state != 'predict' else self.adata
        ds = ds[list(self.load_keys.keys())]
        dims = dict(ds.sizes)
        dims[self.batch_dim] = self.batch_size

        self._generators[state] = xbatcher.BatchGenerator(
            ds=ds,
            input_dims=dims,
            batch_dims={self.batch_dim: self.batch_size},
            cache=None,
        )

    def _get_dataloader(self, state: str) -> Loader:
        source = IterableWrapper(self._generators[state])

        transform = DNATransformApplicator(
            state=state,
            instructions=self.instructions,
            dnatransform=self.dnatransform,
            shuffle_dims=self.shuffle_dims,
            sequence_vars=self.sequence_vars,
        )
        converter = TensorConverter(self.load_keys)
        node = Mapper(node, map_fn=lambda x: converter(transform(x)))

        node = Prefetcher(node, prefetch_factor=self.prefetch_factor)
        if self.pin_memory:
            node = PinMemory(node, self.pin_memory)

        return Loader(node)

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
        self.adata.load()

    def __repr__(self):
        return (
            f"CrAnDataModule(batch_size={self.batch_size}, "
            f"shuffle_dims={self.shuffle_dims}, dnatransform={self.dnatransform})"
        )


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
        sequence_vars=['sequences'],
        join='inner',
        num_workers=0,
        pin_memory=None,
        prefetch_factor=2
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
        self.sequence_vars = sequence_vars
        self.prefetch_factor=prefetch_factor
        # Wrap each CrAnData in its own CrAnDataModule
        self.modules = [
            CrAnDataModule(
                adata,
                batch_size=batch_size,
                shuffle=shuffle,
                dnatransform=dnatransform,
                shuffle_dims=shuffle_dims,
                load_keys=load_keys,
                sequence_vars=sequence_vars
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
        if join not in ('inner', 'outer'):
            raise ValueError("`join` must be either 'inner' or 'outer'")

        # collect all dims present in any dataset
        all_dims = set().union(*(set(ds.coords) for ds in adatas))
        self.global_coords = {}

        for dim in all_dims:
            if dim == batch_dim:
                continue

            # gather each dataset's coordinate array for this dim (if present)
            coord_lists = [ds.coords[dim].values for ds in adatas if dim in ds.coords]
            if not coord_lists:
                continue  # nothing to do
            if join == 'outer':
                # union of all coordinate values
                vals = set().union(*(set(vals) for vals in coord_lists))
            else:  # join == 'inner'
                iter_sets = [set(vals) for vals in coord_lists]
                vals = iter_sets[0].intersection(*iter_sets[1:])
            self.global_coords[dim] = np.array(sorted(vals))
    
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
        transform_applicator = DNATransformApplicator(
            state=state,
            instructions=self.instructions,
            dnatransform=self.dnatransform,
            shuffle_dims=self.shuffle_dims,
            sequence_vars=self.sequence_vars
        )
        converter = TensorConverter(self.load_keys)
        node = Mapper(sampler, 
                      map_fn=lambda x: converter(transform_applicator(reindex_fn(x))))
        node = Prefetcher(node,prefetch_factor=self.prefetch_factor)
        if self.pin_memory is not None:
            node = PinMemory(node,self.pin_memory)
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
            shuffle_dims=self.shuffle_dims,
            sequence_vars=self.sequence_vars
        )
        node = Mapper(multi_node, map_fn=transform_applicator)
        converter = TensorConverter(self.load_keys)
        node = Mapper(node, map_fn=converter)
        return Loader(node)

    def __repr__(self):
        return (f"MetaCrAnDataModule(num_modules={len(self.modules)}, "
                f"batch_dim={self.batch_dim})")
