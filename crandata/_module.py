import numpy as np
import xbatcher
import xarray as xr
from torchdata.nodes import (
    BaseNode, Loader, ParallelMapper, IterableWrapper,
    Prefetcher, PinMemory, MultiNodeWeightedSampler
)

class TensorConverter:
    '''Converter from xarray.Dataset → dict of numpy/torch arrays'''
    def __init__(self, load_keys: dict[str,str], as_numpy: bool = True):
        self.load_keys = load_keys
        self.as_numpy = as_numpy
        if not as_numpy:
            import torch
            self.torch = torch

    def __call__(self, batch: xr.Dataset) -> dict[str, np.ndarray]:
        out = {}
        for xr_key, tensor_key in self.load_keys.items():
            arr = batch[xr_key].values
            if self.as_numpy:
                out[tensor_key] = arr
            else:
                out[tensor_key] = self.torch.as_tensor(arr)
        return out

class DNATransformApplicator:
    '''Reverse‐complement + shuffle applicator, handling multiple sequence vars'''
    def __init__(
        self,
        state: str,
        instructions: dict[str,dict[str,bool]],
        dnatransform,
        shuffle_dims: list[str] | None = None,
        sequence_vars: list[str] | None = None,
    ):
        self.state = state
        self.instructions = instructions
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []
        self.sequence_vars = sequence_vars or ["sequences"]

    def __call__(self, batch: xr.Dataset) -> xr.Dataset:
        # -- 1) reverse‐complement windowing on all sequence_vars present --
        if self.instructions[self.state]["apply_rc"] and self.dnatransform:
            # pick any one seq var to compute window+rc_flags
            for key in self.sequence_vars:
                if key in batch.data_vars:
                    start, end, rc_flags = self.dnatransform.get_window_indices_and_rc(batch[key])
                    batch = batch.isel(seq_len=slice(start, end))
                    # apply rc_flags to each seq var
                    for sv in self.sequence_vars:
                        if sv in batch.data_vars:
                            batch[sv] = self.dnatransform.apply_rc(batch[sv], rc_flags)
                    break

        # -- 2) shuffle any requested dims in one shot --
        if self.instructions[self.state]["shuffle"] and self.shuffle_dims:
            idx = {}
            for dim in self.shuffle_dims:
                if dim in batch.dims:
                    idx[dim] = np.random.permutation(batch.sizes[dim])
            if idx:
                batch = batch.isel(idx)
        return batch

class CrAnDataModule:
    """
    A unified data‐loading module for one or more CrAnData xarray.Datasets that produces
    PyTorch‐compatible dataloaders for training, validation, testing, and prediction.

    This class supports:
      - Sampling from multiple datasets with optional weighting (for meta‐datasets).
      - On‐the‐fly reverse‐complement and windowing of one‐hot DNA sequence arrays.
      - Random shuffling along arbitrary dataset dimensions (e.g. obs).
      - Coordinate unification (inner or outer join) across multiple datasets.
      - Asynchronous (I think?) prefetching and optional pinning of CPU memory.

    Parameters
    ----------
    adatas : Sequence[CrAnData]
        One or more CrAnData objects to draw samples from. If a single dataset is provided,
        it will behave like a standard DataModule; if multiple, samples are drawn according
        to `weights` and mixed at each batch.
    batch_size : int
        Number of slices along the `batch_dim` (e.g. "var") to include in each batch.
    weights : Optional[Sequence[float]]
        Sampling probability for each dataset in `adatas`.  Default to equal weight.
    load_keys : Mapping[str,str], default {'sequences':'sequences'}
        Mapping from variable names in the xarray.Dataset to the keys used in the output
        dictionary of NumPy arrays or tensors.
    dnatransform : Optional[DNATransform]
        Object that computes window bounds and reverse‐complement flags on sequence arrays.
    shuffle_dims : Sequence[str], default []
        Names of dataset dimensions to shuffle when `shuffle=True` (e.g. ["obs", "var"]).
    epoch_size : int, default 100000
        Reference number of batches per epoch (ignored by the loader but useful for logging).
    batch_dim : str, default "var"
        Name of the dimension along which batching and (for meta‐datasets) cross‐dataset mixing occur.
    sequence_vars : Sequence[str], default ["sequences"]
        Names of data variables in the dataset on which reverse‐complement windowing is applied.
    join : {'inner','outer'}, default 'inner'
        How to align coordinates across multiple datasets. Outer fills with np.nan
    num_workers : int, default 0
        Number of threads used by the ParallelMapper to preprocess in parallel.
    prefetch_factor : int, default 2
        Number of batches to prefetch asynchronously ahead of the GPU/CPU consumer.
    pin_memory : Optional[bool], default None
        Device to pin memory to.

    Attributes
    ----------
    modules : List[CrAnDataModule]
        Underlying single‐dataset modules, one per entry in `adatas`.
    global_coords : Dict[str, np.ndarray]
        Unified coordinate arrays computed from all datasets, used for reindexing.
    instructions : Dict[str, Dict[str, bool]]
        Per‐phase flags for whether to apply reverse‐complement ('apply_rc') and shuffling ('shuffle').

    Methods
    -------
    setup(state: str = "train")
        Initializes xbatcher BatchGenerators for each underlying dataset for the given phase.
    train_dataloader
        Returns a PyTorch Loader yielding batches for training.
    val_dataloader
        Returns a PyTorch Loader yielding batches for validation.
    test_dataloader
        Returns a PyTorch Loader yielding batches for testing.
    predict_dataloader
        Returns a PyTorch Loader yielding batches for prediction.
    load()
        Eagerly loads all datasets (if backed on disk) into memory.
    """
    def __init__(
        self,
        adatas,                                
        batch_size: int = 32,
        load_keys: dict[str,str] = {'sequences':'sequences'},
        dnatransform=None,
        shuffle_dims: list[str] = [],
        split: str = 'var-_-split',
        batch_dim: str = 'var',
        sequence_vars: list[str]    = ['sequences'],
        weights: list[float] | None = None,
        join: str = 'inner',    # 'inner' or 'outer'
        num_workers:  int = 0,
        prefetch_factor: int = 2,
        pin_memory:    str | None = None,
    ):
        # ─ normalize adatas to a list ──────────────────────────────────────────
        if not isinstance(adatas, (list,tuple)):
            adatas = [adatas]
        self.adatas       = adatas
        self.batch_size   = batch_size
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []
        self.shuffle      = len(shuffle_dims) > 0
        self.load_keys    = load_keys
        self.split        = split
        self.batch_dim    = batch_dim
        self.sequence_vars = sequence_vars
        self.num_workers  = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory   = pin_memory

        # ─ build state‐specific instructions once ────────────────────────────
        self.instructions = {
            "train":   {"apply_rc": True,  "shuffle": self.shuffle},
            "val":     {"apply_rc": True,  "shuffle": self.shuffle},
            "test":    {"apply_rc": False, "shuffle": False},
            "predict": {"apply_rc": False, "shuffle": False},
        }

        # ─ build per‐dataset weight map for MultiNodeWeightedSampler ──────────
        n = len(adatas)
        if weights is None:
            weights = [1.0/n]*n
        if len(weights)!=n:
            raise ValueError("weights must match number of adatas")
        self.weight_map = {i: float(w)/sum(weights) for i,w in enumerate(weights)}

        # ─ compute global_coords (union or intersection) ─────────────────────
        if join not in ('inner','outer'):
            raise ValueError("join must be 'inner' or 'outer'")
        all_dims = set().union(*(set(ds.coords) for ds in adatas))
        self.global_coords = {}
        for dim in all_dims:
            if dim==batch_dim: continue
            lists = [ds.coords[dim].values for ds in adatas if dim in ds.coords]
            if not lists: continue
            if join=='outer':
                vals = set().union(*(set(x) for x in lists))
            else:
                sets = [set(x) for x in lists]
                vals = sets[0].intersection(*sets[1:])
            self.global_coords[dim] = np.array(sorted(vals))

        # ─ placeholders for per‐dataset generators ────────────────────────────
        self._gens: list[xbatcher.BatchGenerator] = []
        self._converter = TensorConverter(self.load_keys, as_numpy=True)

    def setup(self, state: str="train"):
        """Build one BatchGenerator per dataset for the given state."""
        self._gens.clear()
        for ds in self.adatas:
            mask = ds[self.split].compute() == state
            sub = ds.isel({self.batch_dim: mask}) if state!="predict" else ds
            sub = sub[list(self.load_keys.keys())]
            dims = dict(sub.sizes)
            dims[self.batch_dim] = self.batch_size

            bg = xbatcher.BatchGenerator(
                ds=sub,
                input_dims=dims,
                batch_dims={self.batch_dim:self.batch_size},
                cache=None
            )
            self._gens.append(bg)

    def _get_dataloader(self, state: str) -> Loader:
        # ─ pick source: single‐ds or weighted mix ────────────────────────────
        if len(self._gens)==1:
            source = IterableWrapper(self._gens[0])
        else:
            node_map = {i: IterableWrapper(g) for i,g in enumerate(self._gens)}
            source = MultiNodeWeightedSampler(node_map, self.weight_map)

        # ─ reindex → transform → convert → prefetch → pin → loader ──────────
        # (we reindex *all* coords dims at once)
        reindex_fn = lambda ds: ds.reindex(self.global_coords, fill_value=np.nan)
        transform_applicator = DNATransformApplicator(
            state=state,
            instructions=self.instructions,
            dnatransform=self.dnatransform,
            shuffle_dims=self.shuffle_dims,
            sequence_vars=self.sequence_vars
        )
        node = ParallelMapper(source, 
                      map_fn=lambda x: self._converter(transform_applicator(reindex_fn(x))),
                             num_workers=self.num_workers)
        node = Prefetcher(node,prefetch_factor=self.prefetch_factor)
        if self.pin_memory is not None:
            node = PinMemory(node,self.pin_memory)
        return Loader(node)
        
    @property
    def train_dataloader(self) -> Loader:
        return self._get_dataloader("train")

    @property
    def val_dataloader(self) -> Loader:
        return self._get_dataloader("val")

    @property
    def test_dataloader(self) -> Loader:
        return self._get_dataloader("test")

    @property
    def predict_dataloader(self) -> Loader:
        return self._get_dataloader("predict")

    def load(self):
        """Bring all underlying CrAnData into memory if supported."""
        for ds in self.adatas:
            ds.load()

    def __repr__(self):
        return (f"CrAnDataModule(num_datasets={len(self.adatas)}, "
                f"batch_size={self.batch_size}, shuffle={self.shuffle})")
