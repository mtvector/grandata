import numpy as np
from torchdata.nodes import Loader, IterableWrapper, Prefetcher, PinMemory, MultiNodeWeightedSampler
import threading
import queue
import traceback
import zarr

class GRAnDataModule:
    """
    A unified data‐loading module for one or more zarr‐backed GRAnData datasets that produces
    PyTorch‐compatible dataloaders for training, validation, testing, and prediction.

    This class supports:
      - Sampling from multiple datasets with optional weighting (for meta‐datasets).
      - Optional per-output transforms via a unified transform interface (including DNA windowing/RC).
      - Random shuffling along arbitrary dataset dimensions (e.g. obs).
      - Coordinate unification (inner or outer join) across multiple datasets.
      - Asynchronous prefetching and optional pinning of CPU memory.

    Parameters
    ----------
    adatas : Sequence[GRAnData]
        One or more GRAnData objects to draw samples from. If a single dataset is provided,
        it will behave like a standard DataModule; if multiple, samples are drawn according
        to `weights` and mixed at each batch.
    batch_size : int
        Number of slices along the `batch_dim` (e.g. "var") to include in each batch.
    weights : Optional[Sequence[float]]
        Sampling probability for each dataset in `adatas`.  Default to equal weight.
    load_keys : Mapping[str,str], default {'sequences':'sequences'}
        Mapping from variable names in the xarray.Dataset to the keys used in the output
        dictionary of NumPy arrays or tensors.
    shuffle_dims : Sequence[str], default []
        Names of dataset dimensions to shuffle when `shuffle=True` (e.g. ["obs", "var"]).
    epoch_size : int, default 100000
        Reference number of batches per epoch (ignored by the loader but useful for logging).
    batch_dim : str, default "var"
        Name of the dimension along which batching and (for meta‐datasets) cross‐dataset mixing occur.
    transforms : Optional[Dict[str, Sequence[Callable]]], default None
        Optional per-output transforms. Keys should match the output tensor names from
        ``load_keys`` and values are lists of callables applied in order. Use the special
        key ``"__batch__"`` for transforms that accept (batch, dims_map[, state]).
        Per-tensor transforms may accept (array[, dims[, state]]). If the top-level keys
        are loader states ("train", "val", "test", "predict"), the corresponding dict
        is selected per state.
    sample_weights : Optional[Sequence|Dict], default None
        Optional per-element sampling weights along ``batch_dim``. For a single dataset,
        pass a 1D array of length ``batch_dim``. For multi-dataset, pass a list matching
        ``adatas`` or a dict keyed by dataset index or zarr store path. When provided
        and non-uniform, batches are sampled with replacement and the iterator is
        effectively infinite. When omitted or uniform, batching uses contiguous order
        and exhausts each dataset.
    join : {'inner','outer'}, default 'inner'
        How to align coordinates across multiple datasets. Outer fills with np.nan
    prefetch_factor : int, default 2
        Number of batches to prefetch asynchronously ahead of the GPU/CPU consumer.
    pin_memory : Optional[str], default None
        Optional device string for torchdata PinMemory (e.g. "cuda").

    Attributes
    ----------
    adatas : List[GRAnData]
        Underlying datasets, normalized to a list.
    global_coords : Dict[str, np.ndarray]
        Unified coordinate arrays computed from all datasets, used for reindexing.
    instructions : Dict[str, Dict[str, bool]]
        Per‐phase flags for whether to shuffle.
    weight_map : Dict[int, float]
        Dataset sampling weights used by MultiNodeWeightedSampler.

    Methods
    -------
    setup(state: str = "train")
        Initializes fast zarr batch iterators for each dataset for the given phase.
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
        transforms: dict[str, list] | None = None,
        shuffle_dims: list[str] = [],
        split: str = 'var-_-split',
        batch_dim: str = 'var',
        sample_weights: object | None = None,
        weights: list[float] | None = None,
        join: str = 'inner',    # 'inner' or 'outer'
        prefetch_factor: int = 2,
        pin_memory:    str | None = None,
    ):
        # ─ normalize adatas to a list ──────────────────────────────────────────
        if not isinstance(adatas, (list,tuple)):
            adatas = [adatas]
        self.adatas       = adatas
        self.batch_size   = batch_size
        self.shuffle_dims = shuffle_dims or []
        self.shuffle      = len(shuffle_dims) > 0
        self.load_keys    = load_keys
        self.transforms = transforms or {}
        self.split        = split
        self.batch_dim    = batch_dim
        self.sample_weights = sample_weights
        self.prefetch_factor = prefetch_factor
        self.pin_memory   = pin_memory

        # ─ build state‐specific instructions once ────────────────────────────
        self.instructions = {
            "train":   {"shuffle": self.shuffle},
            "val":     {"shuffle": self.shuffle},
            "test":    {"shuffle": False},
            "predict": {"shuffle": False},
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

        self._fast_configs = None
        self._fast_fill_value = np.nan

    def _build_fast_config(self, ds, state: str, dataset_idx: int):
        if "source" not in getattr(ds, "encoding", {}):
            return None
        reindexers = {}
        if self.global_coords:
            for dim, target_vals in self.global_coords.items():
                if dim == self.batch_dim:
                    continue
                if dim not in ds.coords:
                    reindexers[dim] = np.full(len(target_vals), -1, dtype=int)
                    continue
                src_vals = ds.coords[dim].values
                if np.array_equal(src_vals, target_vals):
                    continue
                mapping = {val: i for i, val in enumerate(src_vals.tolist())}
                indexer = np.full(len(target_vals), -1, dtype=int)
                for i, val in enumerate(target_vals.tolist()):
                    idx = mapping.get(val)
                    if idx is not None:
                        indexer[i] = idx
                reindexers[dim] = indexer
        store_path = ds.encoding["source"]
        group = zarr.open_group(store_path, mode="r")

        if state == "predict":
            indices = np.arange(ds.sizes.get(self.batch_dim, 0))
        else:
            if self.split in group.array_keys():
                split_arr = np.asarray(group[self.split])
                indices = np.flatnonzero(split_arr == state)
            else:
                indices = np.arange(ds.sizes.get(self.batch_dim, 0))

        sample_weights = self._resolve_sample_weights(ds, dataset_idx, indices)
        arrays = {}
        axes = {}
        dims_map = {}
        for xr_key, out_key in self.load_keys.items():
            if xr_key not in group.array_keys():
                return None
            if xr_key not in ds:
                return None
            dims = ds[xr_key].dims
            if self.batch_dim not in dims:
                return None
            arrays[out_key] = group[xr_key]
            axes[out_key] = dims.index(self.batch_dim)
            dims_map[out_key] = dims

        return {
            "arrays": arrays,
            "axes": axes,
            "dims_map": dims_map,
            "indices": indices,
            "sample_weights": sample_weights,
            "batch_size": self.batch_size,
            "prefetch_factor": self.prefetch_factor,
            "reindexers": reindexers,
        }

    def _resolve_sample_weights(self, ds, dataset_idx: int, indices: np.ndarray):
        if self.sample_weights is None:
            return None
        weights = None
        if isinstance(self.sample_weights, dict):
            if dataset_idx in self.sample_weights:
                weights = self.sample_weights[dataset_idx]
            else:
                key = ds.encoding.get("source")
                if key in self.sample_weights:
                    weights = self.sample_weights[key]
        elif isinstance(self.sample_weights, (list, tuple)):
            if len(self.sample_weights) == len(self.adatas):
                weights = self.sample_weights[dataset_idx]
            else:
                weights = self.sample_weights
        else:
            weights = self.sample_weights

        if weights is None:
            return None
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("sample_weights must be a 1D array.")
        if len(weights) != ds.sizes.get(self.batch_dim, len(weights)):
            raise ValueError("sample_weights length must match the batch_dim size.")
        weights = weights[indices]
        total = weights.sum()
        if total <= 0:
            raise ValueError("sample_weights must have a positive sum for the selected state.")
        weights = weights / total
        if np.allclose(weights, weights[0]):
            return None
        return weights

    def _prefetch_iter(self, iterable, prefetch_factor: int):
        if prefetch_factor <= 0:
            yield from iterable
            return

        q = queue.Queue(maxsize=prefetch_factor)
        sentinel = object()

        def _producer():
            try:
                for item in iterable:
                    q.put(item)
            except Exception as exc:
                q.put(exc)
                q.put(traceback.format_exc())
            finally:
                q.put(sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                msg = q.get()
                raise RuntimeError(f"fast_zarr producer failed:\\n{msg}") from item
            yield item

    def _repeat_iter(self, factory):
        while True:
            for item in factory():
                yield item

    def _apply_shuffle(self, arr: np.ndarray, dims: tuple[str, ...], idx_map: dict[str, np.ndarray]):
        for dim, perm in idx_map.items():
            if dim not in dims:
                continue
            axis = dims.index(dim)
            arr = np.take(arr, perm, axis=axis)
        return arr

    def _apply_reindex(self, arr: np.ndarray, dims: tuple[str, ...], reindexers: dict[str, np.ndarray]):
        out = arr
        for dim, indexer in reindexers.items():
            if dim not in dims:
                continue
            axis = dims.index(dim)
            if np.all(indexer >= 0):
                out = np.take(out, indexer, axis=axis)
                continue
            out_shape = list(out.shape)
            out_shape[axis] = len(indexer)
            if np.issubdtype(out.dtype, np.floating):
                out_dtype = out.dtype
            else:
                out_dtype = np.float32
            out_arr = np.full(out_shape, self._fast_fill_value, dtype=out_dtype)
            valid_mask = indexer >= 0
            if np.any(valid_mask):
                valid_idx = indexer[valid_mask]
                taken = np.take(out, valid_idx, axis=axis)
                slc = [slice(None)] * out_arr.ndim
                slc[axis] = np.where(valid_mask)[0]
                out_arr[tuple(slc)] = taken
            out = out_arr
        return out

    def _resolve_transforms(self, state: str):
        if not self.transforms:
            return {}
        if all(key in ("train", "val", "test", "predict") for key in self.transforms.keys()):
            return self.transforms.get(state, {})
        return self.transforms

    def _apply_transforms(self, batch: dict, dims_map: dict[str, tuple[str, ...]], state: str):
        transforms = self._resolve_transforms(state)
        if not transforms:
            return batch
        batch_fns = transforms.get("__batch__", [])
        for fn in batch_fns:
            if getattr(fn, "_stateful", False):
                batch = fn(batch, dims_map, state)
            else:
                try:
                    batch = fn(batch, dims_map, state)
                except TypeError:
                    try:
                        batch = fn(batch, dims_map)
                    except TypeError:
                        batch = fn(batch)
        for out_key, fns in transforms.items():
            if out_key == "__batch__":
                continue
            if out_key not in batch:
                continue
            for fn in fns:
                if getattr(fn, "_stateful", False):
                    batch[out_key] = fn(batch[out_key], dims_map[out_key], state)
                else:
                    try:
                        batch[out_key] = fn(batch[out_key], dims_map[out_key], state)
                    except TypeError:
                        try:
                            batch[out_key] = fn(batch[out_key], dims_map[out_key])
                        except TypeError:
                            batch[out_key] = fn(batch[out_key])
        return batch

    def _fast_batch_iter(self, cfg, state: str):
        arrays = cfg["arrays"]
        axes = cfg["axes"]
        dims_map = cfg["dims_map"]
        indices = cfg["indices"]
        weights = cfg.get("sample_weights")
        reindexers = cfg["reindexers"]
        batch_size = cfg["batch_size"]
        total = len(indices)

        def _make_sel(arr, axis, idx):
            if isinstance(idx, slice):
                return tuple(idx if i == axis else slice(None) for i in range(arr.ndim))
            return tuple(idx if i == axis else slice(None) for i in range(arr.ndim))

        do_shuffle = self.instructions.get(state, {}).get("shuffle", False)

        def _iter_batches():
            if weights is not None:
                while True:
                    yield np.random.choice(indices, size=batch_size, replace=True, p=weights)
            for start in range(0, total, batch_size):
                batch_idx = indices[start:start + batch_size]
                if len(batch_idx) == 0:
                    break
                yield batch_idx

        for sel in _iter_batches():
            if isinstance(sel, np.ndarray) and sel.ndim == 1 and len(sel) > 1 and np.all(np.diff(sel) == 1):
                sel = slice(int(sel[0]), int(sel[-1] + 1))
            batch = {}
            for out_key, arr in arrays.items():
                axis = axes[out_key]
                selection = _make_sel(arr, axis, sel)
                if hasattr(arr, "oindex") and not isinstance(sel, slice):
                    batch[out_key] = np.asarray(arr.oindex[selection])
                else:
                    batch[out_key] = np.asarray(arr[selection])
                if reindexers:
                    batch[out_key] = self._apply_reindex(batch[out_key], dims_map[out_key], reindexers)

            if do_shuffle and self.shuffle_dims:
                shuffle_idx = {}
                for dim in self.shuffle_dims:
                    for out_key, dims in dims_map.items():
                        if dim in dims:
                            axis = dims.index(dim)
                            shuffle_idx[dim] = np.random.permutation(batch[out_key].shape[axis])
                            break
                if shuffle_idx:
                    for out_key, arr in batch.items():
                        batch[out_key] = self._apply_shuffle(arr, dims_map[out_key], shuffle_idx)
            batch = self._apply_transforms(batch, dims_map, state)
            yield batch

        yield from self._prefetch_iter(_iter_batches(), cfg["prefetch_factor"])

    def setup(self, state: str="train"):
        """Build fast zarr loader configs for each dataset and state."""
        configs = []
        for idx, ds in enumerate(self.adatas):
            cfg = self._build_fast_config(ds, state, idx)
            if cfg is None:
                raise ValueError("fast zarr loader requires zarr-backed datasets with compatible coords.")
            configs.append(cfg)
        self._fast_configs = configs

    def _get_dataloader(self, state: str) -> Loader:
        repeat = state in ("train", "val") and len(self._fast_configs) > 1
        if len(self._fast_configs) == 1:
            source = IterableWrapper(self._fast_batch_iter(self._fast_configs[0], state))
        else:
            node_map = {}
            for i, cfg in enumerate(self._fast_configs):
                if repeat:
                    node_map[i] = IterableWrapper(self._repeat_iter(lambda cfg=cfg: self._fast_batch_iter(cfg, state)))
                else:
                    node_map[i] = IterableWrapper(self._fast_batch_iter(cfg, state))
            source = MultiNodeWeightedSampler(node_map, self.weight_map)
        node = Prefetcher(source, prefetch_factor=self.prefetch_factor)
        if self.pin_memory is not None:
            node = PinMemory(node, self.pin_memory)
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
        """Bring all underlying GRAnData into memory if supported."""
        for ds in self.adatas:
            ds.load()

    def __repr__(self):
        return (f"GRAnDataModule(num_datasets={len(self.adatas)}, "
                f"batch_size={self.batch_size}, shuffle={self.shuffle})")


def _apply_dnatransform_array(arr: np.ndarray, dims: tuple[str, ...], rc_flags, start, end, dnatransform):
    var_name, seq_name, nuc_name = dnatransform.dimnames
    if var_name not in dims or seq_name not in dims or nuc_name not in dims:
        return arr
    var_axis = dims.index(var_name)
    seq_axis = dims.index(seq_name)
    nuc_axis = dims.index(nuc_name)

    arr = np.take(arr, np.arange(start, end), axis=seq_axis)
    if not dnatransform.random_rc:
        return arr

    arr = np.moveaxis(arr, var_axis, 0)
    rc_idx = np.flatnonzero(rc_flags)
    if len(rc_idx):
        seq_axis_adj = seq_axis - (1 if seq_axis > var_axis else 0)
        nuc_axis_adj = nuc_axis - (1 if nuc_axis > var_axis else 0)
        arr[rc_idx] = np.flip(arr[rc_idx], axis=(seq_axis_adj, nuc_axis_adj))
    arr = np.moveaxis(arr, 0, var_axis)
    return arr


def make_stateful_transform(fn, apply_states=("train", "val", "test", "predict")):
    """
    Wrap a transform so it only runs for selected loader states.

    The wrapped callable expects the loader state as its last positional argument.
    If the state is not in ``apply_states``, the first positional argument is
    returned unchanged.
    """
    def _wrapped(*args, **kwargs):
        if not args:
            return fn(*args, **kwargs)
        state = args[-1]
        if apply_states and state not in apply_states:
            return args[0]
        return fn(*args[:-1], **kwargs)

    _wrapped._stateful = True
    return _wrapped
