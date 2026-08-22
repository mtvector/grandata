import numpy as np
import xarray as xr

from grandata import GRAnData, GRAnDataModule


def _build_weighted_dataset(tmp_path, value_offset: float, n_var: int = 4):
    obs = np.array(["obs0"])
    var = np.array([f"v{i}" for i in range(n_var)])
    seq_bins = np.arange(1)
    X = np.zeros((len(obs), len(var), len(seq_bins)), dtype=np.float32)
    for i in range(n_var):
        X[0, i, 0] = value_offset + i
    data_vars = {
        "X": xr.DataArray(X, dims=("obs", "var", "seq_bins"), coords={"obs": obs, "var": var, "seq_bins": seq_bins}),
        "var-_-split": xr.DataArray(np.array(["train"] * n_var, dtype=object), dims=("var",), coords={"var": var}),
    }
    adata = GRAnData(**data_vars)
    out_path = tmp_path / f"weighted_{value_offset}.zarr"
    adata.to_zarr(out_path, mode="w")
    return GRAnData.open_zarr(out_path, consolidated=False)


def test_weighted_sampling_single_dataset(tmp_path):
    np.random.seed(0)
    adata = _build_weighted_dataset(tmp_path, value_offset=0.0, n_var=4)
    weights = np.array([0.7, 0.1, 0.1, 0.1], dtype=float)

    module = GRAnDataModule(
        adatas=adata,
        batch_size=1,
        load_keys={"X": "X"},
        sample_weights=weights,
    )
    module.setup("train")
    loader = module.train_dataloader

    counts = np.zeros(4, dtype=int)
    n_batches = 1000
    def _get_x(batch):
        if isinstance(batch, dict):
            return batch["X"]
        return batch

    for i, batch in enumerate(loader):
        idx = int(np.asarray(_get_x(batch)).reshape(-1)[0])
        counts[idx] += 1
        if i + 1 >= n_batches:
            break

    assert counts[0] > n_batches * 0.5


def test_weighted_sampling_meta_module(tmp_path):
    np.random.seed(0)
    adata_a = _build_weighted_dataset(tmp_path, value_offset=10.0, n_var=4)
    adata_b = _build_weighted_dataset(tmp_path, value_offset=20.0, n_var=4)

    module = GRAnDataModule(
        adatas=[adata_a, adata_b],
        batch_size=1,
        load_keys={"X": "X"},
        weights=[0.8, 0.2],
    )
    module.setup("train")
    loader = module.train_dataloader

    n_batches = 1000
    count_a = 0
    count_b = 0
    def _get_x(batch):
        if isinstance(batch, dict):
            return batch["X"]
        return batch

    for i, batch in enumerate(loader):
        value = float(np.asarray(_get_x(batch)).reshape(-1)[0])
        if value >= 20.0:
            count_b += 1
        else:
            count_a += 1
        if i + 1 >= n_batches:
            break

    frac_a = count_a / n_batches
    assert frac_a >= 0.7


def test_chunk_local_weighted_sampling_preserves_marginals(tmp_path):
    np.random.seed(0)
    adata = _build_weighted_dataset(tmp_path, value_offset=0.0, n_var=8)
    weights = np.array([0.7, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025])
    module = GRAnDataModule(
        adatas=adata,
        batch_size=4,
        load_keys={"X": "X"},
        sample_weights=weights,
        sample_weight_chunk_size=4,
    )
    module.setup("train")

    counts = np.zeros(8, dtype=int)
    for batch_index, batch in enumerate(module.train_dataloader):
        values = np.asarray(batch["X"]).reshape(-1).astype(int)
        assert np.unique(values // 4).size == 1
        counts += np.bincount(values, minlength=8)
        if batch_index + 1 >= 2000:
            break

    observed = counts / counts.sum()
    expected = weights / weights.sum()
    assert np.allclose(observed, expected, atol=0.03)


def test_seeded_weighted_sampling_is_prefetch_deterministic(tmp_path):
    adata = _build_weighted_dataset(tmp_path, value_offset=0.0, n_var=8)
    weights = np.array([0.40, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05])

    streams = []
    for seed in (17, 17, 18):
        module = GRAnDataModule(
            adatas=adata,
            batch_size=4,
            load_keys={"X": "X"},
            sample_weights=weights,
            sample_weight_chunk_size=4,
            prefetch_factor=3,
            io_workers=2,
            random_state=seed,
        )
        module.setup("train")
        batches = []
        for batch_index, batch in enumerate(module.train_dataloader):
            batches.append(np.asarray(batch["X"]).reshape(-1).astype(int))
            if batch_index + 1 >= 20:
                break
        streams.append(np.stack(batches))
        np.random.seed(seed * 101)
        np.random.random(1000)

    assert np.array_equal(streams[0], streams[1])
    assert not np.array_equal(streams[0], streams[2])
