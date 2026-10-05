import numpy as np
import pytest
import xarray as xr
import zarr

from grandata import GRAnData, GRAnDataModule


def _build_store(tmp_path, n_obs=6, n_var=8, n_bins=3):
    obs = np.array([f"o{i}" for i in range(n_obs)])
    var = np.array([f"v{i}" for i in range(n_var)])
    X = (np.arange(n_obs)[:, None, None] * 100 + np.arange(n_var)[None, :, None] + np.zeros((1, 1, n_bins))).astype(
        np.float32
    )
    means = (np.arange(n_obs)[:, None] * 10 + np.arange(2)[None, :]).astype(np.float32)
    data_vars = {
        "X": xr.DataArray(X, dims=("obs", "var", "seq_bins"), coords={"obs": obs, "var": var}),
        "means": xr.DataArray(means, dims=("obs", "gene"), coords={"obs": obs}),
        "var-_-split": xr.DataArray(np.array(["train"] * n_var, dtype=object), dims=("var",), coords={"var": var}),
    }
    path = tmp_path / "selected.zarr"
    GRAnData(**data_vars).to_zarr(path, mode="w", encoding={"X": {"chunks": (1, 4, n_bins)}})
    return GRAnData.open_zarr(path, consolidated=False), path


def _batches(module, n):
    module.setup("train")
    out = []
    for batch in module.train_dataloader:
        out.append(batch)
        if len(out) == n:
            break
    return out


def test_selected_rows_match_a_full_read_and_shared_arrays_follow(tmp_path):
    adata, _ = _build_store(tmp_path)
    calls = []

    def pick(var_indices, state):
        calls.append((np.asarray(var_indices).copy(), state))
        return np.array([4, 1]) if var_indices[0] < 4 else np.array([0, 5, 2])

    module = GRAnDataModule(
        adatas=adata,
        batch_size=4,
        load_keys={"X": "X", "means": "means"},
        shared_keys=["means"],
        broadcast_missing_batch_dim=False,
        dim_selectors={"obs": pick},
    )
    first, second = _batches(module, 2)
    np.testing.assert_array_equal(first["__selected__obs"], [4, 1])
    np.testing.assert_array_equal(first["X"][:, :, 0], np.array([[400, 401, 402, 403], [100, 101, 102, 103]]))
    np.testing.assert_array_equal(first["means"][:, 0], [40, 10])
    np.testing.assert_array_equal(second["__selected__obs"], [0, 5, 2])
    assert second["X"].shape == (3, 4, 3)
    np.testing.assert_array_equal(second["X"][:, 0, 0], [4, 504, 204])
    assert calls[0][1] == "train"


def test_only_selected_chunks_are_read(tmp_path, monkeypatch):
    adata, _ = _build_store(tmp_path)
    module = GRAnDataModule(
        adatas=adata,
        batch_size=4,
        load_keys={"X": "X"},
        dim_selectors={"obs": lambda var_indices, state: np.array([3])},
    )
    requested = []
    original = zarr.Array.get_orthogonal_selection

    def spy(self, selection, *args, **kwargs):
        requested.append(selection)
        return original(self, selection, *args, **kwargs)

    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", spy)
    (batch,) = _batches(module, 1)
    assert batch["X"].shape == (1, 4, 3)
    assert requested and all(np.asarray(sel[0]).tolist() == [3] for sel in requested)


def test_selection_follows_obs_shuffle(tmp_path):
    adata, _ = _build_store(tmp_path)
    module = GRAnDataModule(
        adatas=adata,
        batch_size=4,
        load_keys={"X": "X"},
        shuffle_dims=["obs"],
        random_state=3,
        dim_selectors={"obs": lambda var_indices, state: np.array([0, 2, 4, 5])},
    )
    for batch in _batches(module, 3):
        np.testing.assert_array_equal(batch["X"][:, 0, 0] // 100, batch["__selected__obs"])


def test_selected_dimension_must_not_be_reindexed(tmp_path):
    adata, _ = _build_store(tmp_path)
    other = adata.isel(obs=slice(0, 3))
    module = GRAnDataModule(
        adatas=[adata, other],
        batch_size=4,
        load_keys={"X": "X"},
        join="outer",
        dim_selectors={"obs": lambda var_indices, state: np.array([0])},
    )
    with pytest.raises(ValueError, match="dim_selectors"):
        module.setup("train")


def test_batch_indices_trace_rows_back_to_the_store(tmp_path):
    adata, _ = _build_store(tmp_path)
    weights = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    module = GRAnDataModule(
        adatas=adata,
        batch_size=4,
        load_keys={"X": "X"},
        sample_weights=weights,
        random_state=1,
        emit_batch_indices=True,
    )
    for batch in _batches(module, 5):
        positions = batch["__index__var"]
        assert set(positions.tolist()) <= {1, 3, 4, 7}
        np.testing.assert_array_equal(batch["X"][0, :, 0], positions)


def test_batch_indices_are_off_by_default(tmp_path):
    adata, _ = _build_store(tmp_path)
    (batch,) = _batches(GRAnDataModule(adatas=adata, batch_size=4, load_keys={"X": "X"}), 1)
    assert "__index__var" not in batch
