import numpy as np
import xarray as xr

from grandata import GRAnData, concat_grandata


def _build_store(tmp_path, name: str, obs_names: list[str], var_names: list[str], offset: float):
    obs = np.array(obs_names, dtype=object)
    var = np.array(var_names, dtype=object)
    seq_bins = np.arange(2)
    X = np.zeros((len(obs), len(var), len(seq_bins)), dtype=np.float32)
    for i in range(len(obs)):
        for j in range(len(var)):
            X[i, j, :] = offset + i * 100 + j * 10 + np.arange(len(seq_bins))

    adata = GRAnData(
        X=xr.DataArray(
            X,
            dims=("obs", "var", "seq_bins"),
            coords={"obs": obs, "var": var, "seq_bins": seq_bins},
        ),
        **{
            "var-_-split": xr.DataArray(
                np.array(["train"] * len(var), dtype=object),
                dims=("var",),
                coords={"var": var},
            )
        },
    )
    path = tmp_path / f"{name}.zarr"
    adata.to_zarr(path, mode="w")
    return GRAnData.open_zarr(path, consolidated=False)


def test_concat_grandata_outer_join_adds_labels_and_returns_backed_store(tmp_path):
    human = _build_store(tmp_path, "human", ["obs0", "obs1"], ["h0", "h1"], offset=0.0)
    mouse = _build_store(tmp_path, "mouse", ["obs1", "obs2"], ["m0"], offset=1000.0)

    merged_path = tmp_path / "merged_outer.zarr"
    merged = concat_grandata(
        [human, mouse],
        out_path=merged_path,
        concat_dim="var",
        join="outer",
        add_key="var-_-species",
        add_values=["human", "mouse"],
    )

    assert isinstance(merged, GRAnData)
    assert merged.encoding["source"] == str(merged_path)
    assert merged.sizes["var"] == 3
    assert merged.sizes["obs"] == 3
    assert merged["var-_-species"].dims == ("var",)
    assert merged["var-_-species"].values.tolist() == ["human", "human", "mouse"]

    x = np.asarray(merged["X"])
    assert x.shape == (3, 3, 2)
    assert np.isnan(x[2, 0, 0])
    assert np.isclose(x[1, 1, 1], 11.0)
    assert np.isclose(x[1, 2, 0], 110.0)


def test_concat_grandata_inner_join_keeps_intersection_only(tmp_path):
    human = _build_store(tmp_path, "human_inner", ["shared", "human_only"], ["h0"], offset=0.0)
    mouse = _build_store(tmp_path, "mouse_inner", ["shared", "mouse_only"], ["m0"], offset=1000.0)

    merged = concat_grandata(
        [human, mouse],
        out_path=tmp_path / "merged_inner.zarr",
        concat_dim="var",
        join="inner",
    )

    assert merged.sizes["obs"] == 1
    assert merged.coords["obs"].values.tolist() == ["shared"]
    assert merged.sizes["var"] == 2


def test_concat_grandata_shuffle_reorders_concat_dimension(tmp_path):
    adata = _build_store(tmp_path, "shuffle_a", ["obs0"], ["a0", "a1"], offset=0.0)
    bdata = _build_store(tmp_path, "shuffle_b", ["obs0"], ["b0", "b1"], offset=100.0)

    merged = concat_grandata(
        [adata, bdata],
        out_path=tmp_path / "merged_shuffle.zarr",
        concat_dim="var",
        join="outer",
        add_key="var-_-species",
        add_values=["a", "b"],
        shuffle=True,
        random_state=0,
    )

    assert merged.coords["var"].values.tolist() != ["a0", "a1", "b0", "b1"]
    assert sorted(merged["var-_-species"].values.tolist()) == ["a", "a", "b", "b"]

