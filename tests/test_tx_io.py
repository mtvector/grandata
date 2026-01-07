import numpy as np
import h5py
from scipy.sparse import csr_matrix

from grandata import tx_io


def _write_minimal_h5ad(path):
    obs_names = np.array(["o0", "o1", "o2", "o3"], dtype="S")
    var_names = np.array(["g0", "g1", "g2"], dtype="S")
    groups = np.array(["A", "A", "B", "B"], dtype="S")

    x_dense = np.array(
        [
            [1, 0, 2],
            [0, 3, 0],
            [1, 1, 1],
            [0, 0, 4],
        ],
        dtype=np.float64,
    )
    x_csr = csr_matrix(x_dense)

    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.create_dataset("_index", data=obs_names)
        obs.create_dataset("Group", data=groups)

        var = f.create_group("var")
        var.create_dataset("_index", data=var_names)
        var.create_dataset("dummy", data=np.arange(len(var_names), dtype=np.int64))

        xgrp = f.create_group("X")
        xgrp.create_dataset("data", data=x_csr.data)
        xgrp.create_dataset("indices", data=x_csr.indices)
        xgrp.create_dataset("indptr", data=x_csr.indptr)
        xgrp.create_dataset("shape", data=np.asarray(x_csr.shape, dtype=np.int64))

    return x_dense


def test_read_h5ad_selective_and_group_aggr_mean(tmp_path):
    h5ad_path = tmp_path / "mini.h5ad"
    x_dense = _write_minimal_h5ad(h5ad_path)

    ds = tx_io.read_h5ad_selective_to_grandata(
        h5ad_path, selected_fields=["X", "obs", "var"]
    )
    try:
        result = tx_io.group_aggr_xr(
            ds, "X", "obs-_-Group", agg_func=np.mean, materialize=False
        )
        got = np.asarray(result.data.todense())

        expected = np.vstack(
            [
                x_dense[:2].mean(axis=0),
                x_dense[2:].mean(axis=0),
            ]
        )
        assert np.allclose(got, expected)
    finally:
        tx_io.close_h5_backing(ds)
