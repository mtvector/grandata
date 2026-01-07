import numpy as np
import pytest
import xarray as xr

from grandata import GRAnData, GRAnDataModule
from grandata.seq_io import DNATransform, hot_encoding_to_sequence, one_hot_encode_sequence


def _build_module_dataset(tmp_path):
    obs_names = np.array(["obs0", "obs1", "obs2"])
    var_names = np.array(["chr1:0-6", "chr1:6-12", "chr1:12-18", "chr1:18-24"])
    seq_bins = np.arange(2)
    seq_len = np.arange(6)
    nuc = np.array(["A", "C", "G", "T"])

    X = np.zeros((obs_names.size, var_names.size, seq_bins.size), dtype=np.float32)
    for obs_idx in range(obs_names.size):
        for var_idx in range(var_names.size):
            for bin_idx in range(seq_bins.size):
                X[obs_idx, var_idx, bin_idx] = obs_idx * 100 + var_idx * 10 + bin_idx
    rna_tracks = X + 1000

    seqs = [
        "ACGTAC",
        "CCCCCC",
        "GGGGGG",
        "TTTTTT",
    ]
    sequences = np.stack([one_hot_encode_sequence(s) for s in seqs], axis=0)

    data_vars = {
        "X": xr.DataArray(
            X,
            dims=("obs", "var", "seq_bins"),
            coords={"obs": obs_names, "var": var_names, "seq_bins": seq_bins},
        ),
        "rna_tracks": xr.DataArray(
            rna_tracks,
            dims=("obs", "var", "seq_bins"),
            coords={"obs": obs_names, "var": var_names, "seq_bins": seq_bins},
        ),
        "sequences": xr.DataArray(
            sequences,
            dims=("var", "seq_len", "nuc"),
            coords={"var": var_names, "seq_len": seq_len, "nuc": nuc},
        ),
        "var-_-split": xr.DataArray(
            np.array(["train"] * var_names.size, dtype=object),
            dims=("var",),
            coords={"var": var_names},
        ),
    }

    adata = GRAnData(**data_vars)
    out_path = tmp_path / "module_align.zarr"
    adata.to_zarr(out_path, mode="w")
    return GRAnData.open_zarr(out_path, consolidated=False)


def _build_module_dataset_with_rna_means(tmp_path):
    obs_names = np.array(["obs0", "obs1", "obs2"])
    var_names = np.array(["chr1:0-6", "chr1:6-12", "chr1:12-18", "chr1:18-24"])
    seq_bins = np.arange(2)
    gene_names = np.array(["g0", "g1"])

    X = np.zeros((obs_names.size, var_names.size, seq_bins.size), dtype=np.float32)
    for obs_idx in range(obs_names.size):
        for var_idx in range(var_names.size):
            for bin_idx in range(seq_bins.size):
                X[obs_idx, var_idx, bin_idx] = obs_idx * 100 + var_idx * 10 + bin_idx

    rna_means = np.zeros((obs_names.size, gene_names.size), dtype=np.float32)
    for obs_idx in range(obs_names.size):
        for gene_idx in range(gene_names.size):
            rna_means[obs_idx, gene_idx] = obs_idx * 10 + gene_idx

    data_vars = {
        "X": xr.DataArray(
            X,
            dims=("obs", "var", "seq_bins"),
            coords={"obs": obs_names, "var": var_names, "seq_bins": seq_bins},
        ),
        "rna_means": xr.DataArray(
            rna_means,
            dims=("obs", "gene"),
            coords={"obs": obs_names, "gene": gene_names},
        ),
        "var-_-split": xr.DataArray(
            np.array(["train"] * var_names.size, dtype=object),
            dims=("var",),
            coords={"var": var_names},
        ),
    }

    adata = GRAnData(**data_vars)
    out_path = tmp_path / "module_broadcast.zarr"
    adata.to_zarr(out_path, mode="w")
    return GRAnData.open_zarr(out_path, consolidated=False)


def test_grandata_module_alignment_and_shuffle(monkeypatch, tmp_path):
    adata = _build_module_dataset(tmp_path)

    fixed_perm = np.array([2, 0, 1])
    monkeypatch.setattr(np.random, "permutation", lambda n: fixed_perm)

    transform = DNATransform(out_len=4, random_rc=False, max_shift=None, apply_states=("train", "val"))
    transforms = {
        "sequence": [
            transform
        ]
    }
    module = GRAnDataModule(
        adatas=adata,
        batch_size=2,
        load_keys={"X": "atac_tracks", "rna_tracks": "rna_tracks", "sequences": "sequence"},
        transforms=transforms,
        shuffle_dims=["obs"],
    )
    module.setup("train")

    batch = next(iter(module.train_dataloader))
    atac_tracks = batch["atac_tracks"]
    rna_tracks = batch["rna_tracks"]
    sequences = batch["sequence"]

    assert atac_tracks.shape == (3, 2, 2)
    assert rna_tracks.shape == (3, 2, 2)
    assert sequences.shape == (2, 4, 4)

    expected = np.zeros_like(atac_tracks)
    for obs_out, obs_idx in enumerate(fixed_perm):
        expected[obs_out] = np.array(
            [
                [obs_idx * 100 + 0 * 10 + 0, obs_idx * 100 + 0 * 10 + 1],
                [obs_idx * 100 + 1 * 10 + 0, obs_idx * 100 + 1 * 10 + 1],
            ],
            dtype=np.float32,
        )

    np.testing.assert_allclose(atac_tracks, expected)
    np.testing.assert_allclose(rna_tracks, expected + 1000)

    decoded = [hot_encoding_to_sequence(sequences[i]) for i in range(sequences.shape[0])]
    assert decoded == ["CGTA", "CCCC"]


def test_grandata_module_broadcasts_missing_batch_dim(monkeypatch, tmp_path):
    adata = _build_module_dataset_with_rna_means(tmp_path)

    fixed_perm = np.array([2, 0, 1])
    monkeypatch.setattr(np.random, "permutation", lambda n: fixed_perm)

    module = GRAnDataModule(
        adatas=adata,
        batch_size=2,
        load_keys={"X": "atac_tracks", "rna_means": "rna_means"},
        shuffle_dims=["obs"],
    )
    module.setup("train")

    batch = next(iter(module.train_dataloader))
    rna_means = batch["rna_means"]

    assert rna_means.shape == (2, 3, 2)

    expected_base = np.array(
        [
            [20, 21],
            [0, 1],
            [10, 11],
        ],
        dtype=np.float32,
    )
    expected = np.broadcast_to(expected_base, (2,) + expected_base.shape)
    np.testing.assert_allclose(rna_means, expected)
