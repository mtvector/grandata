import numpy as np
import pytest
import xarray as xr

from grandata import GRAnData, GRAnDataModule
from grandata._module import make_paired_dna_target_transform, make_rc_signflip_transform
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


def test_grandata_module_can_keep_missing_batch_dim_shared(monkeypatch, tmp_path):
    adata = _build_module_dataset_with_rna_means(tmp_path)

    fixed_perm = np.array([2, 0, 1])
    monkeypatch.setattr(np.random, "permutation", lambda n: fixed_perm)

    module = GRAnDataModule(
        adatas=adata,
        batch_size=2,
        load_keys={"X": "atac_tracks", "rna_means": "rna_means"},
        in_memory_keys=["rna_means"],
        shared_keys=["rna_means"],
        emit_shuffle_indices=True,
        shuffle_dims=["obs"],
    )
    module.setup("train")

    assert isinstance(module._fast_configs[0]["arrays"]["rna_means"], np.ndarray)
    batch = next(iter(module.train_dataloader))
    assert batch["atac_tracks"].shape == (3, 2, 2)
    assert batch["rna_means"].shape == (3, 2)
    np.testing.assert_allclose(
        batch["rna_means"],
        np.array([[20, 21], [0, 1], [10, 11]], dtype=np.float32),
    )
    np.testing.assert_array_equal(batch["__shuffle_index__obs"], fixed_perm)


def test_paired_transform_keeps_targets_aligned_under_shift_and_rc(monkeypatch):
    dnatransform = DNATransform(
        out_len=4,
        random_rc=True,
        max_shift=2,
        dimnames=("var", "seq_len", "nuc"),
        apply_states=("train",),
        rc_states=("train",),
    )
    transform = make_paired_dna_target_transform(
        seq_key="sequence",
        dnatransform=dnatransform,
        target_configs={
            "atac_tracks": {
                "seq_dim": "seq_bins",
                "apply_window": True,
                "reverse_on_rc": True,
                "sign_flip_on_rc": False,
            },
            "rna_tracks": {
                "seq_dim": "seq_bins",
                "apply_window": True,
                "reverse_on_rc": True,
                "sign_flip_on_rc": True,
            },
        },
        batch_dim="var",
    )

    seq0 = one_hot_encode_sequence("ACGTACGT")
    seq1 = one_hot_encode_sequence("TTTTGGGG")
    sequence = np.stack([seq0, seq1], axis=0)
    atac_tracks = np.array([[[0, 1, 2, 3, 4, 5, 6, 7], [100, 101, 102, 103, 104, 105, 106, 107]]], dtype=np.float32)
    rna_tracks = atac_tracks.copy()
    batch = {"sequence": sequence, "atac_tracks": atac_tracks, "rna_tracks": rna_tracks}
    dims_map = {
        "sequence": ("var", "seq_len", "nuc"),
        "atac_tracks": ("obs", "var", "seq_bins"),
        "rna_tracks": ("obs", "var", "seq_bins"),
    }

    monkeypatch.setattr(np.random, "randint", lambda low, high=None, size=None, dtype=int: 2)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.array([0.2, 0.8], dtype=float))
    out = transform(batch, dims_map, "train")

    decoded = [hot_encoding_to_sequence(out["sequence"][i]) for i in range(2)]
    assert decoded == ["ACGT", "GGGG"]

    expected_atac = np.array([[[7, 6, 5, 4], [104, 105, 106, 107]]], dtype=np.float32)
    expected_rna = np.array([[[-7, -6, -5, -4], [104, 105, 106, 107]]], dtype=np.float32)
    np.testing.assert_allclose(out["atac_tracks"], expected_atac)
    np.testing.assert_allclose(out["rna_tracks"], expected_rna)
    assert out["sequence"].shape == (2, 4, 4)
    assert out["atac_tracks"].shape == (1, 2, 4)
    assert out["rna_tracks"].shape == (1, 2, 4)
    assert np.isfinite(out["sequence"]).all()
    assert np.isfinite(out["atac_tracks"]).all()
    assert np.isfinite(out["rna_tracks"]).all()


def test_make_rc_signflip_transform_reverses_bins_and_sign(monkeypatch):
    dnatransform = DNATransform(
        out_len=4,
        random_rc=True,
        max_shift=2,
        dimnames=("var", "seq_len", "nuc"),
        apply_states=("train",),
        rc_states=("train",),
    )
    transform = make_rc_signflip_transform(
        seq_key="sequence",
        flip_keys=["signed_tracks"],
        dnatransform=dnatransform,
        batch_dim="var",
        target_seq_dims={"signed_tracks": "seq_bins"},
        shift_keys=["signed_tracks"],
    )

    sequence = np.stack(
        [one_hot_encode_sequence("ACGTACGT"), one_hot_encode_sequence("CCCCAAAA")],
        axis=0,
    )
    signed_tracks = np.array([[[0, 1, 2, 3, 4, 5, 6, 7], [100, 101, 102, 103, 104, 105, 106, 107]]], dtype=np.float32)
    batch = {"sequence": sequence, "signed_tracks": signed_tracks}
    dims_map = {
        "sequence": ("var", "seq_len", "nuc"),
        "signed_tracks": ("obs", "var", "seq_bins"),
    }

    monkeypatch.setattr(np.random, "randint", lambda low, high=None, size=None, dtype=int: 2)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.array([0.2, 0.8], dtype=float))
    out = transform(batch, dims_map, "train")

    expected = np.array([[[-7, -6, -5, -4], [104, 105, 106, 107]]], dtype=np.float32)
    np.testing.assert_allclose(out["signed_tracks"], expected)


def _strand_mask_transform():
    dnatransform = DNATransform(
        out_len=4,
        random_rc=True,
        max_shift=None,
        dimnames=("var", "seq_len", "nuc"),
        apply_states=("train",),
        rc_states=("train",),
    )
    strand_mask = {
        "seq_dim": "seq_bins",
        "apply_window": True,
        "reverse_on_rc": True,
        "sign_flip_on_rc": False,
    }
    return make_paired_dna_target_transform(
        seq_key="sequence",
        dnatransform=dnatransform,
        target_configs={
            "rna_forward_locus_mask": {**strand_mask, "swap_on_rc": "rna_reverse_locus_mask"},
            "rna_reverse_locus_mask": strand_mask,
            "gene_body_mask": strand_mask,
        },
        batch_dim="var",
    )


def _strand_mask_batch():
    sequence = np.stack(
        [one_hot_encode_sequence("ACGTACGT"), one_hot_encode_sequence("TTTTGGGG")],
        axis=0,
    )
    # Sample 0 carries a plus-strand gene, sample 1 a minus-strand gene.
    forward = np.array([[0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    reverse = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.float32)
    batch = {
        "sequence": sequence,
        "rna_forward_locus_mask": forward,
        "rna_reverse_locus_mask": reverse,
        "gene_body_mask": forward + reverse,
    }
    dims_map = {
        "sequence": ("var", "seq_len", "nuc"),
        "rna_forward_locus_mask": ("var", "seq_bins"),
        "rna_reverse_locus_mask": ("var", "seq_bins"),
        "gene_body_mask": ("var", "seq_bins"),
    }
    return batch, dims_map


def test_swap_on_rc_moves_strand_masks_to_the_other_strand(monkeypatch):
    """Reverse complementing a plus-strand gene makes it a minus-strand gene."""
    batch, dims_map = _strand_mask_batch()
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.array([0.2, 0.8], dtype=float))
    out = _strand_mask_transform()(batch, dims_map, "train")

    # Sample 0 is reverse complemented: its footprint reverses within the window
    # and moves from the forward mask to the reverse mask. Sample 1 is untouched.
    np.testing.assert_allclose(
        out["rna_forward_locus_mask"],
        np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        out["rna_reverse_locus_mask"],
        np.array([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=np.float32),
    )
    # A strand-agnostic mask reverses in place and never changes keys.
    np.testing.assert_allclose(
        out["gene_body_mask"],
        np.array([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=np.float32),
    )


def test_swap_on_rc_is_inert_when_no_sample_is_reverse_complemented(monkeypatch):
    batch, dims_map = _strand_mask_batch()
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.array([0.8, 0.8], dtype=float))
    out = _strand_mask_transform()(batch, dims_map, "train")

    np.testing.assert_allclose(
        out["rna_forward_locus_mask"],
        np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        out["rna_reverse_locus_mask"],
        np.array([[0, 0, 0, 0], [0, 1, 1, 0]], dtype=np.float32),
    )
