from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from scipy.sparse import csr_matrix

from grandata import GRAnData, tx_io


pybigtools = pytest.importorskip("pybigtools")


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


def test_add_gtf_annotation_masks_paints_gene_body_and_tss_support(tmp_path: Path) -> None:
    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text(
        "chr1\ttest\tgene\t20\t40\t.\t+\t.\tgene_name \"GeneA\";\n"
        "chr1\ttest\tgene\t70\t90\t.\t-\t.\tgene_name \"GeneB\";\n"
        "chr1\ttest\tgene\t45\t55\t.\t+\t.\tgene_name \"NotInRNA\";\n"
    )
    ds = GRAnData(
        data_vars={
            "var-_-chrom": xr.DataArray(np.array(["chr1", "chr1"]), dims=("var",)),
            "var-_-start": xr.DataArray(np.array([0, 50]), dims=("var",)),
            "var-_-end": xr.DataArray(np.array([50, 100]), dims=("var",)),
        },
        coords={"var": np.arange(2), "seq_bins": np.arange(10)},
    )
    store_path = tmp_path / "annotation_masks.zarr"
    ds.to_zarr(store_path)
    backed = GRAnData.open_zarr(store_path, consolidated=False)

    result = tx_io.add_gtf_annotation_masks(
        backed,
        gtf_file=gtf_path,
        gene_names=["GeneA", "GeneB"],
        tss_projection_bases=10,
        chunk_size=1,
    )

    expected_body = np.array([
        [False, False, False, False, True, True, True, True, False, False],
        [False, False, False, False, True, True, True, True, False, False],
    ])
    expected_locus = np.array([
        [False, False, False, False, True, True, False, False, False, False],
        [False, False, False, False, False, False, False, False, True, True],
    ])
    np.testing.assert_array_equal(result["gene_body_mask"].values, expected_body)
    np.testing.assert_array_equal(result["rna_locus_mask"].values, expected_locus)
    np.testing.assert_array_equal(
        result["rna_forward_locus_mask"].values,
        np.vstack([expected_locus[0], np.zeros(10, dtype=bool)]),
    )
    np.testing.assert_array_equal(
        result["rna_reverse_locus_mask"].values,
        np.vstack([np.zeros(10, dtype=bool), expected_locus[1]]),
    )
    assert result["rna_locus_mask"].attrs["annotation_kind"] == "tx_io_tss_projection"


def test_write_tss_bigwigs_preserves_matrix_gene_order(tmp_path: Path) -> None:
    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text(
        "chr1\ttest\tgene\t10\t20\t.\t+\t.\tgene_name \"GeneA\";\n"
        "chr1\ttest\tgene\t30\t40\t.\t+\t.\tgene_name \"GeneB\";\n"
    )
    output_dir = tmp_path / "bigwigs"
    tx_io.write_tss_bigwigs(
        np.array([[1.0, 2.0]]),
        var_names=["GeneB", "GeneA"],
        obs_names=["cell type"],
        gtf_file=str(gtf_path),
        target_dir=str(output_dir),
        gtf_gene_field="gene_name",
        n_bases=5,
        chromsizes={"chr1": 100},
    )

    reader = pybigtools.open(str(output_dir / "cell_type.bw"), mode="r")
    try:
        assert reader.values("chr1", 10, 11)[0] == pytest.approx(2.0)
        assert reader.values("chr1", 30, 31)[0] == pytest.approx(1.0)
    finally:
        reader.close()
