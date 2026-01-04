import numpy as np
import pandas as pd
import pytest
import xarray as xr

import grandata
from grandata import chrom_io, seq_io


pytest.importorskip("pybigtools")
pytest.importorskip("pysam")


def _write_simple_bigwig(path, chrom, size, value):
    import pybigtools

    bw = pybigtools.open(str(path), mode="w")
    bw.write(chroms={chrom: size}, vals=[(chrom, 0, size, float(value))])
    bw.close()


def _make_test_grandata(tmp_path):
    bigwig_dir = tmp_path / "bigwigs"
    bigwig_dir.mkdir()
    _write_simple_bigwig(bigwig_dir / "obsA.bw", "chr1", 20, 1.0)
    _write_simple_bigwig(bigwig_dir / "obsB.bw", "chr1", 20, 2.0)

    region_table = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 10],
            "end": [10, 20],
        }
    )
    backed_path = tmp_path / "data.zarr"
    adata = chrom_io.grandata_from_bigwigs(
        region_table=region_table,
        bigwig_dir=bigwig_dir,
        backed_path=backed_path,
        target_region_width=10,
        array_name="X",
        n_bins=2,
        chunk_size=1,
        tile_size=2,
        obs_chunk_size=1,
        n_workers=1,
    )
    return adata, region_table


def test_grandata_from_bigwigs_and_dataframe(tmp_path):
    adata, region_table = _make_test_grandata(tmp_path)

    assert "X" in adata.data_vars
    assert adata["X"].shape == (2, 2, 2)
    np.testing.assert_allclose(adata["X"].values[0], 1.0)
    np.testing.assert_allclose(adata["X"].values[1], 2.0)

    var_df = adata.get_dataframe("var")
    assert set(var_df.columns) == {"chrom", "start", "end", "region"}
    assert list(var_df.index) == ["chr1:0-10", "chr1:10-20"]
    pd.testing.assert_frame_equal(
        var_df[["chrom", "start", "end"]].reset_index(drop=True),
        region_table.reset_index(drop=True),
    )


def test_add_genome_sequences_to_grandata_backed(tmp_path):
    adata, _ = _make_test_grandata(tmp_path)
    var_df = adata.get_dataframe("var")[["chrom", "start", "end"]]

    fasta_file = tmp_path / "chr1.fa"
    seq = ("ACGT" * 5)[:20]
    fasta_file.write_text(">chr1\n" + seq + "\n")

    import pysam

    pysam.faidx(str(fasta_file))
    chromsizes_file = tmp_path / "chrom.sizes"
    chromsizes_file.write_text("chr1\t20\n")

    genome = grandata.Genome(fasta_file, chromsizes_file)

    adata = seq_io.add_genome_sequences_to_grandata(
        adata,
        ranges_df=var_df,
        genome=genome,
        backed=True,
    )

    sequences = adata["sequences"]
    assert sequences.dims == ("var", "seq_len", "nuc")
    assert sequences.shape == (2, 10, 4)

    for i, row in var_df.reset_index(drop=True).iterrows():
        expected = genome.fetch(row["chrom"], row["start"], row["end"])
        decoded = seq_io.hot_encoding_to_sequence(sequences.isel(var=i).values)
        assert decoded == expected
