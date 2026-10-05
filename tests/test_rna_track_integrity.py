"""RNA tracks must carry each gene's own value, built from the same loci as the masks."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from grandata import GRAnData, chrom_io, tx_io

pybigtools = pytest.importorskip("pybigtools")

# Two genes with gene rows on chr1; three on chr2 with transcripts only (the
# NCBI Mmul10 pattern); "Dup" has gene rows on both chromosomes.
GTF = (
    'chr1\tt\tgene\t1001\t2000\t.\t+\t.\tgene_name "GeneA";\n'
    'chr1\tt\ttranscript\t1001\t2000\t.\t+\t.\tgene_name "GeneA";\n'
    'chr1\tt\tgene\t5001\t6000\t.\t-\t.\tgene_name "Dup";\n'
    'chr2\tt\tgene\t9001\t9500\t.\t+\t.\tgene_name "Dup";\n'
    'chr2\tt\ttranscript\t3001\t3800\t.\t+\t.\tgene_name "GeneC";\n'
    'chr2\tt\ttranscript\t2901\t3500\t.\t+\t.\tgene_name "GeneC";\n'
    'chr2\tt\ttranscript\t7001\t7600\t.\t-\t.\tgene_name "GeneD";\n'
    'chr2\tt\ttranscript\t12001\t12400\t.\t-\t.\tgene_name "GeneD";\n'
    'chr2\tt\ttranscript\t11001\t11500\t.\t-\t.\tgene_name "GeneD";\n'
)
GENES = ["GeneD", "Dup", "GeneA", "GeneC", "NotInGtf"]
EXPRESSION = np.array([[4.0, 2.0, 1.0, 3.0, 9.0], [40.0, 20.0, 10.0, 30.0, 90.0]])
CELLS = ["cell/one", "cell two"]
CHROMSIZES = {"chr1": 20_000, "chr2": 20_000}


@pytest.fixture
def gtf(tmp_path: Path) -> Path:
    path = tmp_path / "genes.gtf"
    path.write_text(GTF)
    return path


def test_loci_derive_missing_gene_rows_and_keep_every_locus(gtf):
    with pytest.warns(UserWarning, match="gene rows cover 2 of 2|lack gene rows"):
        annotation = tx_io.load_gene_loci(gtf, gene_names=GENES)
    loci = {(locus.name, locus.chrom, locus.tss, locus.source) for locus in annotation.loci}
    assert ("GeneC", "chr2", 2901, "transcripts") in loci  # union of its transcripts
    # GeneD's transcripts sit at two loci on the same strand; the span covers both.
    assert ("GeneD", "chr2", 12400, "transcripts") in loci
    assert {("Dup", "chr1", 6000, "gene"), ("Dup", "chr2", 9001, "gene")} <= loci
    summary = annotation.summary()
    assert summary["derived_from_transcripts_gene_count"] == 2
    assert summary["multi_locus_gene_count"] == 1
    assert summary["unmatched_gene_count"] == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert {locus.name for locus in tx_io.load_gene_loci(gtf, gene_names=GENES, missing_gene_rows="skip").loci} == {"GeneA", "Dup"}
    with pytest.raises(ValueError, match="no gene row"):
        tx_io.load_gene_loci(gtf, gene_names=GENES, missing_gene_rows="error")


def test_gene_rows_on_part_of_the_genome_warn(tmp_path):
    path = tmp_path / "partial.gtf"
    path.write_text(
        'chr1\tt\tgene\t1001\t2000\t.\t+\t.\tgene_name "GeneA";\n'
        'chr2\tt\ttranscript\t3001\t3800\t.\t+\t.\tgene_name "GeneC";\n'
    )
    with pytest.warns(UserWarning, match="gene rows cover 1 of 1 sequences|cover 1 of 2"):
        tx_io.load_gene_loci(path, gene_names=["GeneA", "GeneC"])


def _store(tmp_path: Path, regions: pd.DataFrame, obs: list[str], with_var_index: bool = True) -> GRAnData:
    names = [f"{c}:{s}-{e}" for c, s, e in zip(regions.chrom, regions.start, regions.end)]
    data = {
        "var-_-chrom": (("var",), regions.chrom.to_numpy().astype(object)),
        "var-_-start": (("var",), regions.start.to_numpy()),
        "var-_-end": (("var",), regions.end.to_numpy()),
        "obs-_-index": (("obs",), np.asarray(obs, dtype=object)),
    }
    if with_var_index:
        data["var-_-index"] = (("var",), np.asarray(names, dtype=object))
    dataset = xr.Dataset(data, coords={"var": names, "obs": obs, "seq_bins": np.arange(40)})
    path = tmp_path / "store.zarr"
    GRAnData(data_vars=dataset.data_vars, coords=dataset.coords).to_zarr(path, mode="w")
    return GRAnData.open_zarr(str(path), consolidated=False)


REGIONS = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr2", "chr2"],
                        "start": [0, 4000, 2000, 6000, 10000], "end": [4000, 8000, 6000, 10000, 14000]})


def test_tracks_and_masks_cover_the_same_loci_and_pass_the_audit(gtf, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = tx_io.write_tss_bigwigs(
            EXPRESSION, var_names=GENES, obs_names=CELLS, gtf_file=str(gtf), target_dir=str(tmp_path / "bw"),
            gtf_gene_field="gene_name", n_bases=500, chromsizes=CHROMSIZES,
        )
    assert summary["locus_count"] == 5
    adata = _store(tmp_path, REGIONS, ["cell-one", "cell_two"])
    adata = chrom_io.add_bigwig_array(
        adata, region_table=REGIONS, bigwig_dir=str(tmp_path / "bw"), array_name="rna_tracks", obs_dim="obs",
        var_dim="var", seq_dim="seq_bins", target_region_width=4000, n_bins=40, fill_value=0.0, chunk_size=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        masks = tx_io.add_gtf_annotation_masks(adata, gtf_file=gtf, gene_names=GENES, tss_projection_bases=500)
    tracks = np.asarray(adata["rna_tracks"].values)
    locus_mask = np.asarray(masks["rna_locus_mask"].values).astype(bool)
    assert locus_mask.any(axis=1).tolist() == [True, True, True, True, True]
    assert ((np.abs(tracks).max(axis=0) > 0) == locus_mask).all(), "tracks and masks disagree on loci"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audit = tx_io.audit_tss_tracks(
            adata, expression=EXPRESSION, gene_names=GENES, obs_names=CELLS, gtf_file=gtf, n_bases=500, probes=50,
        )
    assert audit["checked"] >= 4 and audit["matching_fraction"] == 1.0


def test_audit_catches_values_on_the_wrong_genes(gtf, tmp_path):
    shuffled = EXPRESSION[:, [1, 2, 3, 0, 4]]  # every gene's value moved to another gene
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tx_io.write_tss_bigwigs(
            shuffled, var_names=GENES, obs_names=CELLS, gtf_file=str(gtf), target_dir=str(tmp_path / "bw"),
            gtf_gene_field="gene_name", n_bases=500, chromsizes=CHROMSIZES,
        )
    adata = chrom_io.add_bigwig_array(
        _store(tmp_path, REGIONS, ["cell-one", "cell_two"]), region_table=REGIONS, bigwig_dir=str(tmp_path / "bw"),
        array_name="rna_tracks", obs_dim="obs", var_dim="var", seq_dim="seq_bins", target_region_width=4000,
        n_bins=40, fill_value=0.0, chunk_size=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audit = tx_io.audit_tss_tracks(
            adata, expression=EXPRESSION, gene_names=GENES, obs_names=CELLS, gtf_file=gtf, n_bases=500, probes=50,
        )
    assert audit["matching_fraction"] < 0.5 and audit["mismatches"]


def test_bigwig_array_aligns_without_var_index_and_refuses_missing_files(gtf, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tx_io.write_tss_bigwigs(
            EXPRESSION, var_names=GENES, obs_names=CELLS, gtf_file=str(gtf), target_dir=str(tmp_path / "bw"),
            gtf_gene_field="gene_name", n_bases=500, chromsizes=CHROMSIZES,
        )
    no_index = _store(tmp_path, REGIONS, ["cell-one", "cell_two"], with_var_index=False)
    filled = chrom_io.add_bigwig_array(
        no_index, region_table=REGIONS, bigwig_dir=str(tmp_path / "bw"), array_name="rna_tracks", obs_dim="obs",
        var_dim="var", seq_dim="seq_bins", target_region_width=4000, n_bins=40, fill_value=0.0, chunk_size=2,
    )
    assert np.abs(np.asarray(filled["rna_tracks"].values)).max() > 0, "aligned output must not be all fill"
    renamed = _store(tmp_path / "renamed", REGIONS, ["cell-one", "cell three"])
    with pytest.raises(ValueError, match="no BigWig"):
        chrom_io.add_bigwig_array(
            renamed, region_table=REGIONS, bigwig_dir=str(tmp_path / "bw"), array_name="rna_tracks", obs_dim="obs",
            var_dim="var", seq_dim="seq_bins", target_region_width=4000, n_bins=40, fill_value=0.0, chunk_size=2,
        )


def test_writer_rejects_a_file_whose_values_do_not_match_their_genes(gtf, tmp_path, monkeypatch):
    real_open = tx_io.pybigtools.open

    class CorruptingWriter:
        def __init__(self, writer):
            self.writer = writer

        def write(self, chroms, vals):
            self.writer.write(chroms=chroms, vals=[(c, s, e, v * 2.0) for c, s, e, v in vals])

        def close(self):
            self.writer.close()

    def open_(path, mode="r"):
        handle = real_open(path, mode=mode)
        return CorruptingWriter(handle) if mode == "w" else handle

    monkeypatch.setattr(tx_io.pybigtools, "open", open_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="wrong value"):
            tx_io.write_tss_bigwigs(
                EXPRESSION, var_names=GENES, obs_names=CELLS, gtf_file=str(gtf), target_dir=str(tmp_path / "bw"),
                gtf_gene_field="gene_name", n_bases=500, chromsizes=CHROMSIZES,
            )


def test_merged_tracks_are_exactly_zero_between_loci(tmp_path):
    """Overlapping float values must not leave rounding residue spanning the gaps."""
    rows, names = [], []
    rng = np.random.default_rng(0)
    for i in range(60):
        start = 1000 + (i // 3) * 2000 + (i % 3) * 300  # triplets of overlapping projections
        rows.append(f'chr1\tt\tgene\t{start}\t{start + 500}\t.\t+\t.\tgene_name "G{i}";\n')
        names.append(f"G{i}")
    gtf = tmp_path / "many.gtf"
    gtf.write_text("".join(rows))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tx_io.write_tss_bigwigs(
            rng.random((1, 60)) * 0.1 + 0.07, var_names=names, obs_names=["c"], gtf_file=str(gtf),
            target_dir=str(tmp_path / "bw"), gtf_gene_field="gene_name", n_bases=1000, chromsizes={"chr1": 50_000},
        )
    reader = pybigtools.open(str(tmp_path / "bw" / "c.bw"), mode="r")
    try:
        values = np.asarray(reader.values("chr1", 0, 50_000, missing=0.0))
    finally:
        reader.close()
    covered = np.zeros(50_000, dtype=bool)
    for i in range(60):
        start = 1000 + (i // 3) * 2000 + (i % 3) * 300
        covered[start : start + 1000] = True
    assert (values[~covered] == 0).all()
    assert (values[covered] > 0.06).all()
