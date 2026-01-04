# Tests

This folder focuses on core data integrity and alignment behaviors used by the
`code/xscgem/data/create_spc_dataset.ipynb` workflow.

What is covered
- `tests/test_chrom_seq_io.py`
  - `chrom_io.grandata_from_bigwigs`: creates a GRAnData from BigWigs and
    verifies expected shapes and values for the `X` track.
  - `GRAnData.get_dataframe("var")`: checks that `chrom`, `start`, `end`, and
    `region` metadata are persisted and round-trip cleanly.
  - `seq_io.add_genome_sequences_to_grandata` (backed): writes one-hot encoded
    DNA sequences into the backing store and verifies exact sequence decoding
    against a reference FASTA.

- `tests/test_module_alignment.py`
  - `GRAnDataModule` batching with `shuffle_dims=["obs"]`: validates that
    multiple arrays (`X`, `rna_tracks`) stay aligned after shuffling, and that
    sequence windows remain aligned to `var` entries.
  - `DNATransform` windowing: confirms the sliced sequence window decodes to the
    expected bases.

- `tests/test_split.py`
  - `train_val_test_split(strategy="chr_auto")`: ensures splits are created and
    include `train`, `val`, and `test` assignments for chromosome-based
    partitioning.

Notes
- The BigWig/FASTA tests require `pybigtools` and `pysam`; they are skipped if
  those modules are unavailable.
