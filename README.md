<div align="center">
  <img src="assets/grandata_icon.png" alt="grandata logo!" width="200">
</div>

# grandata

A *lightweight* python library for efficiently organizing and dataloading multimodal genomics data from multiple datasets (e.g. species) simultaneously. Uses xarray with zarr backing and torchdata-nodes.


Supports reading bigwigs, fasta, h5 via chrom_io, seq_io and tx_io to create GRAnData objects, which are thinly wrapped xarray DataSets, and thus extensible via standard XArray operations.


# Docs

Practical usage guides live in [`docs/README.md`](/src/grandata/docs/README.md):

- [Single- and Multi-Species Data Preparation](/src/grandata/docs/guides/data-preparation/single-and-multi-species.md)
- [RNA and Extra Modalities](/src/grandata/docs/guides/data-preparation/rna-and-extra-modalities.md)
- [Loading Pre-Built `GRAnData` Zarr Stores](/src/grandata/docs/guides/loading/prebuilt-grandata-zarr.md)

# Installation

Begin with a fresh conda enviroment, as one always should.

`conda install poetry`

Install for using with any framework (install after):

`poetry install --extras=cpu`

or install for using with torch cuda:

`poetry install --extras=cuda --with cuda`

## What Is Still Missing

The repository now has guide-level documentation for the main workflows, but it still lacks:

- a rendered docs site or docs navigation
- a full API reference
- more end-to-end examples for advanced transforms and sampling

# TODO

- On disk sparse array access (currently auto-reads to memory)
- HiC Preprocessing
- Improved I/O efficiency beyond nodes prebatching and zarr async
