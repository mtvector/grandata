[<div align="center">
  <img src="assets/grandata_icon.png" alt="grandata logo!" width="200">
</div>](#)

# grandata

A *lightweight* python library for efficiently organizing and dataloading multimodal genomics data from multiple datasets (e.g. species) simultaneously. 

Stack on: zarr/icechunk -> xarray -> xbatcher -> torchdata-nodes -> ...

Supports reading bigwigs, fasta, h5 via chrom_io, seq_io and tx_io to create GRAnData objects, which are thinly wrapped xarray DataSets, and thus extensible via standard XArray operations.


# Installation

Begin with a fresh conda enviroment, as one always should.

`conda install poetry`

Install for using with any framework (install after):

`poetry install --extras=cpu`

or install for using with torch cuda:

`poetry install --extras=cuda --with cuda`
    

# TODO

- On disk sparse array access (currently auto-reads to memory)
- HiC Preprocessing
- Improved I/O efficiency beyond nodes prebatching and zarr async