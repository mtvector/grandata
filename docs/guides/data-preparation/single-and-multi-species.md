# Single- and Multi-Species Data Preparation

This guide walks through preparing `grandata` `.zarr` stores from genomics inputs and then turning those stores into `GRAnDataModule` loaders for one species or multiple species together.

## Overview

`grandata` is built around two layers:

1. `GRAnData`: an `xarray.Dataset` wrapper backed by Zarr
2. `GRAnDataModule`: a fast loader for one or more pre-built `GRAnData` objects

The usual workflow is:

1. Build one `.zarr` store per species from BigWigs and genomic intervals
2. Optionally add DNA sequence arrays from a FASTA
3. Add train/validation/test split labels
4. Open the store with `GRAnData.open_zarr(...)`
5. Pass one or more `GRAnData` objects into `GRAnDataModule`

## Prerequisites

### Required Python packages

The core package expects:

- `xarray`
- `zarr`
- `dask`
- `pybigtools`
- `pysam`
- `pyfaidx`
- `torch`
- `torchdata`

Install from the repository root:

```bash
cd /src/grandata
poetry install --extras=cpu
```

## Input Data Requirements

For each species you typically need:

1. A genome FASTA file
2. A chromosome sizes file or FASTA index
3. A table of genomic regions
4. A directory of BigWig files, usually one file per observation, assay, or sample

## Single-Species Workflow

### 1. Prepare a region table

`grandata.chrom_io.grandata_from_bigwigs(...)` expects a pandas DataFrame with genomic intervals:

```python
import pandas as pd

region_table = pd.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [0, 1000, 0],
        "end": [1000, 2000, 1000],
    }
)
```

At minimum, the columns should be:

- `chrom`
- `start`
- `end`

There are two common ways to create this table.

### Option A: Load an existing BED-like table

If you already have regions on disk, load them into a DataFrame and keep only the interval columns:

```python
import pandas as pd

region_table = pd.read_csv(
    "/path/to/regions.bed",
    sep="\t",
    header=None,
    usecols=[0, 1, 2],
    names=["chrom", "start", "end"],
)
```

This is the simplest path if your peaks, windows, or consensus intervals have already been defined elsewhere.

### Option B: Generate bins with `grandata.bin_genome(...)`

Your earlier notebook uses `grandata` itself to generate a reusable region table from a genome FASTA plus chromosome sizes. That is the most direct built-in way to create the `region_table`.

```python
import grandata

MAX_SHIFT = 5000
WINDOW_SIZE = 50000
FULL_WINDOW_SIZE = WINDOW_SIZE + 2 * MAX_SHIFT
OFFSET = WINDOW_SIZE // 2
N_THRESHOLD = 0.3

genome = grandata.Genome(
    fasta="/path/to/genome.fa",
    chrom_sizes="/path/to/chrom.sizes",
)

region_table = grandata.bin_genome(
    genome,
    bin_size=FULL_WINDOW_SIZE,
    offset=OFFSET,
    n_threshold=N_THRESHOLD,
    output_path="/path/to/binned_genome.bed",
).reset_index(drop=True)
```

This matches the pattern in `/code/xscgem/data/create_spc_dataset.ipynb`:

- `FULL_WINDOW_SIZE` is the region width stored in the dataset
- `OFFSET` controls overlap between adjacent windows
- `n_threshold` filters out windows with too much ambiguous sequence

The resulting DataFrame includes:

- `chrom`
- `start`
- `end`
- `prop_n`

For `grandata_from_bigwigs(...)`, it is fine to pass the full DataFrame, but only the interval columns are required.

### Choosing window parameters

A practical pattern from the notebook is:

```python
MAX_SHIFT = 5000
WINDOW_SIZE = 50000
FULL_WINDOW_SIZE = WINDOW_SIZE + 2 * MAX_SHIFT
OFFSET = WINDOW_SIZE // 2
BIN_SIZE = 50
N_BINS = FULL_WINDOW_SIZE // BIN_SIZE
```

In that setup:

- the stored genomic interval width is `FULL_WINDOW_SIZE`
- model-time random cropping can later extract a smaller `WINDOW_SIZE`
- `BIN_SIZE` sets the signal resolution
- `N_BINS` is the number of bins in the target track array

### 2. Organize BigWig inputs

Point `grandata` at a directory of BigWig files:

```console
bigwigs/
├── sample_A.bw
├── sample_B.bw
└── sample_C.bw
```

Each file becomes an entry along the `obs` dimension.

### 3. Build the initial `GRAnData` store

Create a Zarr-backed object containing region metadata plus a first signal array:

```python
from pathlib import Path
from grandata import chrom_io

adata = chrom_io.grandata_from_bigwigs(
    region_table=region_table,
    bigwig_dir=Path("/path/to/bigwigs"),
    backed_path=Path("/path/to/human.zarr"),
    target_region_width=1000,
    array_name="X",
    obs_dim="obs",
    var_dim="var",
    seq_dim="seq_bins",
    bin_stat="mean",
    n_bins=8,
    chunk_size=512,
)
```

If you generated the region table with `FULL_WINDOW_SIZE` and chose a signal `BIN_SIZE`, then these arguments should usually line up as:

```python
target_region_width = FULL_WINDOW_SIZE
n_bins = FULL_WINDOW_SIZE // BIN_SIZE
```

This writes a backed `.zarr` store and returns a reopened `GRAnData` object.

The resulting object typically contains:

- `obs-_-index`: BigWig-derived observation names
- `var-_-index`: region names like `chr1:0-1000`
- `var-_-chrom`, `var-_-start`, `var-_-end`, `var-_-region`
- `X`: the binned signal array with dimensions `(obs, var, seq_bins)`

### 4. Add DNA sequences from a FASTA

If you want sequence inputs in the same store, add them after creating the backed object:

```python
from grandata import Genome, seq_io

genome = Genome(
    fasta="/path/to/genome.fa",
    chrom_sizes="/path/to/genome.chrom.sizes",
)

var_df = adata.get_dataframe("var")[["chrom", "start", "end"]]

adata = seq_io.add_genome_sequences_to_grandata(
    adata,
    ranges_df=var_df,
    genome=genome,
    key="sequences",
    dimnames=("var", "seq_len", "nuc"),
    backed=True,
)
```

This adds a `sequences` array with dimensions `(var, seq_len, nuc)`.

### 5. Add train/val/test splits

`GRAnDataModule` expects a split column by default at `var-_-split`.

You can create it directly on the object:

```python
from grandata import train_val_test_split

train_val_test_split(
    adata,
    strategy="chr_auto",
    val_size=0.1,
    test_size=0.1,
    random_state=42,
)

adata.to_zarr("/path/to/human.zarr", mode="a")
```

Supported split strategies include:

- `region`: random region-level splitting
- `chr`: explicit chromosome-based splitting
- `chr_auto`: automatically choose validation and test chromosomes

### 6. Build a single-species `GRAnDataModule`

Once the store contains the arrays you need and a split column, build the loader:

```python
from grandata import GRAnData, GRAnDataModule

adata = GRAnData.open_zarr("/path/to/human.zarr", consolidated=False)

module = GRAnDataModule(
    adatas=adata,
    batch_size=32,
    load_keys={
        "X": "signal",
        "sequences": "sequence",
    },
    batch_dim="var",
)

module.setup("train")
batch = next(iter(module.train_dataloader))

print(batch["signal"].shape)
print(batch["sequence"].shape)
```

## Multi-Species Workflow

`grandata` does not impose a single combined multi-species Zarr schema. The intended pattern is to build one `.zarr` store per species, then pass multiple `GRAnData` objects into one `GRAnDataModule`.

## Overview

For multi-species loading:

1. Build one species store at a time
2. Keep comparable arrays and dimension names across stores
3. Open all stores as `GRAnData`
4. Pass the list of stores to `GRAnDataModule`

If you prefer one flat mixed-species store instead, you can also merge multiple backed stores with `concat_grandata(...)`; see the loading guide for the full example in [`docs/guides/loading/prebuilt-grandata-zarr.md`](../loading/prebuilt-grandata-zarr.md).

### 1. Build one store per species

For example:

```python
import grandata
from grandata import chrom_io

species = ["human", "mouse"]
species_codes = {"human": 0, "mouse": 1}

MAX_SHIFT = 5000
WINDOW_SIZE = 50000
FULL_WINDOW_SIZE = WINDOW_SIZE + 2 * MAX_SHIFT
OFFSET = WINDOW_SIZE // 2
N_THRESHOLD = 0.3
BIN_SIZE = 50
N_BINS = FULL_WINDOW_SIZE // BIN_SIZE

human_genome = grandata.Genome("/data/human.fa", "/data/human.chrom.sizes")
mouse_genome = grandata.Genome("/data/mouse.fa", "/data/mouse.chrom.sizes")

human_regions = grandata.bin_genome(
    human_genome,
    bin_size=FULL_WINDOW_SIZE,
    offset=OFFSET,
    n_threshold=N_THRESHOLD,
).reset_index(drop=True)

mouse_regions = grandata.bin_genome(
    mouse_genome,
    bin_size=FULL_WINDOW_SIZE,
    offset=OFFSET,
    n_threshold=N_THRESHOLD,
).reset_index(drop=True)

human = chrom_io.grandata_from_bigwigs(
    region_table=human_regions,
    bigwig_dir="/data/human_bigwigs",
    backed_path="/data/human.zarr",
    target_region_width=FULL_WINDOW_SIZE,
    array_name="X",
    n_bins=N_BINS,
)

mouse = chrom_io.grandata_from_bigwigs(
    region_table=mouse_regions,
    bigwig_dir="/data/mouse_bigwigs",
    backed_path="/data/mouse.zarr",
    target_region_width=FULL_WINDOW_SIZE,
    array_name="X",
    n_bins=N_BINS,
)
```

Then add sequences and split labels to each store the same way as in the single-species case.

If you want a species indicator inside each dataset, the notebook pattern also works well:

```python
import xarray as xr
import numpy as np

human["var-_-species"] = xr.DataArray(
    np.repeat(species_codes["human"], human.sizes["var"]),
    dims="var",
)

mouse["var-_-species"] = xr.DataArray(
    np.repeat(species_codes["mouse"], mouse.sizes["var"]),
    dims="var",
)
```

That makes it easy to request a species label through `load_keys` later.

### 2. Reopen both backed stores

```python
from grandata import GRAnData

human = GRAnData.open_zarr("/data/human.zarr", consolidated=False)
mouse = GRAnData.open_zarr("/data/mouse.zarr", consolidated=False)
```

### 3. Create a multi-species loader

```python
from grandata import GRAnDataModule

module = GRAnDataModule(
    adatas=[human, mouse],
    batch_size=32,
    load_keys={
        "X": "signal",
        "sequences": "sequence",
        "var-_-species": "species",
    },
    batch_dim="var",
    weights=[0.5, 0.5],
    join="inner",
)

module.setup("train")
batch = next(iter(module.train_dataloader))
```

When multiple datasets are supplied, the module samples from them according to `weights`.

### 4. Coordinate alignment across species

The `join` argument controls how non-batch coordinates are aligned:

- `join="inner"`: keep only coordinates shared across all datasets
- `join="outer"`: take the union and fill missing entries with `NaN`

This matters when arrays have comparable dimensions besides the batch dimension, for example:

- shared `obs` names
- shared assay channels
- shared gene coordinates

### 5. Weighting and sampling

There are two different weighting mechanisms:

- `weights`: controls how often each dataset is sampled when multiple `GRAnData` objects are provided
- `sample_weights`: controls weighted sampling of elements along `batch_dim` within a dataset

Example:

```python
module = GRAnDataModule(
    adatas=[human, mouse],
    batch_size=16,
    load_keys={"X": "signal"},
    weights=[0.8, 0.2],
)
```

This biases dataset-level sampling toward the first species.

## Notes and Recommendations

### Keep schemas consistent across species

For the cleanest multi-species loading:

- use the same array names across stores
- use the same batch dimension, usually `var`
- use the same sequence length and binning settings where possible

### Use explicit output names in `load_keys`

`load_keys` maps stored array names to batch dictionary keys:

```python
load_keys = {
    "X": "atac",
    "sequences": "sequence",
}
```

This is useful when downstream code expects stable names independent of the on-disk schema.

### Add splits before training

If `var-_-split` is missing, the loader falls back to using the full `batch_dim`. In most training workflows you should write split labels explicitly before building the module.

## Minimal end-to-end example

```python
from grandata import GRAnData, GRAnDataModule, Genome, chrom_io, seq_io, train_val_test_split

# Build one species store
adata = chrom_io.grandata_from_bigwigs(
    region_table=region_table,
    bigwig_dir="/data/bigwigs",
    backed_path="/data/species.zarr",
    target_region_width=1000,
    array_name="X",
    n_bins=8,
)

genome = Genome("/data/genome.fa", "/data/genome.chrom.sizes")
var_df = adata.get_dataframe("var")[["chrom", "start", "end"]]
adata = seq_io.add_genome_sequences_to_grandata(adata, var_df, genome, backed=True)
train_val_test_split(adata, strategy="chr_auto", random_state=42)
adata.to_zarr("/data/species.zarr", mode="a")

# Load for training
adata = GRAnData.open_zarr("/data/species.zarr", consolidated=False)
module = GRAnDataModule(
    adatas=adata,
    batch_size=32,
    load_keys={"X": "signal", "sequences": "sequence"},
)
module.setup("train")
batch = next(iter(module.train_dataloader))
```
