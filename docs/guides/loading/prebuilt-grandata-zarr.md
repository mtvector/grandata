# Loading Pre-Built `GRAnData` Zarr Stores

This guide explains how to open an existing `grandata` `.zarr` store, inspect its contents, and build loaders from one or more pre-built objects.

## Overview

The main entry point for an existing store is:

```python
from grandata import GRAnData

adata = GRAnData.open_zarr("/path/to/store.zarr", consolidated=False)
```

From there you can:

- inspect available arrays and metadata
- recover `obs` and `var` tables
- load arrays eagerly into memory if needed
- create a `GRAnDataModule` for train/val/test/predict loading

## Open a backed `.zarr` store

Use `GRAnData.open_zarr(...)` rather than plain `xarray.open_zarr(...)` so `grandata` can restore any serialized sparse arrays correctly.

```python
from grandata import GRAnData

adata = GRAnData.open_zarr("/path/to/species.zarr", consolidated=False)
```

## Inspect the store

### List array names

```python
print(adata.array_names)
```

Typical outputs might include:

- `X`
- `sequences`
- `var-_-chrom`
- `var-_-start`
- `var-_-end`
- `var-_-split`
- `obs-_-index`
- `var-_-index`

### Recover `obs` and `var` metadata as DataFrames

`GRAnData` stores table-like metadata as `top-_-column` variables. Use `get_dataframe(...)` to recover them:

```python
obs_df = adata.get_dataframe("obs")
var_df = adata.get_dataframe("var")

print(obs_df.head())
print(var_df.head())
```

This is usually the easiest way to inspect:

- observation names
- genomic coordinates
- split labels

### Read dataset attributes

```python
print(adata.attrs)
```

When sequences have been added with `seq_io.add_genome_sequences_to_grandata(...)`, attributes may include:

- `genome_name`
- `genome_fasta`
- `genome_chrom_sizes`

## Access arrays directly

Like `xarray.Dataset`, `GRAnData` supports key-based access:

```python
signal = adata["X"]
sequence = adata["sequences"]

print(signal.dims, signal.shape)
print(sequence.dims, sequence.shape)
```

If the store is backed, these arrays remain lazily loaded until accessed.

## Load everything into memory

If you want to materialize arrays eagerly:

```python
adata.load()
```

This is optional. In most training workflows you should leave the store backed and let `GRAnDataModule` stream from Zarr.

## Build a loader from one pre-built store

The loader only needs:

1. a `GRAnData` object
2. the names of arrays to load
3. a batch dimension, usually `var`

```python
from grandata import GRAnDataModule

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

## Split-aware loading

By default, `GRAnDataModule` looks for split labels at `var-_-split`.

If that variable exists and contains values like `train`, `val`, and `test`, then:

```python
module.setup("train")
train_batch = next(iter(module.train_dataloader))

module.setup("val")
val_batch = next(iter(module.val_dataloader))
```

If the split variable is missing, the module loads all entries along `batch_dim`.

You can also point to a different split variable:

```python
module = GRAnDataModule(
    adatas=adata,
    batch_size=32,
    load_keys={"X": "signal"},
    split="custom-_-split",
)
```

## Loading multiple pre-built stores together

Pass a list of already-opened `GRAnData` objects:

```python
human = GRAnData.open_zarr("/data/human.zarr", consolidated=False)
mouse = GRAnData.open_zarr("/data/mouse.zarr", consolidated=False)

module = GRAnDataModule(
    adatas=[human, mouse],
    batch_size=32,
    load_keys={
        "X": "signal",
        "sequences": "sequence",
    },
    weights=[0.5, 0.5],
    batch_dim="var",
    join="inner",
)

module.setup("train")
batch = next(iter(module.train_dataloader))
```

For multiple stores:

- `weights` controls dataset-level sampling frequency
- `join` controls coordinate alignment on non-batch dimensions

## Merge multiple stores into one flat backed store

If you want a single mixed store instead of dataset-level sampling across multiple stores, use `concat_grandata(...)`.

This concatenates multiple `GRAnData` objects along one dimension, optionally adds a per-input label such as species identity, optionally shuffles that same dimension, writes a new zarr store, and returns the reopened backed merged object.

```python
from grandata import GRAnData, concat_grandata

human = GRAnData.open_zarr("/data/human.zarr", consolidated=False)
mouse = GRAnData.open_zarr("/data/mouse.zarr", consolidated=False)

merged = concat_grandata(
    [human, mouse],
    out_path="/data/merged.zarr",
    concat_dim="var",
    join="outer",
    add_key="var-_-species",
    add_values=["human", "mouse"],
    shuffle=True,
    random_state=42,
)
```

Typical uses:

- merge species-specific stores into one flat `var` axis
- preserve source identity with `add_key`
- use `join="outer"` to keep the union of non-concatenated coordinates
- use `join="inner"` to keep only the shared intersection

After merging, load the new store exactly like any other backed `GRAnData` object:

```python
from grandata import GRAnDataModule

module = GRAnDataModule(
    adatas=merged,
    batch_size=32,
    load_keys={
        "X": "signal",
        "sequences": "sequence",
        "var-_-species": "species",
    },
    batch_dim="var",
)

module.setup("train")
batch = next(iter(module.train_dataloader))
```

## Output batch structure

`GRAnDataModule` returns dictionaries keyed by the output names from `load_keys`.

Example:

```python
module = GRAnDataModule(
    adatas=adata,
    batch_size=16,
    load_keys={
        "X": "atac_tracks",
        "sequences": "sequence",
    },
)
```

Then each batch contains:

```python
{
    "atac_tracks": ...,
    "sequence": ...,
}
```

This makes it easier to keep model-facing names stable even when on-disk variable names differ.

## Arrays without the batch dimension

If a loaded array does not contain `batch_dim`, `GRAnDataModule` broadcasts it across the batch automatically.

That is useful for arrays like:

- per-observation summaries
- shared embeddings
- metadata matrices indexed by another dimension

## Optional transforms during loading

`GRAnDataModule` can apply transforms during batching. A common pattern is sequence windowing with `DNATransform`.

```python
from grandata import GRAnDataModule
from grandata.seq_io import DNATransform

sequence_transform = DNATransform(
    out_len=1024,
    random_rc=False,
    apply_states=("train", "val"),
)

module = GRAnDataModule(
    adatas=adata,
    batch_size=16,
    load_keys={
        "X": "signal",
        "sequences": "sequence",
    },
    transforms={
        "sequence": [sequence_transform],
    },
)
```

For paired sequence/target transforms, see the helper functions exposed from `grandata._module`:

- `make_paired_dna_target_transform`
- `make_rc_signflip_transform`
- `make_stateful_transform`

## Common inspection pattern

This is a good minimal sanity check after opening a store:

```python
from grandata import GRAnData

adata = GRAnData.open_zarr("/path/to/store.zarr", consolidated=False)

print(adata.array_names)
print(adata.get_dataframe("obs").head())
print(adata.get_dataframe("var").head())
print(adata["X"].shape)
print(adata.attrs)
```

## Troubleshooting

### Store opens, but `GRAnDataModule` errors on missing arrays

Check that every key in `load_keys` exists:

```python
print(adata.array_names)
```

### Batches are empty for `train` or `val`

Check whether `var-_-split` exists and contains the expected values:

```python
var_df = adata.get_dataframe("var")
print(var_df["split"].value_counts())
```

### Metadata is present but not as a DataFrame

`get_dataframe("obs")` and `get_dataframe("var")` only reconstruct variables following the `top-_-column` naming convention. If metadata was written under other names, access it directly as arrays.

### Multiple stores do not align cleanly

Try:

- using the same dimension names in all stores
- using the same variable names in all stores
- switching between `join="inner"` and `join="outer"`

## Minimal example

```python
from grandata import GRAnData, GRAnDataModule

adata = GRAnData.open_zarr("/data/species.zarr", consolidated=False)

module = GRAnDataModule(
    adatas=adata,
    batch_size=32,
    load_keys={"X": "signal", "sequences": "sequence"},
)

module.setup("train")
batch = next(iter(module.train_dataloader))
print(batch["signal"].shape)
```
