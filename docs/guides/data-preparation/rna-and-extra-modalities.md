# Adding RNA and Other Modalities to `GRAnData`

This guide covers the other fields you can add to a `GRAnData` store beyond the initial genomic signal and DNA sequence arrays, with a focus on the RNA-related workflow used in `/code/xscgem/data/create_spc_dataset.ipynb`.

## Overview

After creating a base store with:

- genomic windows
- one or more BigWig-derived signal arrays
- optional DNA sequence arrays

you can enrich the dataset with:

- per-observation RNA summaries such as `rna_means` and `rna_stds`
- RNA-derived genomic track arrays such as `rna_tracks`
- helper metadata arrays such as `var-_-species`

The common pattern is:

1. Read RNA data from an `.h5ad`
2. Aggregate RNA expression over a grouping column
3. Align RNA observation names to the observation names already present in `GRAnData`
4. Store RNA summaries directly as arrays
5. Optionally write TSS-centered RNA BigWigs and ingest them back into the same `.zarr` store as another track tensor

## Prerequisites

The RNA workflow uses:

- `grandata.tx_io.read_h5ad_selective_to_grandata(...)`
- `grandata.tx_io.group_aggr_xr(...)`
- `grandata.tx_io.write_tss_bigwigs(...)`
- `grandata.chrom_io.add_bigwig_array(...)`

You will typically need:

1. An existing `GRAnData` `.zarr` store
2. An RNA `.h5ad`
3. A GTF annotation file
4. A mapping from the RNA grouping names to the `obs` names used by your BigWig-derived dataset

## Starting Point

Assume you already have a backed store:

```python
import grandata

adata = grandata.GRAnData.open_zarr("/path/to/species.zarr", consolidated=False)
```

and an RNA dataset:

```python
rna_h5ad = "/path/to/rna_data.h5ad"
gtf_file = "/path/to/genes.gtf"
```

## Step 1: Read RNA from `.h5ad`

Use selective loading so you do not have to materialize the entire AnnData object at once.

```python
rds = grandata.tx_io.read_h5ad_selective_to_grandata(
    rna_h5ad,
    selected_fields=["X", "obs", "var"],
)
```

This pulls the requested fields into a `GRAnData` object, preserving the `obs` and `var` tables and exposing the main expression matrix as `X`.

If you use lazy HDF5-backed reads and want to close the file afterwards, call:

```python
grandata.tx_io.close_h5_backing(rds)
```

## Step 2: Aggregate RNA by group

In the notebook workflow, RNA was grouped by an observation column such as `obs-_-Group`.

```python
import numpy as np

means = grandata.tx_io.group_aggr_xr(
    rds,
    "X",
    "obs-_-Group",
    agg_func=np.mean,
    materialize=False,
)

stds = grandata.tx_io.group_aggr_xr(
    rds,
    "X",
    "obs-_-Group",
    agg_func=np.std,
    materialize=False,
)
```

These aggregated arrays usually start with dimensions like:

- grouped observation category
- `var`

For downstream use, it is often convenient to rename them to explicit RNA-oriented dimensions:

```python
means = means.rename({"obs-_-Group": "obs", "var": "gene"})
stds = stds.rename({"obs-_-Group": "obs", "var": "gene"})
```

## Step 3: Densify if needed

Depending on how the source data are stored, the grouped arrays may still be sparse-backed. The notebook densified them before writing them into the final dataset.

```python
means.data = means.data.todense()
stds.data = stds.data.todense()
```

If the data are already dense, this step is unnecessary.

## Step 4: Align RNA observation names to the existing `obs`

This is one of the most important practical steps.

In the notebook, group names needed to be sanitized to match the names derived from the ATAC BigWig filenames:

```python
import re

sanitized_group_names = means.coords["obs"].data
sanitized_group_names = [re.sub("/", "-", x) for x in sanitized_group_names]
sanitized_group_names = [re.sub(" ", "_", x) for x in sanitized_group_names]

means.coords["obs"] = sanitized_group_names
stds.coords["obs"] = sanitized_group_names
```

Then restrict everything to the shared set of observations:

```python
shared_obs = list(set(sanitized_group_names) & set(adata.coords["obs"].data))

means = means.sel(obs=shared_obs)
stds = stds.sel(obs=shared_obs)
adata = adata.sel(obs=shared_obs)
```

This keeps the RNA summaries aligned with the already-built genomic signal arrays.

## Step 5: Store RNA summary arrays directly

You can attach these grouped RNA summaries directly to the dataset:

```python
adata["rna_means"] = means
adata["rna_stds"] = stds

adata.to_zarr(adata.encoding["source"], mode="a")
adata = grandata.GRAnData.open_zarr(adata.encoding["source"], consolidated=False)
```

Typical dimensions are:

- `rna_means`: `(obs, gene)`
- `rna_stds`: `(obs, gene)`

These arrays are useful for models that need non-genomic per-observation covariates or auxiliary targets.

## Step 6: Write RNA-derived TSS BigWigs

If you want RNA represented as a genomic track over the same windows as the ATAC or chromatin signal arrays, one practical route is:

1. aggregate RNA by group
2. project each gene’s value to a TSS-centered interval
3. write one BigWig per observation
4. ingest those BigWigs back into the `GRAnData` object

That is exactly what `write_tss_bigwigs(...)` is for.

### Basic example

```python
grandata.tx_io.write_tss_bigwigs(
    means,
    obs_names=shared_obs,
    var_names=means.coords["gene"].values,
    gtf_file=gtf_file,
    target_dir="/path/to/rna_bigwigs",
    gtf_gene_field="gene_name",
)
```

Important arguments:

- `matrix`: usually the grouped RNA means
- `obs_names`: one output BigWig is written per observation
- `var_names`: gene names aligned to the matrix columns
- `gtf_file`: annotation used to place values at gene TSSs
- `gtf_gene_field`: the GTF column to match against your RNA gene names
- `gene_replace_dict`: optional gene-name remapping dictionary if your RNA gene IDs and GTF gene names differ

### Example with gene remapping

The notebook used per-species gene-key logic and optional replacement dictionaries:

```python
gene_key = "gene_name"
gene_replace_dict = None

grandata.tx_io.write_tss_bigwigs(
    means,
    obs_names=shared_obs,
    var_names=means.coords["gene"].values,
    gtf_file=gtf_file,
    target_dir="/path/to/rna_bigwigs",
    gtf_gene_field=gene_key,
    gene_replace_dict=gene_replace_dict,
)
```

## Step 7: Add RNA BigWigs back as a genomic track array

Once the RNA TSS BigWigs have been written, you can add them back to the same store as another array over `(obs, var, seq_bins)`.

```python
region_table = adata.get_dataframe("var")[["chrom", "start", "end"]]

adata = grandata.chrom_io.add_bigwig_array(
    adata,
    region_table=region_table,
    bigwig_dir="/path/to/rna_bigwigs",
    array_name="rna_tracks",
    obs_dim="obs",
    var_dim="var",
    seq_dim="seq_bins",
    target_region_width=adata.attrs["target_region_width"],
    bin_stat="mean",
    chunk_size=adata.attrs.get("chunk_size", 64),
    n_bins=adata.attrs["n_bins"],
)
```

This gives you a second track tensor aligned to the same regions as the original genomic signal.

## Step 8: Add helper metadata arrays

The notebook also added species codes directly onto the `var` dimension:

```python
import numpy as np
import xarray as xr

species_code = 0

adata["var-_-species"] = xr.DataArray(
    np.repeat(species_code, adata.sizes["var"]),
    dims="var",
)
```

This is useful when:

- combining multiple species in one `GRAnDataModule`
- training a model that should know which species a region came from

You can add similar helper arrays for:

- region classes
- chromosome indices
- motif-derived annotations
- QC scores

as long as the dimensions line up with the intended axis.

## Loading all modalities together

After writing the updated store, you can load the extra fields directly through `GRAnDataModule`.

```python
from grandata import GRAnDataModule
from grandata.seq_io import DNATransform

transform = DNATransform(
    out_len=50000,
    random_rc=True,
    max_shift=5000,
    apply_states=("train", "val"),
)

module = GRAnDataModule(
    adatas=adata,
    batch_size=48,
    load_keys={
        "X": "atac_tracks",
        "sequences": "sequence",
        "rna_tracks": "rna_tracks",
        "rna_means": "rna_means",
        "rna_stds": "rna_stds",
        "var-_-species": "species",
    },
    transforms={"sequence": [transform]},
    shuffle_dims=["obs"],
)

module.setup("train")
batch = next(iter(module.train_dataloader))
```

In that setup:

- `atac_tracks` and `rna_tracks` are genomic track tensors
- `sequence` is the DNA input
- `rna_means` and `rna_stds` are observation-level RNA summaries
- `species` is a per-region metadata field

## Recommended pattern

For multimodal datasets, a good progression is:

1. Build the base genomic dataset first
2. Reopen the backed store
3. Add derived fields one at a time
4. Write back to the same `.zarr`
5. Reopen again before building loaders

This keeps each augmentation step easy to validate.

## Minimal end-to-end example

```python
import grandata
import numpy as np
import re

adata = grandata.GRAnData.open_zarr("/data/species.zarr", consolidated=False)
rds = grandata.tx_io.read_h5ad_selective_to_grandata("/data/rna.h5ad", selected_fields=["X", "obs", "var"])

means = grandata.tx_io.group_aggr_xr(rds, "X", "obs-_-Group", agg_func=np.mean, materialize=False)
stds = grandata.tx_io.group_aggr_xr(rds, "X", "obs-_-Group", agg_func=np.std, materialize=False)

means = means.rename({"obs-_-Group": "obs", "var": "gene"})
stds = stds.rename({"obs-_-Group": "obs", "var": "gene"})

means.data = means.data.todense()
stds.data = stds.data.todense()

obs_names = [re.sub(" ", "_", re.sub("/", "-", x)) for x in means.coords["obs"].data]
means.coords["obs"] = obs_names
stds.coords["obs"] = obs_names

shared_obs = list(set(obs_names) & set(adata.coords["obs"].data))
adata = adata.sel(obs=shared_obs)
means = means.sel(obs=shared_obs)
stds = stds.sel(obs=shared_obs)

adata["rna_means"] = means
adata["rna_stds"] = stds

grandata.tx_io.write_tss_bigwigs(
    means,
    obs_names=shared_obs,
    var_names=means.coords["gene"].values,
    gtf_file="/data/genes.gtf",
    target_dir="/data/rna_bigwigs",
    gtf_gene_field="gene_name",
)

adata = grandata.chrom_io.add_bigwig_array(
    adata,
    region_table=adata.get_dataframe("var")[["chrom", "start", "end"]],
    bigwig_dir="/data/rna_bigwigs",
    array_name="rna_tracks",
    obs_dim="obs",
    var_dim="var",
    seq_dim="seq_bins",
    target_region_width=adata.attrs["target_region_width"],
    n_bins=adata.attrs["n_bins"],
)

adata.to_zarr(adata.encoding["source"], mode="a")
```

## Notes

- `write_tss_bigwigs(...)` writes one BigWig per observation, so the output directory should mirror the observation names you want in the final dataset.
- If RNA and genomic observation names do not already match, name sanitation and intersection are mandatory.
- If your gene symbols differ between species or between the RNA matrix and the GTF, use `gene_replace_dict`.
- The extra arrays do not need to be written all at once; incremental augmentation is often easier to debug.
