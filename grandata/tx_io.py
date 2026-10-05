import h5py
import gzip
import numpy as np
import pandas as pd
import xarray as xr
import sparse
import pybigtools
from tqdm import tqdm
from scipy.sparse import csr_matrix
from pathlib import Path
from itertools import product
import re
from typing import Union, Literal, List, Tuple, Dict, Iterator, Mapping, Sequence
from dataclasses import dataclass
import warnings
import dask.array as da
import weakref

from . import GRAnData
# from gtfparse import read_gtf


def read_gtf(gtf: str) -> pd.DataFrame:
    df = pd.read_csv(
        gtf,
        sep="\t",
        header=None,
        comment="#",
        dtype=str,        # keep everything as string
        names=[
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )

    fields = [
        "gene_id",
        "transcript_id",
        "exon_number",
        "gene",
        "gene_name",
        "gene_source",
        "gene_biotype",
        "transcript_name",
        "transcript_source",
        "transcript_biotype",
        "protein_id",
        "exon_id",
        "tag",
    ]

    for field in fields:
        pattern = rf"{field}\s+\"([^\"]+)\""
        df[field] = df["attribute"].str.extract(pattern)

    df.drop(columns="attribute", inplace=True)

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    return df


@dataclass(frozen=True)
class GeneLocus:
    """One painted gene locus; ``start``/``end`` are the GTF integer positions."""

    name: str
    chrom: str
    start: int
    end: int
    strand: str
    source: str  # "gene" row, or "transcripts" when derived from transcript spans

    @property
    def tss(self) -> int:
        return self.start if self.strand == "+" else self.end


@dataclass
class GeneAnnotation:
    """Loci for the requested genes, with what had to be inferred to get them."""

    loci: list[GeneLocus]
    requested: int
    unmatched: list[str]
    derived: list[str]
    multi_locus: list[str]
    gene_row_sequences: int
    transcript_sequences: int

    def summary(self) -> dict:
        return {
            "requested_gene_count": self.requested,
            "matched_gene_count": len({locus.name for locus in self.loci}),
            "unmatched_gene_count": len(self.unmatched),
            "derived_from_transcripts_gene_count": len(self.derived),
            "multi_locus_gene_count": len(self.multi_locus),
            "locus_count": len(self.loci),
            "gene_row_sequences": self.gene_row_sequences,
            "transcript_sequences": self.transcript_sequences,
        }


def load_gene_loci(
    gtf_file: str | Path,
    *,
    gene_names: Sequence[str],
    gtf_gene_field: str = "gene_name",
    gene_replace_dict: Mapping[str, str] | None = None,
    missing_gene_rows: Literal["derive", "skip", "error"] = "derive",
) -> GeneAnnotation:
    """Gene loci for ``gene_names`` from a GTF, the single source for tracks and masks.

    Names are matched after ``gene_replace_dict``. A name's ``gene`` rows are
    all kept (paralogs, or several genes renamed to one symbol, give several
    loci). Some GTFs (e.g. NCBI Mmul10) carry ``gene`` rows for only part of
    the genome; for genes with transcripts but no gene row,
    ``missing_gene_rows="derive"`` uses the span of the transcripts at the
    gene's most-transcribed locus, ``"skip"`` drops them and ``"error"`` raises.
    Inferences are reported as warnings and in :meth:`GeneAnnotation.summary`.
    """
    if missing_gene_rows not in ("derive", "skip", "error"):
        raise ValueError("missing_gene_rows must be 'derive', 'skip' or 'error'")
    wanted = {str(name) for name in gene_names}
    pattern = re.compile(rf'(?:^|;\s*){re.escape(gtf_gene_field)}\s+"([^"]+)"')
    gene_rows: dict[str, list[GeneLocus]] = {}
    spans: dict[tuple[str, str, str], list[int]] = {}
    gene_row_sequences: set[str] = set()
    transcript_sequences: set[str] = set()
    path = Path(gtf_file)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in ("gene", "transcript"):
                continue
            if fields[2] == "gene":
                gene_row_sequences.add(fields[0])
            else:
                transcript_sequences.add(fields[0])
            match = pattern.search(fields[8])
            if match is None:
                continue
            name = match.group(1)
            if gene_replace_dict is not None:
                name = gene_replace_dict.get(name, name)
            if name not in wanted:
                continue
            start, end = int(fields[3]), int(fields[4])
            if fields[2] == "gene":
                gene_rows.setdefault(name, []).append(GeneLocus(name, fields[0], start, end, fields[6], "gene"))
            else:
                key = (name, fields[0], fields[6])
                span = spans.setdefault(key, [start, end, 0])
                span[0], span[1], span[2] = min(span[0], start), max(span[1], end), span[2] + 1
    loci = [locus for rows in gene_rows.values() for locus in rows]
    only_transcripts = sorted({key[0] for key in spans} - set(gene_rows))
    if only_transcripts and missing_gene_rows == "error":
        raise ValueError(
            f"{len(only_transcripts)} genes have transcripts but no gene row in {gtf_file} "
            f"(e.g. {only_transcripts[:5]}); pass missing_gene_rows='derive' to use transcript spans"
        )
    derived: list[str] = []
    if missing_gene_rows == "derive":
        by_name: dict[str, list[tuple[tuple[str, str, str], list[int]]]] = {}
        for key, span in spans.items():
            by_name.setdefault(key[0], []).append((key, span))
        for name in only_transcripts:
            (_, chrom, strand), (start, end, _) = max(by_name[name], key=lambda item: item[1][2])
            loci.append(GeneLocus(name, chrom, start, end, strand, "transcripts"))
            derived.append(name)
    matched = {locus.name for locus in loci}
    counts: dict[str, int] = {}
    for locus in loci:
        counts[locus.name] = counts.get(locus.name, 0) + 1
    annotation = GeneAnnotation(
        loci=loci,
        requested=len(wanted),
        unmatched=sorted(wanted - matched),
        derived=derived,
        multi_locus=sorted(name for name, count in counts.items() if count > 1),
        gene_row_sequences=len(gene_row_sequences),
        transcript_sequences=len(transcript_sequences),
    )
    if transcript_sequences - gene_row_sequences:
        warnings.warn(
            f"{gtf_file}: gene rows cover {len(gene_row_sequences)} of {len(transcript_sequences)} sequences "
            f"with transcripts; {len(derived)} genes were "
            + ("derived from transcript spans" if missing_gene_rows == "derive" else "left without loci"),
            stacklevel=2,
        )
    elif derived:
        warnings.warn(f"{gtf_file}: {len(derived)} genes lack gene rows; derived from transcript spans", stacklevel=2)
    if annotation.multi_locus:
        warnings.warn(
            f"{len(annotation.multi_locus)} gene names have more than one locus "
            f"(e.g. {annotation.multi_locus[:5]}); each locus is painted with that name's value",
            stacklevel=2,
        )
    return annotation


def _paint_interval_mask(
    *,
    chroms: np.ndarray,
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    intervals_by_chrom: Mapping[str, Sequence[tuple[int, int]]],
    n_bins: int,
) -> np.ndarray:
    """Paint interval overlaps into a boolean region-by-bin matrix."""
    mask = np.zeros((len(chroms), n_bins), dtype=bool)
    sorted_intervals: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for chrom, intervals in intervals_by_chrom.items():
        values = np.asarray(sorted(intervals), dtype=np.int64)
        if values.size:
            starts = values[:, 0]
            ends = values[:, 1]
            sorted_intervals[chrom] = (starts, ends, int(np.max(ends - starts)))

    for region_index, (chrom, region_start, region_end) in enumerate(
        zip(chroms.astype(str), region_starts, region_ends)
    ):
        indexed = sorted_intervals.get(chrom)
        width = int(region_end) - int(region_start)
        if indexed is None or width <= 0:
            continue
        starts, ends, max_span = indexed
        candidate_start = int(
            np.searchsorted(starts, int(region_start) - max_span, side="right")
        )
        candidate_end = int(np.searchsorted(starts, int(region_end), side="left"))
        for interval_start, interval_end in zip(
            starts[candidate_start:candidate_end], ends[candidate_start:candidate_end]
        ):
            if int(interval_end) <= int(region_start):
                continue
            overlap_start = max(int(interval_start), int(region_start))
            overlap_end = min(int(interval_end), int(region_end))
            left = max(0, (overlap_start - int(region_start)) * n_bins // width)
            right = min(
                n_bins,
                ((overlap_end - int(region_start)) * n_bins + width - 1) // width,
            )
            if right > left:
                mask[region_index, left:right] = True
    return mask


def add_gtf_annotation_masks(
    adata: GRAnData,
    *,
    gtf_file: str | Path,
    gene_names: Sequence[str],
    gtf_gene_field: str = "gene_name",
    gene_replace_dict: Mapping[str, str] | None = None,
    tss_projection_bases: int = 1000,
    var_dim: str = "var",
    seq_dim: str = "seq_bins",
    gene_body_array_name: str = "gene_body_mask",
    rna_locus_array_name: str = "rna_locus_mask",
    rna_forward_locus_array_name: str = "rna_forward_locus_mask",
    rna_reverse_locus_array_name: str = "rna_reverse_locus_mask",
    chunk_size: int = 128,
    missing_gene_rows: Literal["derive", "skip", "error"] = "derive",
) -> GRAnData:
    """Add matched GTF gene-body and tx_io TSS-support masks to a GRAnData store.

    Loci come from :func:`load_gene_loci`, the same source ``write_tss_bigwigs``
    uses, so masks and RNA tracks always cover the same genes. GTF integer
    positions are used directly, and each RNA locus is
    ``[TSS, TSS + tss_projection_bases)``. Only genes represented by
    ``gene_names`` are painted. Never use the gene-body mask as the structural
    support of an RNA track; use ``rna_locus_array_name``.
    """
    if tss_projection_bases < 1:
        raise ValueError("tss_projection_bases must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for key in (f"{var_dim}-_-chrom", f"{var_dim}-_-start", f"{var_dim}-_-end"):
        if key not in adata:
            raise KeyError(f"GRAnData is missing required interval field: {key}")
    if seq_dim not in adata.sizes:
        raise KeyError(f"GRAnData is missing sequence dimension: {seq_dim}")

    annotation = load_gene_loci(
        gtf_file,
        gene_names=gene_names,
        gtf_gene_field=gtf_gene_field,
        gene_replace_dict=gene_replace_dict,
        missing_gene_rows=missing_gene_rows,
    )
    matched = [
        (locus.chrom, locus.start, locus.end, locus.tss, locus.tss + tss_projection_bases, locus.strand)
        for locus in annotation.loci
    ]
    body_intervals: dict[str, list[tuple[int, int]]] = {}
    locus_intervals: dict[str, list[tuple[int, int]]] = {}
    forward_locus_intervals: dict[str, list[tuple[int, int]]] = {}
    reverse_locus_intervals: dict[str, list[tuple[int, int]]] = {}
    for chrom, body_start, body_end, locus_start, locus_end, strand in matched:
        body_intervals.setdefault(chrom, []).append((body_start, body_end))
        locus_intervals.setdefault(chrom, []).append((locus_start, locus_end))
        strand_intervals = forward_locus_intervals if strand == "+" else reverse_locus_intervals
        strand_intervals.setdefault(chrom, []).append((locus_start, locus_end))

    raw_chroms = np.asarray(adata[f"{var_dim}-_-chrom"].values)
    chroms = np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
        for value in raw_chroms.tolist()
    ], dtype=object)
    region_starts = np.asarray(adata[f"{var_dim}-_-start"].values, dtype=np.int64)
    region_ends = np.asarray(adata[f"{var_dim}-_-end"].values, dtype=np.int64)
    n_bins = int(adata.sizes[seq_dim])
    gene_body_mask = _paint_interval_mask(
        chroms=chroms,
        region_starts=region_starts,
        region_ends=region_ends,
        intervals_by_chrom=body_intervals,
        n_bins=n_bins,
    )
    rna_locus_mask = _paint_interval_mask(
        chroms=chroms,
        region_starts=region_starts,
        region_ends=region_ends,
        intervals_by_chrom=locus_intervals,
        n_bins=n_bins,
    )
    rna_forward_locus_mask = _paint_interval_mask(
        chroms=chroms,
        region_starts=region_starts,
        region_ends=region_ends,
        intervals_by_chrom=forward_locus_intervals,
        n_bins=n_bins,
    )
    rna_reverse_locus_mask = _paint_interval_mask(
        chroms=chroms,
        region_starts=region_starts,
        region_ends=region_ends,
        intervals_by_chrom=reverse_locus_intervals,
        n_bins=n_bins,
    )
    provenance = {
        "gtf_file": str(gtf_file),
        "gtf_gene_field": gtf_gene_field,
        "tss_projection_bases": tss_projection_bases,
        "missing_gene_rows": missing_gene_rows,
        **annotation.summary(),
    }
    dataset = xr.Dataset({
        gene_body_array_name: xr.DataArray(
            da.from_array(gene_body_mask, chunks=(chunk_size, n_bins)),
            dims=(var_dim, seq_dim),
            attrs={**provenance, "annotation_kind": "outer_gene_range"},
        ),
        rna_locus_array_name: xr.DataArray(
            da.from_array(rna_locus_mask, chunks=(chunk_size, n_bins)),
            dims=(var_dim, seq_dim),
            attrs={**provenance, "annotation_kind": "tx_io_tss_projection"},
        ),
        rna_forward_locus_array_name: xr.DataArray(
            da.from_array(rna_forward_locus_mask, chunks=(chunk_size, n_bins)),
            dims=(var_dim, seq_dim),
            attrs={**provenance, "annotation_kind": "tx_io_forward_tss_projection"},
        ),
        rna_reverse_locus_array_name: xr.DataArray(
            da.from_array(rna_reverse_locus_mask, chunks=(chunk_size, n_bins)),
            dims=(var_dim, seq_dim),
            attrs={**provenance, "annotation_kind": "tx_io_reverse_tss_projection"},
        ),
    })
    source = getattr(adata, "encoding", {}).get("source")
    if source is None:
        result = adata.copy()
        result[gene_body_array_name] = dataset[gene_body_array_name]
        result[rna_locus_array_name] = dataset[rna_locus_array_name]
        result[rna_forward_locus_array_name] = dataset[rna_forward_locus_array_name]
        result[rna_reverse_locus_array_name] = dataset[rna_reverse_locus_array_name]
        return result
    dataset.to_zarr(source, mode="a")
    return GRAnData.open_zarr(source, consolidated=False)

def read_h5ad_selective_to_grandata(
    filename: Union[str, Path],
    mode: Literal["r", "r+"] = "r",
    selected_fields: List[str] = None,
    use_dask: bool = True,
    chunks: dict | None = None,
) -> GRAnData:
    """
    Based on similar function from ANTIPODE.
    Read just the specified top‐level AnnData fields (e.g. "X","obs","var","layers", etc.)
    from an .h5ad file via h5py, and return a GRAnData (xarray.Dataset).
    This version unpacks obs/var into -_- columns so we never pass a DataFrame
    into GRAnData.__init__. If use_dask is True, datasets are wrapped as
    dask arrays and sparse CSR matrices are stored as components.
    """
    selected_fields = selected_fields or ["X", "obs", "var"]

    def h5_tree(g):
        out = {}
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                out[k] = h5_tree(v)
            else:
                try:
                    out[k] = len(v)
                except TypeError:
                    out[k] = "scalar"
        return out

    def prune_tree(tree_dict, keep_keys):
        """
        Return a pruned version of `tree_dict` that includes only the top‐level
        keys in `keep_keys` (if present), along with their entire nested structure.
        """
        pruned = {}
        for k in keep_keys:
            if k in tree_dict:
                pruned[k] = tree_dict[k]
        return pruned

    def read_h5_to_dict(group, subtree, eager_groups=None):
        eager_groups = set(eager_groups or [])

        def helper(grp, sub, top_key=None):
            out = {}
            for k, v in sub.items():
                if isinstance(v, dict):
                    out[k] = (
                        helper(grp[k], v, top_key=k if top_key is None else top_key)
                        if (k in grp and isinstance(grp[k], h5py.Group))
                        else None
                    )
                else:
                    if k in grp and isinstance(grp[k], h5py.Dataset):
                        ds = grp[k]
                        if ds.shape == ():
                            out[k] = ds[()]
                        else:
                            if use_dask and (top_key not in eager_groups):
                                if ds.dtype.hasobject:
                                    out[k] = da.from_array(ds, chunks=ds.shape)
                                else:
                                    out[k] = da.from_array(ds, chunks=chunks or "auto")
                            else:
                                arr = ds[...]
                                if arr.dtype.kind == "S":
                                    # decode raw bytes to Unicode
                                    arr = arr.astype("U")
                                out[k] = arr
                    else:
                        out[k] = None
            return out
        return helper(group, subtree)

    def convert_to_dataframe(d: dict) -> pd.DataFrame:
        # infer length from first non‐dict value
        length = next((len(v) for v in d.values() if not isinstance(v, dict)), None)
        if length is None:
            raise ValueError("Cannot infer obs/var length")
        cols = {}
        for k, v in d.items():
            if isinstance(v, dict) and {"categories", "codes"} <= set(v):
                codes = np.asarray(v["codes"], int)
                cats = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in v["categories"]
                ]
                if len(codes) == length:
                    cols[k] = pd.Categorical.from_codes(codes, cats)
            elif isinstance(v, dict) and {"data", "indices", "indptr"} <= set(v):
                max_ind = max(v["indices"]) + 1 if len(v["indices"]) > 0 else 0
                shape = tuple(v.get("shape", (length, max_ind)))
                cols[k] = csr_matrix(
                    (v["data"], v["indices"], v["indptr"]), shape=shape
                )
            elif not isinstance(v, dict):
                arr = np.asarray(v)
                if arr.ndim == 1 and arr.shape[0] == length:
                    if arr.dtype.kind == "O":
                        arr = np.array(
                            [
                                x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x
                                for x in arr
                            ],
                            dtype="U",
                        )
                    if arr.dtype.kind == "S":
                        # decode raw bytes to Unicode
                        arr = arr.astype("U")
                    cols[k] = arr
        return pd.DataFrame(cols)

    # ————— Read HDF5 and prune ——————————————————————————————————

    f = h5py.File(filename, mode)
    full_tree = h5_tree(f)
    pruned = prune_tree(full_tree, selected_fields)
    raw = read_h5_to_dict(f, pruned, eager_groups={"obs", "var"})

    data_vars = {}
    coords = {}

    # — obs: unpack into coords + obs-_-col ——————————————————————————————
    if "obs" in raw:
        od = raw["obs"]
        idx = od.pop("_index", None)
        obs_df = convert_to_dataframe(od)
        if idx is not None:
            decoded_idx = []
            for x in idx:
                if isinstance(x, (bytes, np.bytes_)):
                    decoded_idx.append(x.decode("utf-8"))
                else:
                    decoded_idx.append(str(x))
            obs_df.index = decoded_idx
        coords["obs"] = obs_df.index.to_numpy()
    
        # unpack columns…
        for col in obs_df.columns:
            data_vars[f"obs-_-{col}"] = xr.DataArray(
                obs_df[col].values,
                dims=("obs",),
                coords={"obs": coords["obs"]},
            )
        data_vars["obs-_-index"] = xr.DataArray(coords["obs"], dims=("obs",))

    # — var: same pattern ——————————————————————————————————————————————
    if "var" in raw:
        vd = raw["var"]
        idx = vd.pop("_index", None)
        var_df = convert_to_dataframe(vd)
        if idx is not None:
            decoded_idx = []
            for x in idx:
                if isinstance(x, (bytes, np.bytes_)):
                    decoded_idx.append(x.decode("utf-8"))
                else:
                    decoded_idx.append(str(x))
            var_df.index = decoded_idx
        coords["var"] = var_df.index.to_numpy()
    
        for col in var_df.columns:
            data_vars[f"var-_-{col}"] = xr.DataArray(
                var_df[col].values,
                dims=("var",),
                coords={"var": coords["var"]},
            )
        data_vars["var-_-index"] = xr.DataArray(coords["var"], dims=("var",))

    # — X matrix ——————————————————————————————————————————————————
    if "X" in raw:
        xraw = raw["X"]
        if isinstance(xraw, dict) and {"data", "indices", "indptr"} <= set(xraw):
            # Keep CSR components as separate arrays; materialize on demand via helper.
            data_vars["X_data"] = xr.DataArray(xraw["data"], dims=("X_nnz",))
            data_vars["X_indices"] = xr.DataArray(xraw["indices"], dims=("X_nnz",))
            data_vars["X_indptr"] = xr.DataArray(xraw["indptr"], dims=("X_indptr",))
            shape = xraw.get("shape", (len(coords["obs"]), len(coords["var"])))
            data_vars["X_shape"] = xr.DataArray(np.asarray(shape), dims=("X_shape_dim",))
        else:
            data_vars["X"] = xr.DataArray(xraw, dims=("obs", "var"), coords=coords)

    # — layers/obsm/varm/obsp ——————————————————————————————————————————
    for grp in ("layers", "obsm", "varm", "obsp"):
        if grp in raw:
            for name, val in raw[grp].items():
                if val is None:
                    continue
                if isinstance(val, dict) and {"data", "indices", "indptr"} <= set(val):
                    # Keep CSR components; name them with the group prefix.
                    prefix = f"{grp}-_-{name}"
                    data_vars[f"{prefix}_data"] = xr.DataArray(val["data"], dims=(f"{prefix}_nnz",))
                    data_vars[f"{prefix}_indices"] = xr.DataArray(val["indices"], dims=(f"{prefix}_nnz",))
                    data_vars[f"{prefix}_indptr"] = xr.DataArray(val["indptr"], dims=(f"{prefix}_indptr",))
                    shape = val.get("shape")
                    if shape is None:
                        if grp == "layers":
                            shape = (len(coords["obs"]), len(coords["var"]))
                        elif grp == "obsm":
                            shape = (len(coords["obs"]),)
                        elif grp == "varm":
                            shape = (len(coords["var"]),)
                        else:
                            shape = (len(coords["obs"]), len(coords["obs"]))
                    data_vars[f"{prefix}_shape"] = xr.DataArray(np.asarray(shape), dims=(f"{prefix}_shape_dim",))
                    continue
                else:
                    arr = val

                if grp == "layers":
                    dims, c = ("obs", "var"), coords
                elif grp == "obsm":
                    d2 = f"obsm_{name}"
                    dims, c = ("obs", d2), {"obs": coords["obs"], d2: np.arange(arr.shape[1])}
                elif grp == "varm":
                    d2 = f"varm_{name}"
                    dims, c = ("var", d2), {"var": coords["var"], d2: np.arange(arr.shape[1])}
                else:  # obsp
                    d2 = f"obsp_{name}"
                    dims, c = ("obs", d2), {"obs": coords["obs"], d2: coords["obs"]}

                data_vars[f"{grp}-_-{name}"] = xr.DataArray(arr, dims=dims, coords=c)

    # ——— Finally, build and return GRAnData ——————————————————————
    ds = GRAnData(data_vars=data_vars, coords=coords)
    if use_dask:
        # Keep the HDF5 file open for lazy dask reads.
        ds.attrs["_h5py_file"] = f
        ds.attrs["_h5py_file_finalizer"] = weakref.finalize(ds, f.close)
    else:
        f.close()
    return ds


def materialize_csr_array(
    ds: xr.Dataset,
    prefix: str,
    dense: bool = False,
):
    """
    Materialize CSR components stored in the dataset into a scipy CSR matrix
    or a dense ndarray (if dense=True). Prefix examples: "X", "layers-_-counts".
    """
    data = ds[f"{prefix}_data"].data
    indices = ds[f"{prefix}_indices"].data
    indptr = ds[f"{prefix}_indptr"].data
    shape = tuple(ds[f"{prefix}_shape"].values.tolist())
    if hasattr(data, "compute"):
        data = data.compute()
    if hasattr(indices, "compute"):
        indices = indices.compute()
    if hasattr(indptr, "compute"):
        indptr = indptr.compute()
    csr_mat = csr_matrix((data, indices, indptr), shape=shape)
    if dense:
        return csr_mat.toarray()
    return csr_mat


def close_h5_backing(ds: xr.Dataset) -> None:
    """
    Close the backing HDF5 file for datasets created with use_dask=True.
    """
    f = ds.attrs.pop("_h5py_file", None)
    finalizer = ds.attrs.pop("_h5py_file_finalizer", None)
    if finalizer is not None:
        finalizer()
    elif f is not None:
        f.close()

def _sanitize_obs_name(name: str) -> str:
    return re.sub(" ", "_", re.sub("/", "-", str(name)))


def _isolated_loci(loci: list[GeneLocus], n_bases: int) -> list[int]:
    """Indices of loci whose TSS projection overlaps no other projection."""
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for index, locus in enumerate(loci):
        by_chrom.setdefault(locus.chrom, []).append((locus.tss, index))
    isolated = []
    for entries in by_chrom.values():
        entries.sort()
        for position, (tss, index) in enumerate(entries):
            before = entries[position - 1][0] if position > 0 else None
            after = entries[position + 1][0] if position + 1 < len(entries) else None
            if (before is None or tss - before >= n_bases) and (after is None or after - tss >= n_bases):
                isolated.append(index)
    return isolated


def write_tss_bigwigs(
    matrix: np.ndarray | xr.DataArray,
    var_names: list[str] | None,
    obs_names: list[str] | None,
    gtf_file: str,
    target_dir: str,
    gtf_gene_field: str = 'gene',
    n_bases: int = 1000,
    chromsizes: dict[str, int] = None,
    gene_replace_dict = None,
    missing_gene_rows: Literal["derive", "skip", "error"] = "derive",
    verify_loci: int = 200,
) -> dict:
    """
    Write signed TSS-aligned transcription bigWig files (1 per obs),
    merging any overlapping TSS intervals by summing their values.

    Gene loci come from :func:`load_gene_loci` (shared with
    ``add_gtf_annotation_masks``); every locus is painted with the value of its
    own gene, looked up by name. After writing, each file is read back at up to
    ``verify_loci`` loci whose projection overlaps no other, and a mismatch with
    the intended value raises. Returns the annotation summary.

    Parameters
    ----------
    matrix : np.ndarray | xr.DataArray
        Shape (n_obs, n_var), transcription values. If DataArray, obs/var names
        can be inferred from its coords.
    var_names : list[str] | None
        Names of genes, in the same order as matrix columns. Must be unique.
    obs_names : list[str] | None
        Names for each observation (e.g. clusters, pseudobulk sets).
    gtf_file : str
        Path to a gene annotation GTF.
    target_dir : str
        Folder where output .bw files are written.
    n_bases : int
        Number of bases downstream of the TSS to represent.
    chromsizes : dict[str, int], optional
        Chromosome sizes. If not provided, inferred from the loci.
    gene_replace_dict : dict
        Dictionary to convert GTF gene names to new names
    missing_gene_rows : {"derive", "skip", "error"}
        Genes with transcripts but no gene row; see :func:`load_gene_loci`.
    verify_loci : int
        Loci read back per file after writing; 0 disables the check.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = None
    if isinstance(matrix, xr.DataArray):
        obs_dim, var_dim = matrix.dims[:2]
        if obs_names is None:
            obs_names = matrix.coords[obs_dim].astype(str).tolist()
        if var_names is None:
            var_names = matrix.coords[var_dim].astype(str).tolist()
    if obs_names is None or var_names is None:
        raise ValueError("obs_names and var_names must be provided for ndarray inputs.")
    var_names = [str(name) for name in var_names]
    column_of = {name: index for index, name in enumerate(var_names)}
    if len(column_of) != len(var_names):
        raise ValueError("var_names must be unique")
    if len(var_names) != matrix.shape[1] or len(obs_names) != matrix.shape[0]:
        raise ValueError(f"matrix shape {tuple(matrix.shape)} does not match obs_names/var_names")

    annotation = load_gene_loci(
        gtf_file,
        gene_names=var_names,
        gtf_gene_field=gtf_gene_field,
        gene_replace_dict=gene_replace_dict,
        missing_gene_rows=missing_gene_rows,
    )
    loci = annotation.loci
    value_columns = np.asarray([column_of[locus.name] for locus in loci], dtype=np.int64)
    if chromsizes is None:
        chromsizes = {}
        for locus in loci:
            chromsizes[locus.chrom] = max(chromsizes.get(locus.chrom, 0), locus.end, locus.tss + n_bases)
    rng = np.random.default_rng(0)
    isolated = _isolated_loci(loci, n_bases)
    probes = rng.choice(isolated, size=min(verify_loci, len(isolated)), replace=False) if verify_loci else []

    for obs_idx, obs_name in enumerate(obs_names):
        obs_name = _sanitize_obs_name(obs_name)
        print('writing',obs_name)
        path = target_dir / f"{obs_name}.bw"

        if isinstance(matrix, xr.DataArray):
            row_vals = np.asarray(matrix.isel({obs_dim: obs_idx}).data).ravel()
        else:
            row_vals = np.asarray(matrix[obs_idx]).ravel()
        gene_vals = row_vals[value_columns]

        # 1) Build the raw interval list: (chrom, start, end, signed_value)
        raw_intervals: list[tuple[str, int, int, float]] = []
        for locus, value in zip(loci, gene_vals):
            start = locus.tss
            end = locus.tss + n_bases
            if start >= chromsizes.get(locus.chrom, 0):
                continue
            end = min(end, chromsizes[locus.chrom])
            raw_intervals.append((locus.chrom, start, end, float(value) * (1.0 if locus.strand == "+" else -1.0)))

        # 2) Group intervals by chromosome
        chrom_to_intervals: dict[str, list[tuple[int, int, float]]] = {}
        for chrom, s, e, v in raw_intervals:
            chrom_to_intervals.setdefault(chrom, []).append((s, e, v))

        # 3) For each chromosome, merge overlapping intervals via sweep-line
        merged_values: list[tuple[str, int, int, float]] = []
        for chrom, iv_list in chrom_to_intervals.items():
            events: list[tuple[int, float, int]] = []
            for s, e, v in iv_list:
                if e <= s:
                    continue
                events.append((s, +v, +1))
                events.append((e, -v, -1))
            events.sort(key=lambda x: x[0])
            # Count open intervals: with none open the value is exactly zero.
            # Trusting the running float sum left ~1e-15 residues that became
            # segments spanning every gap between genes.
            current_sum = 0.0
            open_count = 0
            prev_pos = None
            idx = 0
            n_events = len(events)
            while idx < n_events:
                pos = events[idx][0]
                if prev_pos is not None and pos > prev_pos and open_count > 0 and current_sum != 0.0:
                    merged_values.append((chrom, prev_pos, pos, current_sum))
                while idx < n_events and events[idx][0] == pos:
                    current_sum += events[idx][1]
                    open_count += events[idx][2]
                    idx += 1
                if open_count == 0:
                    current_sum = 0.0
                prev_pos = pos

        # 4) Write the merged intervals, sorted by chromosome and start
        merged_values.sort(key=lambda x: (x[0], x[1]))
        writer = pybigtools.open(str(path), mode='w')
        writer.write(chroms=chromsizes, vals=merged_values)
        writer.close()

        # 5) Read back isolated loci: each must carry its own gene's signed value.
        if len(probes):
            reader = pybigtools.open(str(path), mode="r")
            try:
                wrong = []
                for index in probes:
                    locus = loci[index]
                    if locus.tss >= chromsizes.get(locus.chrom, 0):
                        continue
                    expected = float(gene_vals[index]) * (1.0 if locus.strand == "+" else -1.0)
                    if not np.isfinite(expected):
                        continue
                    got = float(reader.values(locus.chrom, locus.tss, locus.tss + 1, missing=0.0)[0])
                    if not np.isclose(got, expected, rtol=1e-4, atol=1e-6):
                        wrong.append((locus.name, locus.chrom, locus.tss, expected, got))
            finally:
                reader.close()
            if wrong:
                raise RuntimeError(f"{path}: {len(wrong)} of {len(probes)} verified loci carry the wrong value, e.g. {wrong[:3]}")
    return annotation.summary()


def audit_tss_tracks(
    adata: GRAnData,
    *,
    expression: np.ndarray,
    gene_names: Sequence[str],
    obs_names: Sequence[str],
    gtf_file: str | Path,
    gtf_gene_field: str = "gene_name",
    gene_replace_dict: Mapping[str, str] | None = None,
    array_name: str = "rna_tracks",
    n_bases: int = 1000,
    probes: int = 500,
    missing_gene_rows: Literal["derive", "skip", "error"] = "derive",
    var_dim: str = "var",
    seq_dim: str = "seq_bins",
    obs_dim: str = "obs",
    seed: int = 0,
) -> dict:
    """Check a store's binned RNA tracks against the expression they were built from.

    For random isolated TSS loci that lie inside a stored region, compares the
    track's bin at the projection's midpoint with the gene's signed value in
    ``expression`` ``(obs, gene)``. Returns the matching fraction and examples
    of mismatches; a correctly built store matches ~1.0.
    """
    gene_names = [str(name) for name in gene_names]
    column_of = {name: index for index, name in enumerate(gene_names)}
    annotation = load_gene_loci(
        gtf_file, gene_names=gene_names, gtf_gene_field=gtf_gene_field,
        gene_replace_dict=gene_replace_dict, missing_gene_rows=missing_gene_rows,
    )
    loci = annotation.loci
    chroms = np.asarray([str(v) for v in np.asarray(adata[f"{var_dim}-_-chrom"].values).tolist()])
    starts = np.asarray(adata[f"{var_dim}-_-start"].values, dtype=np.int64)
    ends = np.asarray(adata[f"{var_dim}-_-end"].values, dtype=np.int64)
    n_bins = int(adata.sizes[seq_dim])
    store_obs = [str(v) for v in np.asarray(adata[f"{obs_dim}-_-index"].values).tolist()]
    obs_rows = [(store_obs.index(_sanitize_obs_name(name)), row) for row, name in enumerate(obs_names)
                if _sanitize_obs_name(name) in store_obs]
    if not obs_rows:
        raise ValueError("no obs_names match the store's obs")
    by_chrom = {chrom: np.flatnonzero(chroms == chrom) for chrom in np.unique(chroms)}
    rng = np.random.default_rng(seed)
    order = rng.permutation(_isolated_loci(loci, n_bases))
    checked, wrong = 0, []
    array = adata[array_name]
    for index in order:
        if checked >= probes:
            break
        locus = loci[index]
        middle = locus.tss + n_bases // 2
        pool = by_chrom.get(locus.chrom)
        if pool is None:
            continue
        inside = pool[(starts[pool] <= locus.tss) & (ends[pool] >= locus.tss + n_bases)]
        if inside.size == 0:
            continue
        region = int(inside[0])
        width = int(ends[region] - starts[region])
        bin_index = int((middle - starts[region]) * n_bins // width)
        store_row, expression_row = obs_rows[int(rng.integers(len(obs_rows)))]
        expected = float(np.nan_to_num(expression[expression_row, column_of[locus.name]]))
        expected *= 1.0 if locus.strand == "+" else -1.0
        got = float(np.nan_to_num(array[store_row, region, bin_index].values))
        checked += 1
        if not np.isclose(got, expected, rtol=1e-3, atol=1e-5):
            wrong.append({"gene": locus.name, "chrom": locus.chrom, "tss": locus.tss, "expected": expected, "got": got})
    return {
        "checked": checked,
        "matching_fraction": 1.0 - len(wrong) / max(checked, 1),
        "mismatches": wrong[:10],
        **annotation.summary(),
    }


def group_aggr_xr(
    ds: xr.Dataset,
    array_name: str,
    categories: Union[str, List[str]],
    agg_func=np.mean,
    normalize: bool = False,
    materialize: bool = False,
    progress: bool = False,
) -> xr.DataArray:
    """
    Group–aggregate an xarray.Dataset along 'obs' by one or more categorical
    obs columns, using xarray.groupby on the specified data array, and return
    a DataArray whose dimensions correspond to each category plus the var dimension.

    Parameters
    ----------
    ds
        An xarray.Dataset containing:
          - a DataArray `ds[array_name]` with dims ("obs","var") or similar,
          - one or more obs columns named "obs-_-<category>".
    array_name
        Name of the DataArray in `ds` to aggregate (e.g. "X", "layers-_-counts", "obsp-_-contacts").
    categories
        Single category name or list of names like obs-_-<category>).
    agg_func
        Aggregation function (e.g. np.mean, np.median, np.std).
    normalize
        If True, each observation is normalized by its row-sum before grouping.
    materialize
        If False, return the grouped DataArray without densifying or reshaping.
    progress
        If True, show a progress bar when computing dask-backed results.

    Returns
    -------
    xr.DataArray
        A DataArray with dimensions:
          - one dimension per category (named exactly as in `categories`),
          - plus the var dimension (same name as in `ds[array_name]`).
        The coords along each category axis are the observed levels of that category
        (in first-appearance order), and the coord along the var axis is carried
        over from the original DataArray.
    """
    # — normalize categories list —
    if isinstance(categories, str):
        categories = [categories]
    if not categories:
        raise ValueError("Must supply at least one category name")

    # — pick the DataArray and its dims —
    has_csr_components = f"{array_name}_data" in ds
    if array_name in ds:
        da = ds[array_name]
        obs_dim, var_dim = da.dims[:2]
        # capture the original var-axis coordinate
        var_coord = da.coords[var_dim]
    elif has_csr_components:
        da = None
        obs_dim, var_dim = "obs", "var"
        var_coord = ds.coords[var_dim]
    else:
        raise KeyError(f"No variable named '{array_name}' and no CSR components found.")

    # — collect category arrays & orders —
    category_orders = {}
    cat_arrs = []
    for cat in categories:
        arr = ds[cat].data
        if hasattr(arr, "compute"):
            arr = arr.compute()
        arr = np.asarray(arr).astype(str)
        # preserve first-appearance order
        seen = dict.fromkeys(arr.tolist())
        category_orders[cat] = list(seen.keys())
        cat_arrs.append(arr)

    # — build grouping labels —
    if len(categories) == 1:
        group_key = categories[0]
        group_labels = cat_arrs[0]
    else:
        sep = "____"
        combo = cat_arrs[0].astype("U")
        for arr in cat_arrs[1:]:
            combo = np.char.add(np.char.add(combo, sep), arr)
        group_key = sep.join(categories)
        group_labels = combo

    # — fast sparse aggregation path for CSR components —
    if has_csr_components and agg_func in (np.mean, np.std):
        csr_mat = materialize_csr_array(ds, array_name, dense=False)
        obs_dim, var_dim = "obs", "var"
        n_obs, n_vars = csr_mat.shape

        if normalize:
            row_sums = np.asarray(csr_mat.sum(axis=1)).ravel()
            inv = np.reciprocal(row_sums, where=row_sums != 0)
            csr_mat = csr_mat.multiply(inv[:, None])

        if len(categories) == 1:
            cat = categories[0]
            levels = category_orders[cat]
            level_index = {v: i for i, v in enumerate(levels)}
            col_idx = np.fromiter((level_index[v] for v in group_labels), dtype=int, count=n_obs)
            dims = [cat, var_dim]
            coords = {cat: levels, var_dim: ds.coords[var_dim]}
            n_groups = len(levels)
        else:
            lists_of_levels = [category_orders[c] for c in categories]
            all_combos = list(product(*lists_of_levels))
            combo_strs = [sep.join(c) for c in all_combos]
            combo_index = {v: i for i, v in enumerate(combo_strs)}
            col_idx = np.fromiter((combo_index[v] for v in group_labels), dtype=int, count=n_obs)
            dims = categories + [var_dim]
            coords = {var_dim: ds.coords[var_dim]}
            for cat in categories:
                coords[cat] = category_orders[cat]
            n_groups = len(combo_strs)
        rows = np.arange(n_obs, dtype=int)
        ones = np.ones(n_obs, dtype=float)
        g_mat = csr_matrix((ones, (rows, col_idx)), shape=(n_obs, n_groups))
        counts = np.bincount(col_idx, minlength=n_groups).astype(float)
        inv_counts = np.reciprocal(counts, where=counts != 0)
        sum_mat = g_mat.T @ csr_mat
        mean_mat = sum_mat.multiply(inv_counts[:, None])

        if agg_func is np.mean:
            result_mat = mean_mat
        else:
            sumsq_mat = g_mat.T @ csr_mat.multiply(csr_mat)
            mean_sq = sumsq_mat.multiply(inv_counts[:, None])
            var_mat = mean_sq - mean_mat.multiply(mean_mat)
            var_mat.data = np.maximum(var_mat.data, 0.0)
            var_mat.data = np.sqrt(var_mat.data)
            result_mat = var_mat

        return xr.DataArray(
            sparse.COO.from_scipy_sparse(result_mat),
            dims=dims,
            coords=coords,
        )

    # — build a combined grouping key (string) for xarray.groupby —
    grouping = xr.DataArray(group_labels, dims=obs_dim, coords={obs_dim: ds.coords[obs_dim]})

    # assign the grouping coordinate (internally) so we can group by it
    da = da.assign_coords(**{group_key: grouping})

    # — optional normalize each row by its sum —
    if normalize:
        da = da / da.sum(dim=var_dim, keep_attrs=True)

    # — groupby & reduce over obs_dim —
    grouped = da.groupby(group_key).reduce(agg_func, dim=obs_dim)
    if not materialize:
        return grouped
    if hasattr(grouped.data, "compute"):
        if progress:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                grouped = grouped.compute()
        else:
            grouped = grouped.compute()
    arr = np.asarray(grouped.data)

    # — reorder and reshape into (*category_sizes, n_vars) —
    n_vars = da.sizes[var_dim]
    if len(categories) == 1:
        cat = categories[0]
        levels = category_orders[cat]
        # the grouped index gives the observed levels in the grouping order
        observed = grouped[ group_key ].values.astype(str).tolist()
        idx = [levels.index(v) for v in observed]
        result = arr[idx, :]
        # dims and coords for the single‐category case
        dims = [cat, var_dim]
        coords = {
            cat: levels,
            var_dim: var_coord
        }
    else:
        # build the full cartesian product of category levels
        lists_of_levels = [category_orders[c] for c in categories]
        all_combos = list(product(*lists_of_levels))
        combo_strs = [sep.join(c) for c in all_combos]
        observed = grouped[group_key].values.astype(str).tolist()
        idx = [combo_strs.index(v) for v in observed]
        reshaped = arr[idx, :]
        sizes = [len(category_orders[c]) for c in categories]
        result = reshaped.reshape(*sizes, n_vars)
        # dims and coords for the multi‐category case
        dims = categories + [var_dim]
        coords = {var_dim: var_coord}
        for cat in categories:
            coords[cat] = category_orders[cat]

    # — construct and return the aggregated DataArray —
    return xr.DataArray(data=result, dims=dims, coords=coords)
