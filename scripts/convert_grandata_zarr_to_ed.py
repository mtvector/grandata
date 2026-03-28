#!/usr/bin/env python3
"""
Convert a GRAnData zarr store to Enhancer-Designer zarr format.

GRAnData layout (xarray-backed zarr v3):
    X[obs, var, seq_bins]          float32  (samples × regions × signal_bins)
    sequences[var, seq_len, nuc]   uint8    (regions × seq_len × 4, one-hot ACGT)
    var-_-chrom[var]               str      chromosome names
    var-_-start[var]               int64    region start coordinates
    var-_-end[var]                 int64    region end coordinates
    var-_-split[var]               str      "train"/"val"/"test" labels (optional)
    var-_-species[var]             str      species labels (optional, for multi-species stores)
    obs-_-index[obs]               str      sample/celltype names

Enhancer-Designer layout (zarr v3):
    {species}/seq[N, L]            int8     A=0 C=1 G=2 T=3 N=4 pad=-1
    {species}/target[N, T, F]      float32  regions × signal_bins × samples
    {species}/id[N]                str      "{chrom}_{start}_{end}" identifiers
    {species}.attrs: celltype_names, n_celltypes
    tables/{species}_coords.parquet  chrom, start, end, id, left_pad, right_pad [, split]

Usage:
    python scripts/convert_grandata_zarr.py \\
        --input /path/to/grandata.zarr \\
        --output /path/to/output_root \\
        --species Homo_sapiens \\
        [--species-col var-_-species] \\
        [--signal-key X] \\
        [--seq-key sequences] \\
        [--chunk-size 64]

The output directory will contain:
    dataset.zarr/
    tables/
"""

import argparse
import sys
from pathlib import Path

import logging

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import Blosc

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Integer encoding: A=0, C=1, G=2, T=3, N=4 — matches enhancerdesigner convention
N_INDEX = 4


def _open_grandata(grandata_zarr: Path):
    """Open a GRAnData zarr store via grandata if available, else return None."""
    try:
        import grandata as gd  # noqa: PLC0415

        return gd.GRAnData.open_zarr(str(grandata_zarr), consolidated=False)
    except ImportError:
        logger.warning("grandata package not found; falling back to raw zarr open.")
        return None
    except Exception as exc:
        logger.warning(f"grandata.GRAnData.open_zarr failed ({exc}); falling back to raw zarr.")
        return None


def _get_var_table(grandata_zarr: Path) -> pd.DataFrame:
    """
    Return the var table (chrom, start, end, [split, species, ...]) from a GRAnData store.
    Falls back to reading raw zarr arrays if grandata is not installed.
    """
    adata = _open_grandata(grandata_zarr)
    if adata is not None:
        var_df = adata.get_dataframe("var")
        if var_df is None or var_df.empty:
            raise ValueError(f"No 'var' table found in {grandata_zarr}.")
        return var_df.reset_index(drop=True)

    # Fallback: reconstruct from raw zarr arrays named "var-_-*"
    root = zarr.open(str(grandata_zarr), mode="r")
    var_cols: dict[str, np.ndarray] = {}
    for key in root.array_keys():
        if key.startswith("var-_-"):
            col = key[len("var-_-"):]
            var_cols[col] = root[key][:]
    if not var_cols:
        raise ValueError(f"No 'var-_-*' arrays found in {grandata_zarr}.")
    df = pd.DataFrame(var_cols)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return df.reset_index(drop=True)


def _get_obs_names(grandata_zarr: Path) -> list[str]:
    """
    Return the obs index (sample/celltype names) from a GRAnData store.
    """
    adata = _open_grandata(grandata_zarr)
    if adata is not None:
        obs_df = adata.get_dataframe("obs")
        if obs_df is not None and not obs_df.empty:
            return list(obs_df.index)
        # Try coords in obs
        return [str(v) for v in adata["obs-_-index"].values]

    root = zarr.open(str(grandata_zarr), mode="r")
    if "obs-_-index" in root:
        return [str(v) for v in root["obs-_-index"][:]]
    raise ValueError(f"No 'obs-_-index' array found in {grandata_zarr}.")


def _check_array_exists(grandata_zarr: Path, key: str) -> bool:
    root = zarr.open(str(grandata_zarr), mode="r")
    return key in root


def _convert_one_species(
    grandata_zarr: Path,
    output_root: Path,
    species: str,
    var_mask: np.ndarray,
    obs_names: list[str],
    signal_key: str,
    seq_key: str,
    chunk_size: int,
) -> None:
    """
    Write one species group into the Enhancer-Designer zarr + coords parquet.

    Parameters
    ----------
    var_mask : boolean array of shape (total_N,) selecting the regions for this species.
    """
    src_root = zarr.open(str(grandata_zarr), mode="r")
    var_indices = np.where(var_mask)[0]
    N = len(var_indices)
    F = len(obs_names)

    logger.info(f"[{species}] N={N} regions, F={F} samples/celltypes")

    # ── Validate source arrays ────────────────────────────────────────────────
    has_signal = signal_key in src_root
    has_seq = seq_key in src_root

    if not has_signal:
        logger.warning(
            f"[{species}] Signal array '{signal_key}' not found in {grandata_zarr}. "
            "The 'target' array will be skipped."
        )
    if not has_seq:
        logger.warning(
            f"[{species}] Sequence array '{seq_key}' not found in {grandata_zarr}. "
            "The 'seq' array will be skipped. "
            "You can add sequences later using seq_io.add_genome_sequences_to_grandata()."
        )
    if not has_signal and not has_seq:
        raise ValueError(
            f"[{species}] Neither '{signal_key}' nor '{seq_key}' found in {grandata_zarr}. "
            "Nothing to convert."
        )

    # ── Infer dimensions ──────────────────────────────────────────────────────
    L: int | None = None
    T: int | None = None

    if has_seq:
        seq_src = src_root[seq_key]
        # Shape: (total_var, seq_len, 4)
        if seq_src.ndim != 3 or seq_src.shape[2] != 4:
            raise ValueError(
                f"Expected '{seq_key}' to have shape (var, seq_len, 4), "
                f"got {seq_src.shape}."
            )
        L = seq_src.shape[1]
        logger.info(f"[{species}] Sequence length L={L}")

    if has_signal:
        sig_src = src_root[signal_key]
        # Shape: (obs, total_var, seq_bins)  i.e. (F, total_N, T)
        if sig_src.ndim != 3:
            raise ValueError(
                f"Expected '{signal_key}' to have shape (obs, var, seq_bins), "
                f"got {sig_src.shape}."
            )
        T = sig_src.shape[2]
        logger.info(f"[{species}] Signal bins T={T}")

    # ── Open / create destination zarr group ──────────────────────────────────
    out_zarr = output_root / "dataset.zarr"
    out_zarr.mkdir(parents=True, exist_ok=True)
    dst_root = zarr.open_group(str(out_zarr), mode="a")

    if species in dst_root:
        logger.warning(f"[{species}] Group already exists in destination zarr — overwriting arrays.")
    grp = dst_root.require_group(species)

    comp = Blosc(cname="zstd", clevel=5, shuffle=2)
    B = chunk_size  # chunk along N

    # ── Pre-allocate destination arrays ───────────────────────────────────────
    if has_seq:
        if "seq" in grp:
            del grp["seq"]
        seq_dst = grp.create_array(
            "seq",
            shape=(N, L),
            chunks=(B, L),
            dtype="int8",
            compressor=comp,
        )
        logger.info(f"[{species}] Created seq array: {seq_dst.shape}")

    if has_signal:
        if "target" in grp:
            del grp["target"]
        tgt_dst = grp.create_array(
            "target",
            shape=(N, T, F),
            chunks=(B, T, min(512, F)),
            dtype="float32",
            compressor=comp,
        )
        tgt_dst.attrs["celltype_names"] = obs_names
        tgt_dst.attrs["n_celltypes"] = F
        logger.info(f"[{species}] Created target array: {tgt_dst.shape}")

    # ID array — "{chrom}_{start}_{end}"
    if "id" in grp:
        del grp["id"]

    # ── Write in chunks ───────────────────────────────────────────────────────
    for batch_start in range(0, N, B):
        batch_end = min(batch_start + B, N)
        src_idx = var_indices[batch_start:batch_end]

        if has_seq:
            # one-hot (b, L, 4) uint8 → integer (b, L) int8
            oh = seq_src.oindex[src_idx.tolist()]  # (b, L, 4)
            oh = oh.astype(np.uint8)
            indices = oh.argmax(axis=-1).astype(np.int8)  # A=0,C=1,G=2,T=3
            # All-zero rows → N (index 4)
            is_n = oh.sum(axis=-1) == 0  # (b, L)
            indices[is_n] = N_INDEX
            seq_dst[batch_start:batch_end] = indices

        if has_signal:
            # sig_src: (F, total_N, T) → select var axis → (F, b, T) → transpose → (b, T, F)
            sig_chunk = sig_src.oindex[:, src_idx.tolist(), :]  # (F, b, T)
            tgt_dst[batch_start:batch_end] = sig_chunk.transpose(1, 2, 0).astype(np.float32)

        logger.debug(
            f"[{species}] Written batch {batch_start}–{batch_end} / {N}"
        )

    # ── Group-level metadata ──────────────────────────────────────────────────
    grp.attrs["celltype_names"] = obs_names
    grp.attrs["n_celltypes"] = F

    logger.info(f"[{species}] Zarr arrays written.")


def _write_coords_parquet(
    output_root: Path,
    species: str,
    var_df: pd.DataFrame,
    var_mask: np.ndarray,
) -> None:
    """Write {species}_coords.parquet from the filtered var table."""
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    sub = var_df[var_mask].copy().reset_index(drop=True)

    required_cols = {"chrom", "start", "end"}
    missing = required_cols - set(sub.columns)
    if missing:
        raise ValueError(
            f"var table is missing required columns: {missing}. "
            f"Available columns: {list(sub.columns)}"
        )

    coords = pd.DataFrame({
        "chrom": sub["chrom"].astype(str),
        "start": sub["start"].astype(int),
        "end": sub["end"].astype(int),
        "id": sub["chrom"].astype(str) + "_"
              + sub["start"].astype(str) + "_"
              + sub["end"].astype(str),
        "left_pad": 0,
        "right_pad": 0,
    })

    if "split" in sub.columns:
        coords["split"] = sub["split"].values
        split_counts = coords["split"].value_counts().to_dict()
        logger.info(f"[{species}] Splits: {split_counts}")

    out_path = tables_dir / f"{species}_coords.parquet"
    coords.to_parquet(out_path, index=False)
    logger.info(f"[{species}] Coords parquet written: {out_path} ({len(coords)} rows)")


def convert(
    grandata_zarr: Path,
    output_root: Path,
    species_list: list[str],
    species_col: str | None,
    signal_key: str,
    seq_key: str,
    chunk_size: int,
) -> None:
    """Main conversion logic."""
    logger.info(f"Source: {grandata_zarr}")
    logger.info(f"Output: {output_root}")

    # ── Load var table ────────────────────────────────────────────────────────
    logger.info("Loading var table...")
    var_df = _get_var_table(grandata_zarr)
    logger.info(f"var table: {len(var_df)} regions, columns: {list(var_df.columns)}")

    # ── Load obs names ────────────────────────────────────────────────────────
    logger.info("Loading obs names...")
    obs_names = _get_obs_names(grandata_zarr)
    logger.info(f"obs names ({len(obs_names)}): {obs_names[:5]}{'...' if len(obs_names) > 5 else ''}")

    # ── Determine per-species region masks ────────────────────────────────────
    if len(species_list) == 1 and species_col is None:
        # Single species: all regions belong to this species
        species_masks = {species_list[0]: np.ones(len(var_df), dtype=bool)}
    else:
        if species_col is None:
            raise ValueError(
                "Multiple species requested but --species-col was not specified. "
                "Provide --species-col (e.g. 'var-_-species') to split regions by species."
            )
        if species_col not in var_df.columns:
            available = [c for c in var_df.columns if "species" in c.lower()]
            raise ValueError(
                f"Column '{species_col}' not found in var table. "
                f"Possible species columns: {available}. "
                f"All columns: {list(var_df.columns)}"
            )
        species_masks = {
            sp: (var_df[species_col] == sp).values
            for sp in species_list
        }
        for sp, mask in species_masks.items():
            n = mask.sum()
            if n == 0:
                logger.warning(
                    f"Species '{sp}' matched 0 regions in column '{species_col}'. "
                    f"Available values: {var_df[species_col].unique()[:10].tolist()}"
                )

    # ── Convert each species ──────────────────────────────────────────────────
    for species, mask in species_masks.items():
        n_regions = mask.sum()
        if n_regions == 0:
            logger.warning(f"[{species}] Skipping — 0 matching regions.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Converting species: {species} ({n_regions} regions)")
        logger.info(f"{'='*60}")

        _convert_one_species(
            grandata_zarr=grandata_zarr,
            output_root=output_root,
            species=species,
            var_mask=mask,
            obs_names=obs_names,
            signal_key=signal_key,
            seq_key=seq_key,
            chunk_size=chunk_size,
        )
        _write_coords_parquet(
            output_root=output_root,
            species=species,
            var_df=var_df,
            var_mask=mask,
        )

    logger.info("\nConversion complete.")
    logger.info(f"Output zarr:   {output_root / 'dataset.zarr'}")
    logger.info(f"Output tables: {output_root / 'tables'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GRAnData zarr store to Enhancer-Designer format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        metavar="GRANDATA_ZARR",
        help="Path to the source GRAnData zarr store.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        metavar="OUTPUT_ROOT",
        help=(
            "Output root directory. Will contain dataset.zarr/ and tables/. "
            "Created if it does not exist."
        ),
    )
    parser.add_argument(
        "-s", "--species",
        type=str,
        nargs="+",
        required=True,
        metavar="SPECIES",
        help=(
            "One or more species names to write as groups in the destination zarr "
            "(e.g. Homo_sapiens Mus_musculus). "
            "For a single-species store, any label can be used."
        ),
    )
    parser.add_argument(
        "--species-col",
        type=str,
        default=None,
        metavar="COL",
        help=(
            "Column in the var table that identifies species per region "
            "(e.g. 'var-_-species'). Required when --species lists more than one entry."
        ),
    )
    parser.add_argument(
        "--signal-key",
        type=str,
        default="X",
        metavar="KEY",
        help="Name of the signal array in the GRAnData zarr (default: 'X').",
    )
    parser.add_argument(
        "--seq-key",
        type=str,
        default="sequences",
        metavar="KEY",
        help="Name of the one-hot sequence array in the GRAnData zarr (default: 'sequences').",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        metavar="N",
        help="Number of regions per zarr chunk along the N axis (default: 64).",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input zarr not found: {args.input}")
        sys.exit(1)

    convert(
        grandata_zarr=args.input,
        output_root=args.output,
        species_list=args.species,
        species_col=args.species_col,
        signal_key=args.signal_key,
        seq_key=args.seq_key,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
