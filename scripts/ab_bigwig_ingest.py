import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import grandata


def _write_simple_bigwig(path, chrom, size, value):
    import pybigtools

    bw = pybigtools.open(str(path), mode="w")
    bw.write(chroms={chrom: size}, vals=[(chrom, 0, size, float(value))])
    bw.close()


def _make_synthetic_inputs(tmp_root: Path, n_obs=4, n_var=100, genome_size=10000):
    bigwig_dir = tmp_root / "bigwigs"
    bigwig_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_obs):
        _write_simple_bigwig(bigwig_dir / f"obs{i}.bw", "chr1", genome_size, i + 1)

    starts = np.arange(0, n_var * 50, 50, dtype=int)
    ends = starts + 200
    regions = pd.DataFrame({"chrom": "chr1", "start": starts, "end": ends})
    regions_path = tmp_root / "regions.bed"
    regions.to_csv(regions_path, sep="\t", header=False, index=False)
    return bigwig_dir, regions_path


def _load_regions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2])
    df.columns = ["chrom", "start", "end"]
    return df


def _run_current(args, region_table, out_path):
    start = time.perf_counter()
    grandata.chrom_io.grandata_from_bigwigs(
        bigwig_dir=args.bigwig_dir,
        region_table=region_table,
        backed_path=out_path,
        array_name="X",
        obs_dim="obs",
        var_dim="var",
        seq_dim="seq_bins",
        target_region_width=args.target_region_width,
        bin_stat=args.bin_stat,
        tile_size=args.tile_size,
        chunk_size=args.chunk_size,
        n_bins=args.n_bins,
        fill_value=args.fill_value,
        obs_chunk_size=args.obs_chunk_size,
        n_workers=args.n_workers,
    )
    return time.perf_counter() - start


def _run_dask(args, region_table, out_path):
    start = time.perf_counter()
    grandata.chrom_io.grandata_from_bigwigs_dask(
        bigwig_dir=args.bigwig_dir,
        region_table=region_table,
        backed_path=out_path,
        array_name="X",
        obs_dim="obs",
        var_dim="var",
        seq_dim="seq_bins",
        target_region_width=args.target_region_width,
        bin_stat=args.bin_stat,
        chunk_size=args.chunk_size,
        n_bins=args.n_bins,
        fill_value=args.fill_value,
        obs_chunk_size=args.obs_chunk_size,
    )
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="A/B benchmark for bigwig ingestion.")
    parser.add_argument("--bigwig-dir", type=Path, help="Directory of BigWig files.")
    parser.add_argument("--regions", type=Path, help="BED file with chrom/start/end.")
    parser.add_argument("--out-dir", type=Path, default=Path("bench_out"))
    parser.add_argument("--mode", choices=("current", "dask", "both"), default="both")
    parser.add_argument("--scheduler", choices=("threads", "processes", "single"), default="threads")
    parser.add_argument("--target-region-width", type=int, default=2000)
    parser.add_argument("--bin-stat", type=str, default="mean")
    parser.add_argument("--tile-size", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--obs-chunk-size", type=int, default=16)
    parser.add_argument("--n-bins", type=int, default=1)
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--synthetic", action="store_true", help="Generate small synthetic data.")
    parser.add_argument("--synthetic-obs", type=int, default=4)
    parser.add_argument("--synthetic-vars", type=int, default=100)
    parser.add_argument("--synthetic-genome-size", type=int, default=10000)
    args = parser.parse_args()

    if args.synthetic:
        tmp_root = Path(tempfile.mkdtemp(prefix="grandata_bench_"))
        args.bigwig_dir, args.regions = _make_synthetic_inputs(
            tmp_root,
            n_obs=args.synthetic_obs,
            n_var=args.synthetic_vars,
            genome_size=args.synthetic_genome_size,
        )

    if args.bigwig_dir is None or args.regions is None:
        raise SystemExit("Provide --bigwig-dir and --regions, or use --synthetic.")

    region_table = _load_regions(args.regions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import dask
    except ImportError:
        dask = None

    if dask is not None:
        if args.scheduler == "single":
            dask.config.set(scheduler="single-threaded")
        else:
            dask.config.set(scheduler=args.scheduler)

    results = []

    if args.mode in ("current", "both"):
        out_path = args.out_dir / "current.zarr"
        duration = _run_current(args, region_table, out_path)
        results.append(("current", duration, out_path))

    if args.mode in ("dask", "both"):
        out_path = args.out_dir / "dask.zarr"
        duration = _run_dask(args, region_table, out_path)
        results.append(("dask", duration, out_path))

    for label, duration, path in results:
        print(f"{label}: {duration:.2f}s -> {path}")


if __name__ == "__main__":
    main()
