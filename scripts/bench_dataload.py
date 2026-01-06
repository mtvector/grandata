import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import grandata


def _maybe_psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def _set_dask_scheduler(scheduler: str):
    try:
        import dask
    except ImportError:
        return
    if scheduler == "single":
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler=scheduler)
    config = dask.config.config
    dask_cfg = config.get("scheduler")
    num_workers = config.get("num_workers")
    pool = config.get("pool")
    print(f"dask_config scheduler={dask_cfg} num_workers={num_workers} pool={pool}")


def _parse_load_keys(text: str | None):
    if not text:
        return {"X": "X"}
    return json.loads(text)


def _parse_list(text: str | None):
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _track_peak_rss(process, sample_every: int, step: int, peak: int):
    if process is None:
        return peak
    if step % sample_every != 0:
        return peak
    rss = process.memory_info().rss
    return max(peak, rss)


def _run_profiled(func, args):
    import cProfile
    import pstats
    import io

    profile_out = args.profile_out
    profile_sort = args.profile_sort
    profile_top = args.profile_top

    pr = cProfile.Profile()
    pr.enable()
    func(args)
    pr.disable()

    if profile_out:
        pr.dump_stats(profile_out)
        print(f"profile_out={profile_out}")

    if args.profile_print or not profile_out:
        stream = io.StringIO()
        stats = pstats.Stats(pr, stream=stream).sort_stats(profile_sort)
        stats.print_stats(profile_top)
        print(stream.getvalue())


def bench_ingest(args):
    _set_dask_scheduler(args.scheduler)
    region_table = np.loadtxt(args.regions, dtype=object, ndmin=2)
    region_df = pd.DataFrame(
        region_table[:, :3], columns=["chrom", "start", "end"]
    )
    region_df["start"] = region_df["start"].astype(int)
    region_df["end"] = region_df["end"].astype(int)

    start = time.perf_counter()
    grandata.chrom_io.grandata_from_bigwigs(
        bigwig_dir=args.bigwig_dir,
        region_table=region_df,
        backed_path=args.out_path,
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
    elapsed = time.perf_counter() - start
    print(f"ingest_seconds={elapsed:.2f} out={args.out_path}")


def bench_loader(args):
    _set_dask_scheduler(args.scheduler)
    load_keys = _parse_load_keys(args.load_keys)
    shuffle_dims = _parse_list(args.shuffle_dims)
    sequence_vars = _parse_list(args.sequence_vars) or ["sequences"]

    adatas = [grandata.GRAnData.open_zarr(p, consolidated=False) for p in args.zarr]

    transforms = {}
    if args.dna_window is not None:
        dnatransform = grandata.seq_io.DNATransform(
            out_len=args.dna_window,
            random_rc=args.random_rc,
            max_shift=args.max_shift,
            apply_states=("train", "val"),
        )
        seq_out_keys = [load_keys.get(name, name) for name in sequence_vars]
        for key in seq_out_keys:
            transforms.setdefault(key, []).append(dnatransform)

    module = grandata.GRAnDataModule(
        adatas=adatas,
        batch_size=args.batch_size,
        load_keys=load_keys,
        transforms=transforms,
        shuffle_dims=shuffle_dims,
        split=args.split,
        batch_dim=args.batch_dim,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory if args.pin_memory else None,
    )
    module.setup(args.stage)
    loader = getattr(module, f"{args.stage}_dataloader")

    psutil = _maybe_psutil()
    process = psutil.Process(os.getpid()) if psutil else None
    peak_rss = 0

    start = time.perf_counter()
    for i, _batch in enumerate(loader):
        peak_rss = _track_peak_rss(process, args.mem_sample_every, i, peak_rss)
        if i + 1 >= args.n_batches:
            break
    elapsed = time.perf_counter() - start
    per_batch = elapsed / max(args.n_batches, 1)

    print(f"batches={args.n_batches} total_seconds={elapsed:.2f} per_batch_seconds={per_batch:.4f}")
    if process is not None:
        print(f"peak_rss_bytes={peak_rss}")


def bench_sequences(args):
    _set_dask_scheduler(args.scheduler)
    ranges = np.loadtxt(args.ranges, dtype=object, ndmin=2)
    ranges_df = pd.DataFrame(ranges[:, :3], columns=["chrom", "start", "end"])
    ranges_df["start"] = ranges_df["start"].astype(int)
    ranges_df["end"] = ranges_df["end"].astype(int)

    if args.zarr:
        adata = grandata.GRAnData.open_zarr(args.zarr, consolidated=False)
    else:
        adata = grandata.GRAnData(coords={"var": np.arange(len(ranges_df))})
        adata.attrs["chunk_size"] = args.chunk_size
        if args.out_path:
            adata.to_zarr(args.out_path, mode="w")
            adata = grandata.GRAnData.open_zarr(args.out_path, consolidated=False)

    genome = grandata.Genome(
        fasta=args.fasta,
        chrom_sizes=args.chromsizes if args.chromsizes else None,
    )

    start = time.perf_counter()
    _ = grandata.seq_io.add_genome_sequences_to_grandata(
        adata,
        ranges_df=ranges_df,
        genome=genome,
        key=args.key,
        seq_length=args.seq_length,
        backed=args.backed,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - start
    print(f"sequence_seconds={elapsed:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GRAnData data loading.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Benchmark BigWig ingestion to zarr.")
    ingest.add_argument("--bigwig-dir", required=True, type=Path)
    ingest.add_argument("--regions", required=True, type=Path, help="BED with chrom/start/end.")
    ingest.add_argument("--out-path", required=True, type=Path)
    ingest.add_argument("--target-region-width", type=int, required=True)
    ingest.add_argument("--bin-stat", default="mean")
    ingest.add_argument("--chunk-size", type=int, default=512)
    ingest.add_argument("--obs-chunk-size", type=int, default=16)
    ingest.add_argument("--n-bins", type=int, default=1)
    ingest.add_argument("--fill-value", type=float, default=0.0)
    ingest.add_argument("--scheduler", choices=("threads", "processes", "single"), default="threads")
    ingest.add_argument("--profile", action="store_true")
    ingest.add_argument("--profile-out", type=Path)
    ingest.add_argument("--profile-sort", default="cumtime")
    ingest.add_argument("--profile-top", type=int, default=25)
    ingest.add_argument("--profile-print", action="store_true")
    ingest.set_defaults(func=bench_ingest)

    loader = sub.add_parser("loader", help="Benchmark GRAnDataModule batch loading.")
    loader.add_argument("--zarr", nargs="+", required=True)
    loader.add_argument("--stage", choices=("train", "val", "test", "predict"), default="train")
    loader.add_argument("--load-keys", help="JSON mapping of data_vars to output names.")
    loader.add_argument("--batch-size", type=int, default=32)
    loader.add_argument("--batch-dim", default="var")
    loader.add_argument("--split", default="var-_-split")
    loader.add_argument("--shuffle-dims", help="Comma-separated dims to shuffle.")
    loader.add_argument("--sequence-vars", help="Comma-separated sequence var names.")
    loader.add_argument("--dna-window", type=int)
    loader.add_argument("--random-rc", action="store_true")
    loader.add_argument("--max-shift", type=int)
    loader.add_argument("--prefetch-factor", type=int, default=2)
    loader.add_argument("--pin-memory", default="")
    loader.add_argument("--n-batches", type=int, default=100)
    loader.add_argument("--mem-sample-every", type=int, default=10)
    loader.add_argument("--scheduler", choices=("threads", "processes", "single"), default="threads")
    loader.add_argument("--profile", action="store_true")
    loader.add_argument("--profile-out", type=Path)
    loader.add_argument("--profile-sort", default="cumtime")
    loader.add_argument("--profile-top", type=int, default=25)
    loader.add_argument("--profile-print", action="store_true")
    loader.set_defaults(func=bench_loader)

    seq = sub.add_parser("sequence", help="Benchmark FASTA sequence extraction.")
    seq.add_argument("--ranges", required=True, type=Path, help="BED with chrom/start/end.")
    seq.add_argument("--fasta", required=True, type=Path)
    seq.add_argument("--chromsizes", type=Path)
    seq.add_argument("--zarr", type=Path, help="Existing zarr to append sequences.")
    seq.add_argument("--out-path", type=Path, help="Write a new zarr before adding sequences.")
    seq.add_argument("--key", default="sequences")
    seq.add_argument("--seq-length", type=int)
    seq.add_argument("--batch-size", type=int)
    seq.add_argument("--chunk-size", type=int, default=128)
    seq.add_argument("--backed", action="store_true")
    seq.add_argument("--scheduler", choices=("threads", "processes", "single"), default="threads")
    seq.add_argument("--profile", action="store_true")
    seq.add_argument("--profile-out", type=Path)
    seq.add_argument("--profile-sort", default="cumtime")
    seq.add_argument("--profile-top", type=int, default=25)
    seq.add_argument("--profile-print", action="store_true")
    seq.set_defaults(func=bench_sequences)

    args = parser.parse_args()
    if args.profile:
        _run_profiled(args.func, args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
