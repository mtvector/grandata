import os
import pandas as pd
from tqdm import tqdm
import pyfaidx

def create_sliding_bins(chromsizes: dict[str, int], bin_size: int, offset: int) -> pd.DataFrame:
    """
    Create sliding bins for each chromosome with a specified bin size and offset.

    For example, with bin_size=100 and offset=50, the bins on a chromosome will be:
      [1, 101], [51, 151], [101, 201], etc.
      
    Parameters
    ----------
    chromsizes : dict
        Dictionary mapping chromosome names to their lengths.
    bin_size : int
        The length of each bin.
    offset : int
        The step size to advance for each new bin. 
        (If offset < bin_size, bins will overlap.)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["chrom", "start", "end"], one row per bin.
    """
    rows = []
    for chrom, length in chromsizes.items():
        start = 1
        while start + bin_size <= length:
            rows.append([chrom, start, start + bin_size])
            start += offset
    bins_df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    return bins_df

def calculate_n_content(seq: str) -> float:
    """
    Calculate the fraction of 'N' bases in a DNA sequence.

    Parameters
    ----------
    seq : str
        The DNA sequence.

    Returns
    -------
    float
        Proportion of bases that are 'N'.
    """
    if len(seq) == 0:
        return 0.0
    return seq.upper().count('N') / len(seq)

def ranges_proportion_n(bed_df: pd.DataFrame, fasta_path: str) -> list[float]:
    """
    For each region in the BED DataFrame, compute the proportion of 'N's from the FASTA.

    Parameters
    ----------
    bed_df : pd.DataFrame
        DataFrame with columns ["chrom", "start", "end"].
    fasta_path : str
        Path to the FASTA file.

    Returns
    -------
    list[float]
        List with the proportion of N's for each region.
    """
    ref = pyfaidx.Fasta(fasta_path)
    results = []
    for _, row in tqdm(bed_df.iterrows(), total=bed_df.shape[0], desc="Calculating N content"):
        chrom, start, end = row["chrom"], row["start"], row["end"]
        # pyfaidx uses 0-based indexing; adjust accordingly.
        seq = ref[chrom][start - 1:end].seq
        results.append(calculate_n_content(seq))
    return results

def bin_genome(genome, bin_size: int, offset: int, n_threshold: float = 0.3, output_path: str = None) -> pd.DataFrame:
    """
    Generate sliding bins from a genome, compute N content in each bin,
    filter bins based on a threshold, and optionally write to a BED file.

    Parameters
    ----------
    genome : Genome
        A Genome object with attributes:
            - chrom_sizes: dict mapping chromosome names to lengths.
            - fasta: a pysam.FastaFile object whose filename attribute is the FASTA path.
    bin_size : int
        Size (in bases) of each bin.
    offset : int
        Step size for the sliding window.
    n_threshold : float, optional
        Maximum allowed proportion of 'N's per bin (default 0.3).
    output_path : str, optional
        If provided, writes the resulting bins (chrom, start, end) as a BED file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing bins that have an N content less than n_threshold,
        with columns ["chrom", "start", "end", "prop_n"].
    """
    bins = create_sliding_bins(genome.chrom_sizes, bin_size, offset)
    # Extract the FASTA file path (decoding bytes if necessary)
    fasta_path = (
        genome.fasta.filename.decode("utf-8")
        if isinstance(genome.fasta.filename, bytes)
        else genome.fasta.filename
    )
    n_props = ranges_proportion_n(bins, fasta_path)
    bins['prop_n'] = n_props
    # Filter out bins with too high an N content
    filtered_bins = bins[bins['prop_n'] < n_threshold].copy()
    
    if output_path:
        filtered_bins.loc[:, ["chrom", "start", "end"]].to_csv(
            output_path, header=False, index=False, sep='\t'
        )
    
    return filtered_bins
