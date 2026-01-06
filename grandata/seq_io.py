"""seq_io.py contains sequence loading and transform utilities"""

import numpy as np
import pandas as pd
import xarray as xr
import json
from tqdm import tqdm

# Create a hot encoding table for DNA nucleotides (A, C, G, T)
def get_hot_encoding_table(
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: int = 0,
    dtype=np.uint8,
) -> np.ndarray:
    """Return a (256 x len(alphabet)) table mapping ASCII byte values to one-hot encodings."""
    def str_to_uint8(string: str) -> np.ndarray:
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)
    
    table = np.zeros((256, len(alphabet)), dtype=dtype)
    eye = np.eye(len(alphabet), dtype=dtype)
    # set uppercase and lowercase for the nucleotides
    table[str_to_uint8(alphabet.upper())] = eye
    table[str_to_uint8(alphabet.lower())] = eye
    # assign neutral bases (if any) to be all zeros (or neutral_value)
    table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    table[str_to_uint8(neutral_alphabet.lower())] = neutral_value
    return table

HOT_ENCODING_TABLE = get_hot_encoding_table()
HOT_DECODING_TABLE = np.array([ord("N")] * 16, dtype=np.uint8)
HOT_DECODING_TABLE[1] = ord("A")
HOT_DECODING_TABLE[2] = ord("C")
HOT_DECODING_TABLE[4] = ord("G")
HOT_DECODING_TABLE[8] = ord("T")


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence using the pre-computed HOT_ENCODING_TABLE.
    
    Parameters
    ----------
    sequence : str
        The DNA sequence to encode.
        
    Returns
    -------
    np.ndarray
        A (seq_length, 4) numpy array of type np.uint8.
    """
    # Convert sequence to its byte representation and look up the encoding for each character.
    return HOT_ENCODING_TABLE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def create_one_hot_encoded_array(
    ranges_df: pd.DataFrame, genome, seq_length: int = None
) -> xr.DataArray:
    """
    Create a one-hot encoded xarray DataArray from genomic ranges.
    
    Parameters
    ----------
    ranges_df : pd.DataFrame
        DataFrame with genomic ranges. Expected columns: 'chrom', 'start', 'end', and optionally 'strand'.
    genome : Genome
        A Genome object (from _genome.py) that provides a fetch(chrom, start, end, strand) method.
    seq_length : int, optional
        The sequence length to fetch. If None, it is inferred as (end - start) from the first row.
    
    Returns
    -------
    xr.DataArray
        One-hot encoded sequences with dimensions [var, seq_len, 4].
    """
    sequences = []
    
    # Infer sequence length if not provided.
    if seq_length is None:
        seq_length = int(ranges_df.iloc[0]['end']) - int(ranges_df.iloc[0]['start'])
    
    for idx, row in tqdm(ranges_df.iterrows(),total=ranges_df.shape[0]):
        chrom = row['chrom']
        start = int(row['start'])
        end = int(row['end'])
        if (end - start) != seq_length:
            raise ValueError(f"Row {idx} has length {(end - start)} which differs from expected {seq_length}.")
        # Default strand is '+' if not provided.
        strand = row.get('strand', '+') if 'strand' in row else '+'
        
        # Fetch the sequence using the Genome object. The fetch method should accept chrom, start, end, and strand.
        seq = genome.fetch(chrom, start, end, strand=strand)
        
        # Ensure the sequence is of the expected length; pad with 'N' if needed.
        if len(seq) < seq_length:
            seq = seq.ljust(seq_length, 'N')
        elif len(seq) > seq_length:
            seq = seq[:seq_length]
        
        # One-hot encode the sequence.
        one_hot = one_hot_encode_sequence(seq)  # shape (seq_length, 4)
        sequences.append(one_hot)
    
    # Stack all sequences into a numpy array of shape (n_ranges, seq_length, 4)
    encoded_array = np.stack(sequences, axis=0)
    da = xr.DataArray(encoded_array)
    return da

def add_genome_sequences_to_grandata(
    adata: xr.Dataset,
    ranges_df: pd.DataFrame,
    genome,
    key: str = "sequences",
    dimnames = ('var','seq_len','nuc',),
    seq_length: int = None,
    backed: bool = True,
    batch_size: int | None = None,
) -> xr.Dataset:
    """
    Create a one-hot encoded array of genomic sequences using the provided genome and ranges,
    add it to the GRAnData (an xarray.Dataset) under the specified key (default 'sequences'),
    and store genome metadata in the Dataset's attributes for later retrieval.
    
    Parameters
    ----------
    adata : xr.Dataset
        GRAnData object represented as an xarray.Dataset.
    ranges_df : pd.DataFrame
        DataFrame with genomic ranges. Expected columns: 'chrom', 'start', 'end', and optionally 'strand'.
    genome
        Genome object (from _genome.py) with a fetch method to extract sequences.
    key : str, optional
        Key under which to store the one-hot encoded array in adata. Default is 'sequences'.
    seq_length : int, optional
        The sequence length to fetch. If None, it is inferred from ranges_df.
    dimnames: list or tuple of length 3 naming your var, and sequence length and nucleotide dimensions.
    backed: If True and adata has a backing store, write sequences via dask/xarray to zarr
        without loading all sequences into memory.
    batch_size: Number of regions to process per batch when streaming.
    
    Returns
    -------
    xr.Dataset
        Updated GRAnData object with the one-hot encoded array added as a data variable and genome metadata stored in attrs.
    """
    if seq_length is None:
        seq_length = int(ranges_df.iloc[0]['end']) - int(ranges_df.iloc[0]['start'])
    if batch_size is None:
        batch_size = adata.attrs.get('chunk_size', 128)

    if backed and "source" in getattr(adata, "encoding", {}):
        try:
            import dask.array as da
        except ImportError as exc:
            raise ImportError("add_genome_sequences_to_grandata requires dask when backed=True.") from exc

        fasta_path = str(getattr(genome, "_fasta", None)) if hasattr(genome, "_fasta") else None
        if fasta_path is None:
            raise ValueError("Genome FASTA path is required for backed sequence writing.")

        chrom_col = ranges_df["chrom"].astype(str).to_numpy()
        start_col = ranges_df["start"].astype(int).to_numpy()
        end_col = ranges_df["end"].astype(int).to_numpy()
        if "strand" in ranges_df.columns:
            strand_col = ranges_df["strand"].astype(str).to_numpy()
        else:
            strand_col = np.full(ranges_df.shape[0], "+", dtype=object)

        def _load_block(_block, block_info=None):
            info = block_info[0]["array-location"]
            var_loc = info[0]
            var_slice = var_loc if isinstance(var_loc, slice) else slice(var_loc[0], var_loc[1])
            return _extract_sequence_chunk(
                fasta_path=fasta_path,
                chroms=chrom_col,
                starts=start_col,
                ends=end_col,
                strands=strand_col,
                var_slice=var_slice,
                seq_length=seq_length,
            )

        template = da.zeros(
            (ranges_df.shape[0], seq_length, 4),
            chunks=(batch_size, seq_length, 4),
            dtype=np.uint8,
        )
        data = da.map_blocks(_load_block, template, dtype=np.uint8)

        da_sequences = xr.DataArray(
            data,
            dims=dimnames,
        )
        adata[key] = da_sequences
        adata.attrs["genome_name"] = genome.name
        adata.attrs["genome_fasta"] = fasta_path
        adata.attrs["genome_chrom_sizes"] = json.dumps(genome.chrom_sizes) if len(genome.chrom_sizes.keys()) < 1000 else None
        try:
            from dask.diagnostics import ProgressBar
        except Exception:
            ProgressBar = None
        if ProgressBar is None:
            adata.to_zarr(adata.encoding["source"], mode="a")
        else:
            with ProgressBar():
                adata.to_zarr(adata.encoding["source"], mode="a")
        if hasattr(adata.__class__, "open_zarr"):
            return adata.__class__.open_zarr(adata.encoding["source"], consolidated=False)
        return xr.open_zarr(adata.encoding["source"], consolidated=False)

    # Create the one-hot encoded DataArray in memory (small datasets).
    da = create_one_hot_encoded_array(ranges_df, genome, seq_length=seq_length)
    da = da.rename(dict(zip(da.dims,dimnames)))
    adata[key] = da.chunk({dimnames[0]:adata.attrs['chunk_size'],dimnames[1]:seq_length,dimnames[2]:4})
    
    # Store key genome metadata in the dataset's attributes.
    adata.attrs["genome_name"] = genome.name
    adata.attrs["genome_fasta"] = str(genome._fasta) if hasattr(genome, "_fasta") else None
    adata.attrs["genome_chrom_sizes"] = json.dumps(genome.chrom_sizes) if len(genome.chrom_sizes.keys())<1000 else None
    
    return adata


def _extract_sequence_chunk(
    *,
    fasta_path: str,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    strands: np.ndarray,
    var_slice: slice,
    seq_length: int,
) -> np.ndarray:
    try:
        import pysam
    except ImportError as exc:
        raise ImportError("pysam is required to fetch sequences.") from exc

    sl_chroms = chroms[var_slice]
    sl_starts = starts[var_slice]
    sl_ends = ends[var_slice]
    sl_strands = strands[var_slice]
    out = np.empty((len(sl_chroms), seq_length, 4), dtype=np.uint8)

    fasta = pysam.FastaFile(fasta_path)
    try:
        for i, (chrom, start, end, strand) in enumerate(
            zip(sl_chroms, sl_starts, sl_ends, sl_strands)
        ):
            seq = fasta.fetch(reference=chrom, start=int(start), end=int(end))
            if strand == "-":
                seq = reverse_complement(seq)
            if len(seq) < seq_length:
                seq = seq.ljust(seq_length, "N")
            elif len(seq) > seq_length:
                seq = seq[:seq_length]
            out[i] = one_hot_encode_sequence(seq)
    finally:
        fasta.close()
    return out

def hot_encoding_to_sequence(one_hot_encoded_sequence: np.ndarray) -> str:
    """
    Decode a one hot encoded sequence to a DNA sequence string. Directly from CREsted

    Parameters
    ----------
    one_hot_encoded_sequence
        A numpy array with shape (x, 4) with dtype=np.float32.

    Returns
    -------
    The DNA sequence string of length x.
    """
    # Convert hot encoded seqeuence from:
    #   (x, 4) with dtype=np.float32
    # to:
    #   (x, 4) with dtype=np.uint8
    # and finally combine ACGT dimensions to:
    #   (x, 1) with dtype=np.uint32
    hes_u32 = one_hot_encoded_sequence.astype(np.uint8).view(np.uint32)

    # Do some bitshifting magic to decode uint32 to DNA sequence string.
    sequence = (
        HOT_DECODING_TABLE[
            (
                (
                    hes_u32 << 31 >> 31
                )  # A: 2^0  : 1        -> 1 = A in HOT_DECODING_TABLE
                | (
                    hes_u32 << 23 >> 30
                )  # C: 2^8  : 256      -> 2 = C in HOT_DECODING_TABLE
                | (
                    hes_u32 << 15 >> 29
                )  # G: 2^16 : 65536    -> 4 = G in HOT_DECODING_TABLE
                | (
                    hes_u32 << 7 >> 28
                )  # T: 2^24 : 16777216 -> 8 = T in HOT_DECODING_TABLE
            ).astype(np.uint8)
        ]
        .tobytes()
        .decode("ascii")
    )

    return sequence

def reverse_complement(sequence: str | list[str] | np.ndarray) -> str | np.ndarray:
    """
    Perform reverse complement on either a one-hot encoded array or a (list of) DNA sequence string(s).

    Parameters
    ----------
    sequence
        The DNA sequence string(s) or one-hot encoded array to reverse complement.

    Returns
    -------
    The reverse complemented DNA sequence string or one-hot encoded array.
    """

    def complement_str(seq: str) -> str:
        complement = str.maketrans("ACGTacgt", "TGCAtgca")
        return seq.translate(complement)[::-1]

    if isinstance(sequence, str):
        return complement_str(sequence)
    elif isinstance(sequence, list):
        return [complement_str(seq) for seq in sequence]
    elif isinstance(sequence, np.ndarray):
        if sequence.ndim == 2:
            if sequence.shape[1] == 4:
                return sequence[::-1, ::-1]
            elif sequence.shape[0] == 4:
                return sequence[:, ::-1][:, ::-1]
            else:
                raise ValueError(
                    "One-hot encoded array must have shape (W, 4) or (4, W)"
                )
        elif sequence.ndim == 3:
            if sequence.shape[1] == 4:
                return sequence[:, ::-1, ::-1]
            elif sequence.shape[2] == 4:
                return sequence[:, ::-1, ::-1]
            else:
                raise ValueError(
                    "One-hot encoded array must have shape (B, 4, W) or (B, W, 4)"
                )
        else:
            raise ValueError("One-hot encoded array must have 2 or 3 dimensions")
    else:
        raise TypeError(
            "Input must be either a DNA sequence string or a one-hot encoded array"
        )

class DNATransform:
    def __init__(
        self,
        out_len: int,
        random_rc: bool = False,
        max_shift: int = None,
        dimnames = ('var','seq_len','nuc',),
        apply_states: tuple[str, ...] = ("train", "val"),
        rc_states: tuple[str, ...] | None = None,
    ):
        """
        Initialize a DNATransform.

        Parameters
        ----------
        out_len : int
            The desired output window length. Must be <= seq_len.
        random_rc : bool, default False
            If True, each sample has a 50% chance to be reverse complemented.
        max_shift : int, optional
            The maximum number of bases to shift the window center away from the sequence midpoint.
            Defaults to 0 shift if not provided.
        apply_states : tuple[str, ...], default ("train", "val")
            Loader states in which windowing is applied when called as a transform.
        rc_states : tuple[str, ...] | None
            Loader states in which reverse complementing is applied. Defaults to apply_states.

        Notes
        -----
        This object is callable and can be used directly in ``GRAnDataModule`` transforms.
        The call signature is ``fn(array, dims, state=None)``; ``dims`` should include
        ``dimnames`` so the transform can locate the sequence and nucleotide axes.
        """
        self.out_len = out_len
        self.random_rc = random_rc
        self.max_shift_param = max_shift
        self.dimnames = dimnames
        self.apply_states = apply_states
        self.rc_states = rc_states if rc_states is not None else apply_states

    def get_window_indices(self, seq_len: int, shift: int = 0) -> (int, int):
        """
        Compute the start and end indices for window extraction.

        Parameters
        ----------
        seq_len : int
            Original sequence length.
        shift : int, optional
            Desired shift from the center (default is 0). If a nonzero value is provided,
            it will be clipped to the maximum allowed value.

        Returns
        -------
        start_index, end_index : (int, int)
            The start and end indices for slicing.
        """
        if self.out_len > seq_len:
            raise ValueError(f"out_len ({self.out_len}) cannot be larger than seq_len ({seq_len}).")
        allowed_max_shift = (seq_len - self.out_len) // 2
        effective_shift = np.clip(shift, -allowed_max_shift, allowed_max_shift)
        mid = seq_len // 2
        new_center = mid + effective_shift
        start_index = new_center - (self.out_len // 2)
        end_index = start_index + self.out_len
        return start_index, end_index

    def get_window_indices_and_rc(self, da: xr.DataArray, shift: int = None):
        """
        Given a one-hot encoded DataArray (with dims [self.dimnames[0], self.dimnames[1], self.dimnames[2]]),
        compute the indices for window extraction and return the indices along with
        a boolean array indicating which samples should be reverse complemented.

        Parameters
        ----------
        da : xarray.DataArray
            Input one-hot encoded sequences.
        shift : int, optional
            Desired shift value. Defaults to 0 if not provided.

        Returns
        -------
        start, end : int
            The start and end indices to slice the seq_len dimension.
        rc_flags : numpy.ndarray
            Boolean array of length equal to the number of samples (var dimension)
            indicating for each sample whether reverse complement should be applied.
        """
        if shift is None:
            shift = 0
        seq_len = da.sizes[self.dimnames[1]]
        start, end = self.get_window_indices(seq_len, shift)
        if self.random_rc:
            rc_flags = np.random.rand(da.sizes[self.dimnames[0]]) < 0.5
        else:
            rc_flags = np.zeros(da.sizes[self.dimnames[0]], dtype=bool)
        return start, end, rc_flags

    def __call__(self, arr: np.ndarray, dims: tuple[str, ...], state: str | None = None):
        if state is not None and self.apply_states and state not in self.apply_states:
            return arr
        var_name, seq_name, nuc_name = self.dimnames
        if var_name not in dims or seq_name not in dims or nuc_name not in dims:
            return arr
        var_axis = dims.index(var_name)
        seq_axis = dims.index(seq_name)
        nuc_axis = dims.index(nuc_name)

        shift = 0
        if self.max_shift_param:
            shift = np.random.randint(-self.max_shift_param, self.max_shift_param + 1)
        seq_len = arr.shape[seq_axis]
        start_idx, end_idx = self.get_window_indices(seq_len, shift=shift)
        arr = np.take(arr, np.arange(start_idx, end_idx), axis=seq_axis)

        if not self.random_rc or (state is not None and self.rc_states and state not in self.rc_states):
            return arr

        rc_flags = np.random.rand(arr.shape[var_axis]) < 0.5
        if not np.any(rc_flags):
            return arr
        arr = np.moveaxis(arr, var_axis, 0)
        rc_idx = np.flatnonzero(rc_flags)
        seq_axis_adj = seq_axis - (1 if seq_axis > var_axis else 0)
        nuc_axis_adj = nuc_axis - (1 if nuc_axis > var_axis else 0)
        arr[rc_idx] = np.flip(arr[rc_idx], axis=(seq_axis_adj, nuc_axis_adj))
        arr = np.moveaxis(arr, 0, var_axis)
        return arr

    def reverse_complement(self, da):
        """
        Reverse complement a one-hot encoded sequence or batch DataSet.
        Assumes that the DataArray has dims [self.dimnames[1], self.dimnames[2]].
        The reverse complement is implemented by reversing the order along
        the 'seq_len' dimension and swapping the nucleotide channels (e.g. ACGT → TGCA).
        """
        # Reverse the sequence order along 'seq_len' and the nucleotide order along 'nuc'
        return da.isel(seq_len=slice(None, None, -1)).isel(nuc=slice(None, None, -1))

    def apply_rc(self, window, rc_flags):
        """
        Apply reverse complementing to each sample in the window DataArray where indicated.

        Parameters
        ----------
        window : xarray.DataArray
            DataArray with dimensions [self.dimnames[0], self.dimnames[1], self.dimnames[2]] representing the extracted window.
        rc_flags : numpy.ndarray
            Boolean array of length equal to window.sizes[self.dimnames[0]] indicating which samples
            should be reverse complemented.

        Returns
        -------
        xarray.DataArray
            DataArray with reverse complement applied where appropriate.
        """
        if not self.random_rc:
            return window
        samples = []
        for i in range(window.sizes[self.dimnames[0]]):
            sample = window.isel(var=i)
            if rc_flags[i]:
                sample = self.reverse_complement(sample)
            samples.append(sample)
        return xr.concat(samples, dim=self.dimnames[0])
