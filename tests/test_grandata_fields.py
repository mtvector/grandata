import os
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import pybigtools  # Use pybigtools instead of pyBigWig

from grandata.grandata import GRAnData
from grandata.chrom_io import import_bigwigs, add_contact_strengths_to_varp
from grandata._genome import Genome
from grandata._anndatamodule import MetaAnnDataModule
from grandata._dataloader import AnnDataLoader
from grandata.utils import hot_encoding_to_sequence
from grandata.chrom_io import import_bigwigs

# -----------------------------------------------------------------------------
# Test 1: GRAnData fields and properties
# -----------------------------------------------------------------------------

def test_yanndata_fields(tmp_path: Path):
    # Create dummy data for all fields.
    X = xr.DataArray(np.arange(20).reshape(4, 5), dims=["obs", "var"])
    obs = pd.DataFrame({"a": list("ABCD")}, index=["obs1", "obs2", "obs3", "obs4"])
    var = pd.DataFrame({"b": list("VWXYZ")[:5]}, index=["var1", "var2", "var3", "var4", "var5"])
    obsm = {"embedding": xr.DataArray(np.random.rand(4, 2), dims=["obs", "dim"])}
    varm = {"feature": xr.DataArray(np.random.rand(5, 3), dims=["var", "dim"])}
    layers = {"layer1": X.copy()}
    varp = {"contacts": xr.DataArray(np.random.rand(5, 5), dims=["var_0", "var_1"])}
    obsp = {"adj": xr.DataArray(np.random.rand(4, 4), dims=["obs_0", "obs_1"])}

    # Pass explicit axis_indices so that obs_names and var_names come from the DataFrame indices.
    ydata = GRAnData(X, obs=obs, var=var, uns={"extra": "test"},
                     obsm=obsm, varm=varm, layers=layers, varp=varp, obsp=obsp,
                     axis_indices={"obs": obs.index.to_numpy(), "var": var.index.to_numpy()})
    
    np.testing.assert_array_equal(ydata.X.values, X.values)
    pd.testing.assert_frame_equal(ydata.obs, obs)
    pd.testing.assert_frame_equal(ydata.var, var)
    assert "embedding" in ydata.obsm
    assert "feature" in ydata.varm
    assert "layer1" in ydata.layers
    assert "contacts" in ydata.varp
    assert "adj" in ydata.obsp
    assert ydata.shape == X.shape
    # Now, with axis_indices provided, obs_names and var_names should match the indices.
    assert list(ydata.obs_names) == list(obs.index)
    assert list(ydata.var_names) == list(var.index)

# -----------------------------------------------------------------------------
# Test 2: obs loading after HDF5 save/load
# -----------------------------------------------------------------------------

def test_obs_loaded_correctly(tmp_path: Path):
    X = xr.DataArray(np.arange(12).reshape(3, 4), dims=["obs", "var"])
    obs = pd.DataFrame({"col": [1, 2, 3]}, index=["o1", "o2", "o3"])
    var = pd.DataFrame({"col": [10, 20, 30, 40]}, index=["v1", "v2", "v3", "v4"])
    ydata = GRAnData(X, obs=obs, var=var, axis_indices={"obs": obs.index.to_numpy(), "var": var.index.to_numpy()})
    h5_path = tmp_path / "test_adata.h5"
    ydata.to_h5(str(h5_path))
    ydata_loaded = GRAnData.from_h5(str(h5_path))
    pd.testing.assert_frame_equal(ydata_loaded.obs, obs)
    pd.testing.assert_frame_equal(ydata_loaded.var, var)

# -----------------------------------------------------------------------------
# Test 3: Batches composition and size from AnnDataLoader
# -----------------------------------------------------------------------------

def test_batches_composition():
    X = xr.DataArray(np.random.rand(6, 10), dims=["obs", "var"])
    obs = pd.DataFrame({"col": np.arange(6)}, index=[f"obs{i}" for i in range(6)])
    var = pd.DataFrame({"col": np.arange(10)}, index=[f"var{j}" for j in range(10)])
    ydata = GRAnData(X, obs=obs, var=var, axis_indices={"obs": obs.index.to_numpy(), "var": var.index.to_numpy()})
    
    # Create a dummy dataset that mimics __getitem__ of AnnDataset.
    class DummyDataset:
        def __init__(self, ydata):
            # Mimic a real dataset by storing the GRAnData under 'adata'
            self.adata = ydata
            self.augmented_probs = np.ones(6)
            # Create an index_manager with augmented_indices equal to range(6)
            self.index_manager = type("IM", (), {"augmented_indices": list(range(6))})
            self.precomputed_sequences = None
        def __getitem__(self, idx):
            # Return a dict with "sequence" and "y" keys. For simplicity, use ydata.X.
            # Here, we index along the 'obs' dimension.
            seq = self.adata.X.isel(obs=idx).values  # shape (var,)
            return {"sequence": seq, "y": seq}
        def __len__(self):
            return 6

    dataset = DummyDataset(ydata)
    loader = AnnDataLoader(dataset, batch_size=2, shuffle=False,
                           drop_remainder=False, epoch_size=6, stage="test",
                           shuffle_obs=False)
    batch = next(iter(loader.data))
    for key in ['sequence', 'y']:
        assert key in batch
        # The batch shape should be (obs, batch, ...) where batch equals 2.
        # In this test, since each sample returns an array of shape (var,)
        # we expect the final stacked shape to be (var, 2)
        # (because in our dummy samples, dim0 is "var" since there is no explicit obs dim).
        assert batch[key].shape[1] == 2

# -----------------------------------------------------------------------------
# Test 4: Backed files and lazy loading via GRAnData.from_h5
# -----------------------------------------------------------------------------

def test_lazy_loading(tmp_path: Path):
    X = xr.DataArray(np.random.rand(50, 20), dims=["obs", "var"])
    ydata = GRAnData(X)
    h5_path = tmp_path / "lazy.h5"
    ydata.to_h5(str(h5_path))
    ydata_loaded = GRAnData.from_h5(str(h5_path), backed=["X"])
    # Check that the loaded X contains a _lazy_obj attribute.
    lazy_obj = ydata_loaded.X.attrs.get("_lazy_obj")
    assert lazy_obj is not None, "Lazy object not found in attributes"
    assert lazy_obj.__class__.__name__ == "LazyH5Array"

# -----------------------------------------------------------------------------
# Test 5: DNA sequence retrieval and shifting correctness
# -----------------------------------------------------------------------------

def test_dna_sequence_retrieval_and_shift(tmp_path: Path):
    fasta_file = tmp_path / "chr1.fa"
    seq = ("ACGT" * 250)[:1000]
    fasta_file.write_text(">chr1\n" + seq + "\n")
    
    chromsizes_file = tmp_path / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")
    
    dummy_genome = Genome(str(fasta_file), chrom_sizes=str(chromsizes_file))
    
    from grandata._dataset import SequenceLoader
    regions = ["chr1:100-110:+"]
    loader = SequenceLoader(dummy_genome, in_memory=True, always_reverse_complement=False,
                              deterministic_shift=False, max_stochastic_shift=5, regions=regions)
    retrieved_seq = loader.get_sequence("chr1:100-110:+", shift=0)
    expected_seq = seq[100:110]
    assert retrieved_seq == expected_seq

    retrieved_seq_shift = loader.get_sequence("chr1:100-110:+", shift=2)
    expected_seq_shift = seq[102:112]
    assert retrieved_seq_shift == expected_seq_shift

# -----------------------------------------------------------------------------
# Test 6: MetaAnnDataModule loads the correct genomic DNA per row
# -----------------------------------------------------------------------------

def test_meta_module_genomic_dna(tmp_path: Path):
    fasta_file = tmp_path / "chr1.fa"
    seq = ("ACGT" * 250)[:1000]
    fasta_file.write_text(">chr1\n" + seq + "\n")
    
    chromsizes_file = tmp_path / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")
    genome = Genome(str(fasta_file), chrom_sizes=str(chromsizes_file))
    
    consensus = pd.DataFrame({
        0: ["chr1"] * 3,
        1: [100, 200, 300],
        2: [110, 210, 310]
    })
    consensus_file = tmp_path / "consensus.bed"
    consensus_file.write_text(consensus.to_csv(sep="\t", header=False, index=False))
    
    bigwigs_dir = tmp_path / "bigwigs"
    bigwigs_dir.mkdir()
    bigwig_file = bigwigs_dir / "test.bw"
    bw = pybigtools.open(str(bigwig_file), mode="w")
    bw.write(chroms={"chr1": 1000}, vals=[("chr1", 0, 1000, 5.0)])
    bw.close()
    
    backed_path = tmp_path / "chrom_data.h5"
    adata = import_bigwigs(
        bigwigs_folder=bigwigs_dir,
        regions_file=consensus_file,
        backed_path=str(backed_path),
        target_region_width=10,
        chromsizes_file=str(chromsizes_file),
    )
    # Ensure obs and obsm exist.
    if adata.var is None:
        adata.var = pd.DataFrame()
    adata.var["split"] = "train"
    if adata.obsm is None:
        adata.obsm = {}
    adata.obsm["dummy"] = xr.DataArray(np.random.rand(adata.obs.shape[0], 5),
                                        dims=["types", "genes"])
    
    adata1 = adata.copy()
    adata2 = adata.copy()
    meta_module = MetaAnnDataModule(
        adatas=[adata1, adata2],
        genomes=[genome, genome],
        data_sources={'y': 'X'},
        in_memory=True,
        random_reverse_complement=False,
        max_stochastic_shift=0,
        deterministic_shift=False,
        shuffle_obs=False,
        shuffle=False,
        batch_size=1,
        epoch_size=2
    )
    meta_module.setup("fit")
    loader = meta_module.train_dataloader
    batch = next(iter(loader.data))
    sample_seq = batch["sequence"][:, 0]
    sample_seq_np = sample_seq.cpu().numpy() if hasattr(sample_seq, "cpu") else np.asarray(sample_seq)
    decoded_seq = hot_encoding_to_sequence(sample_seq_np)
    expected_seq = seq[100:110]
    assert decoded_seq == expected_seq

# -----------------------------------------------------------------------------
# Test 7: Verify that obs dimensions are shuffled correctly in the batch
# -----------------------------------------------------------------------------

def test_obs_shuffling(monkeypatch):
    import torch
    dummy_sample = {
        "sequence": np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]),  # shape (3, 3) where dim0 is obs, dim1 is var
        "y": np.array([[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]])
    }
    fixed_perm = torch.tensor([2, 0, 1])
    monkeypatch.setattr(torch, "randperm", lambda n: fixed_perm)
    
    # Modify the dummy dataset to have an "adata" attribute.
    class DummyDataset:
        def __init__(self, sample):
            # Build a minimal fake adata with an obs DataFrame.
            self.adata = type("FakeAdata", (), {})()
            self.adata.obs = pd.DataFrame(index=["obs0", "obs1", "obs2"])
            # For consistency, set meta_obs_names to be the same as obs.index.
            self.adata.meta_obs_names = np.array(self.adata.obs.index)
            self.sample = sample
        def __getitem__(self, idx):
            return self.sample
        def __len__(self):
            return 1

    dataset = DummyDataset(dummy_sample)
    loader = AnnDataLoader(dataset, batch_size=1, shuffle_obs=True)
    batch = loader.batch_collate_fn([dummy_sample, dummy_sample])
    expected_sequence = np.stack([dummy_sample["sequence"][fixed_perm.numpy()]] * 2, axis=1)
    expected_y = np.stack([dummy_sample["y"][fixed_perm.numpy()]] * 2, axis=1)
    np.testing.assert_array_equal(batch["sequence"].cpu().numpy(), expected_sequence)
    np.testing.assert_array_equal(batch["y"].cpu().numpy(), expected_y)
