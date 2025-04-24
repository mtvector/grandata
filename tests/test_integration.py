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
# Fixture for temporary test setup
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_setup(tmp_path: Path):
    """
    Set up a temporary directory structure with necessary files:
      - Base directory with subdirectories for beds and bigwigs.
      - A chromsizes file.
      - A consensus BED file.
      - A simple BigWig file (created with pybigtools).
    """
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    beds_dir = base_dir / "beds"
    bigwigs_dir = base_dir / "bigwigs"
    beds_dir.mkdir()
    bigwigs_dir.mkdir()

    # Create a chromsizes file.
    chromsizes_file = base_dir / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")

    # Create two example BED files.
    bed_data_A = pd.DataFrame({0: ["chr1", "chr1"],
                               1: [100, 300],
                               2: [200, 400]})
    bed_data_B = pd.DataFrame({0: ["chr1", "chr1"],
                               1: [150, 350],
                               2: [250, 450]})
    (beds_dir / "ClassA.bed").write_text(bed_data_A.to_csv(sep="\t", header=False, index=False))
    (beds_dir / "ClassB.bed").write_text(bed_data_B.to_csv(sep="\t", header=False, index=False))

    # Create a consensus BED file.
    consensus = pd.DataFrame({0: ["chr1", "chr1", "chr1"],
                              1: [100, 300, 350],
                              2: [200, 400, 450]})
    consensus_file = base_dir / "consensus.bed"
    consensus_file.write_text(consensus.to_csv(sep="\t", header=False, index=False))

    # Create a simple BigWig file using pybigtools.
    bigwig_file = bigwigs_dir / "test.bw"
    bw = pybigtools.open(str(bigwig_file), mode="w")
    bw.write(chroms={"chr1": 1000}, vals=[("chr1", 0, 1000, 5.0)])
    bw.close()

    # Path for HDF5 backing file.
    backed_path = base_dir / "chrom_data.h5"

    return {
        "base_dir": base_dir,
        "beds_dir": beds_dir,
        "bigwigs_dir": bigwigs_dir,
        "chromsizes_file": chromsizes_file,
        "consensus_file": consensus_file,
        "bigwig_file": bigwig_file,
        "backed_path": backed_path,
    }

# -----------------------------------------------------------------------------
# Test 1: GRAnData fields and properties
# -----------------------------------------------------------------------------

def test_yanndata_fields(tmp_path: Path):
    # Create dummy data.
    X = xr.DataArray(np.arange(20).reshape(4, 5), dims=["obs", "var"])
    obs = pd.DataFrame({"a": list("ABCD")}, index=["obs1", "obs2", "obs3", "obs4"])
    var = pd.DataFrame({"b": list("VWXYZ")[:5]}, index=["var1", "var2", "var3", "var4", "var5"])
    obsm = {"embedding": xr.DataArray(np.random.rand(4, 2), dims=["obs", "dim"])}
    varm = {"feature": xr.DataArray(np.random.rand(5, 3), dims=["var", "dim"])}
    layers = {"layer1": X.copy()}
    varp = {"contacts": xr.DataArray(np.random.rand(5, 5), dims=["var_0", "var_1"])}
    obsp = {"adj": xr.DataArray(np.random.rand(4, 4), dims=["obs_0", "obs_1"])}
    
    ydata = GRAnData(X, obs=obs, var=var, uns={"extra": "test"},
                     obsm=obsm, varm=varm, layers=layers, varp=varp, obsp=obsp)
    # Force axis_indices so that obs_names and var_names return proper values.
    ydata.axis_indices["obs"] = np.array(ydata.obs.index)
    ydata.axis_indices["var"] = np.array(ydata.var.index)
    
    np.testing.assert_array_equal(ydata.X.values, X.values)
    pd.testing.assert_frame_equal(ydata.obs, obs)
    pd.testing.assert_frame_equal(ydata.var, var)
    assert "embedding" in ydata.obsm
    assert "feature" in ydata.varm
    assert "layer1" in ydata.layers
    assert "contacts" in ydata.varp
    assert "adj" in ydata.obsp
    assert ydata.shape == X.shape
    assert list(ydata.obs_names) == list(obs.index)
    assert list(ydata.var_names) == list(var.index)

# -----------------------------------------------------------------------------
# Test 2: obs loading after HDF5 save/load
# -----------------------------------------------------------------------------

def test_obs_loaded_correctly(tmp_path: Path):
    X = xr.DataArray(np.arange(12).reshape(3, 4), dims=["obs", "var"])
    obs = pd.DataFrame({"col": [1, 2, 3]}, index=["o1", "o2", "o3"])
    var = pd.DataFrame({"col": [10, 20, 30, 40]}, index=["v1", "v2", "v3", "v4"])
    ydata = GRAnData(X, obs=obs, var=var)
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
    ydata = GRAnData(X, obs=obs, var=var)
    ydata.axis_indices["obs"] = np.array(obs.index)
    ydata.axis_indices["var"] = np.array(var.index)
    
    # Create a dummy dataset that mimics AnnDataset's __getitem__.
    class DummyDataset:
        def __init__(self, ydata):
            self.ydata = ydata
            self.augmented_probs = np.ones(6)
            self.index_manager = type("IM", (), {"augmented_indices": list(range(6))})
            self.precomputed_sequences = None
        def __getitem__(self, idx):
            seq = self.ydata.X.isel(obs=idx).values
            return {"sequence": seq, "y": seq}
        def __len__(self):
            return 6
    dataset = DummyDataset(ydata)
  
