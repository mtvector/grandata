import numpy as np
import pandas as pd
import xarray as xr
import h5py
from pathlib import Path
import pytest
from grandata.grandata import GRAnData

# -----------------------------------------------------------------------------
# Test 1: GRAnData fields and dynamic axis properties
# -----------------------------------------------------------------------------
def test_grandata_fields_and_properties(tmp_path: Path):
    # Create dummy data for all fields.
    X = xr.DataArray(np.arange(20).reshape(4, 5), dims=["obs", "var"])
    obs = pd.DataFrame({"a": list("ABCD")}, index=["obs1", "obs2", "obs3", "obs4"])
    var = pd.DataFrame({"b": list("VWXYZ")[:5]}, index=["var1", "var2", "var3", "var4", "var5"])
    obsm = {"embedding": xr.DataArray(np.random.rand(4, 2), dims=["obs", "other"])}
    varm = {"feature": xr.DataArray(np.random.rand(5, 3), dims=["var", "other"])}
    layers = {"layer1": X.copy()}
    varp = {"contacts": xr.DataArray(np.random.rand(5, 5), dims=["var_0", "var_1"])}
    obsp = {"adj": xr.DataArray(np.random.rand(4, 4), dims=["obs_0", "obs_1"])}
    
    # Create GRAnData with axis_indices for the primary axes.
    data = GRAnData(
        X, obs=obs, var=var, uns={"extra": "test"},
        obsm=obsm, varm=varm, layers=layers, varp=varp, obsp=obsp,
        axis_indices={"obs": list(obs.index), "var": list(var.index)}
    )
    
    # Check that the primary array and other properties are correct.
    assert np.array_equal(data.primary.data, X.data)
    pd.testing.assert_frame_equal(data.obs, obs)
    pd.testing.assert_frame_equal(data.var, var)
    assert "embedding" in data.obsm
    assert "feature" in data.varm
    assert "layer1" in data.layers
    assert "contacts" in data.varp
    assert "adj" in data.obsp
    
    # Check shape and dynamic axis properties.
    assert data.shape == X.shape
    # Dynamic axis coordinate properties:
    assert list(data.obs_names) == list(obs.index)
    assert list(data.var_names) == list(var.index)
    # Dynamic axis size properties:
    assert data.n_obs == X.sizes["obs"]
    assert data.n_var == X.sizes["var"]

# -----------------------------------------------------------------------------
# Test 2: HDF5 save and load with lazy loading
# -----------------------------------------------------------------------------
def test_hdf5_save_load(tmp_path: Path):
    X = xr.DataArray(np.arange(12).reshape(3, 4), dims=["obs", "var"])
    obs = pd.DataFrame({"col": [1, 2, 3]}, index=["o1", "o2", "o3"])
    var = pd.DataFrame({"col": [10, 20, 30, 40]}, index=["v1", "v2", "v3", "v4"])
    data = GRAnData(X, obs=obs, var=var, axis_indices={"obs": list(obs.index), "var": list(var.index)})
    h5_path = tmp_path / "test_adata.h5"
    data.to_h5(str(h5_path))
    loaded = GRAnData.from_h5(str(h5_path), backed=["X"])
    pd.testing.assert_frame_equal(loaded.obs, obs)
    pd.testing.assert_frame_equal(loaded.var, var)
    assert loaded.shape == data.shape
    # Check that the primary array is lazy.
    assert hasattr(loaded._data["X"], "attrs") and "_lazy_obj" in loaded._data["X"].attrs

# -----------------------------------------------------------------------------
# Test 3: Lazy conversion to in-memory arrays (to_memory)
# -----------------------------------------------------------------------------
def test_to_memory(tmp_path: Path):
    X = xr.DataArray(np.arange(24).reshape(4,6), dims=["obs", "var"])
    data = GRAnData(X, axis_indices={"obs": [f"o{i}" for i in range(4)],
                                       "var": [f"v{i}" for i in range(6)]})
    h5_path = tmp_path / "lazy.h5"
    data.to_h5(str(h5_path))
    lazy_data = GRAnData.from_h5(str(h5_path), backed=["X"])
    # Ensure primary array is lazy.
    assert hasattr(lazy_data._data["X"], "attrs") and "_lazy_obj" in lazy_data._data["X"].attrs
    # Convert all lazy arrays to in-memory.
    lazy_data.to_memory()
    # Verify that lazy attribute is removed and data is preserved.
    assert not (hasattr(lazy_data._data["X"], "attrs") and "_lazy_obj" in lazy_data._data["X"].attrs)
    np.testing.assert_array_equal(lazy_data.primary.data, X.data)

# -----------------------------------------------------------------------------
# Test 4: Prevent further slicing on a backed (lazy) slice
# -----------------------------------------------------------------------------
def test_prevent_further_slicing(tmp_path: Path):
    X = xr.DataArray(np.arange(24).reshape(4,6), dims=["obs", "var"])
    data = GRAnData(X, axis_indices={"obs": [f"o{i}" for i in range(4)],
                                       "var": [f"v{i}" for i in range(6)]})
    h5_path = tmp_path / "sliced.h5"
    data.to_h5(str(h5_path))
    lazy_data = GRAnData.from_h5(str(h5_path), backed=["X"])
    sliced = lazy_data[0:2]
    with pytest.raises(ValueError):
        _ = sliced[0:1]

# -----------------------------------------------------------------------------
# Test 5: Adding new properties after initialization
# -----------------------------------------------------------------------------
def test_add_property():
    X = xr.DataArray(np.arange(20).reshape(4,5), dims=["obs", "var"])
    data = GRAnData(X, axis_indices={"obs": [f"o{i}" for i in range(4)],
                                       "var": [f"v{i}" for i in range(5)]})
    new_prop = np.arange(8).reshape(4,2)
    data.add_property("new_feature", new_prop, dims=["obs", "feat"],
                      axis_indices={"obs": [f"o{i}" for i in range(4)],
                                    "feat": ["f1", "f2"]})
    # Check that new_feature exists and has the correct dims and coordinates.
    assert "new_feature" in data._data
    new_arr = data.new_feature
    assert list(new_arr.dims) == ["obs", "feat"]
    np.testing.assert_array_equal(new_arr.coords["obs"].values, np.array([f"o{i}" for i in range(4)]))
    np.testing.assert_array_equal(new_arr.coords["feat"].values, np.array(["f1", "f2"]))

# -----------------------------------------------------------------------------
# Test 6: Dynamic axis properties for any axis
# -----------------------------------------------------------------------------
def test_dynamic_axis_properties():
    # Create a primary array with three axes.
    X = xr.DataArray(np.arange(60).reshape(3,4,5), dims=["time", "obs", "var"])
    axis_indices = {"time": [f"t{i}" for i in range(3)],
                    "obs": [f"o{i}" for i in range(4)],
                    "var": [f"v{i}" for i in range(5)]}
    data = GRAnData(X, axis_indices=axis_indices)
    # Check dynamic axis coordinate properties.
    assert list(data.time_names) == [f"t{i}" for i in range(3)]
    assert list(data.obs_names) == [f"o{i}" for i in range(4)]
    assert list(data.var_names) == [f"v{i}" for i in range(5)]
    # Check dynamic axis size properties.
    assert data.n_time == 3
    assert data.n_obs == 4
    assert data.n_var == 5
