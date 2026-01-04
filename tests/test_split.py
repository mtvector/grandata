import numpy as np
import xarray as xr

from grandata import GRAnData, train_val_test_split


def test_train_val_test_split_chr_auto():
    var_names = np.array(
        [
            "chr1:0-10",
            "chr1:10-20",
            "chr2:0-10",
            "chr2:10-20",
            "chr3:0-10",
            "chr3:10-20",
        ]
    )
    data = xr.DataArray(np.zeros((var_names.size,)), dims=("var",), coords={"var": var_names})
    adata = GRAnData(X=data)

    train_val_test_split(adata, strategy="chr_auto", val_size=0.2, test_size=0.2, random_state=0)

    split = adata["var-_-split"].values.tolist()
    assert set(split).issubset({"train", "val", "test"})
    assert "val" in split
    assert "test" in split
