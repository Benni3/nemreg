# tests/test_dataset.py
import numpy as np
import pytest
import nemreg as nr

def test_dataset_1d_shapes():
    x = np.linspace(0, 1, 5)
    y = 2*x + 1
    ds = nr.Dataset(x=x, y=y, name="ds")

    # your Dataset stores x as (n,d)
    assert ds.x.ndim == 2
    assert ds.x.shape == (5, 1)
    assert ds.y.shape == (5,)

def test_dataset_multivariable_shapes():
    n = 7
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(1, 2, n)
    X = np.column_stack([x1, x2])  # (n,2)
    y = 3*x1 - 2*x2 + 0.5

    ds = nr.Dataset(x=X, y=y, name="mv")
    assert ds.x.shape == (n, 2)
    assert ds.y.shape == (n,)
    assert ds.n_features == 2

def test_dataset_scalar_yerr_expands():
    x = np.linspace(0, 1, 4)
    y = x
    ds = nr.Dataset(x=x, y=y, yerr=0.1)
    assert ds.yerr.shape == (4,)
    assert np.allclose(ds.yerr, 0.1)

def test_dataset_scalar_xerr_expands_for_multivariable():
    n = 6
    X = np.column_stack([np.arange(n), np.arange(n)+1.0])
    y = np.arange(n)
    ds = nr.Dataset(x=X, y=y, xerr=0.2)
    assert ds.xerr.shape == (n, 2)
    assert np.allclose(ds.xerr, 0.2)

def test_dataset_raises_on_mismatch():
    with pytest.raises(ValueError):
        nr.Dataset(x=[1,2,3], y=[1,2])