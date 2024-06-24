import pytest
from train import load_data

def test_load_data_shapes():
    X_train, y_train, X_test, y_test = load_data()
    assert X_train.shape == (60000, 784)
    assert y_train.shape == (60000,)
    assert X_test.shape == (10000, 784)
    assert y_test.shape == (10000,)

def test_load_data_range():
    X_train, y_train, X_test, y_test = load_data()
    assert X_train.min() >= 0 and X_train.max() <= 1
    assert X_test.min() >= 0 and X_test.max() <= 1
