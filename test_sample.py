import pytest
from sample import add, sub, mul

def test_add():
    assert add(3, 4) == 7

def test_sub():
    assert sub(5, 3) == 2

def test_mul():
    assert mul(4, 5) == 20
