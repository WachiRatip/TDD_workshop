import pytest
import numpy as np

from tdd.linear_ssvm import logHingeLoss

def test_nan_found():
    X = np.array([[-1, 0], [0, 1], [1, 1]])
    Y = np.array([0, 1, 1])
    clf = logHingeLoss(mu=1e-10)
    with pytest.raises(ValueError):
        clf.fit(X, Y)