import pytest

from tdd.linear_ssvm import logHingeLoss

@pytest.mark.parametrize(
    'invalid_mu', [-1, 0, -0.5],
)
def test_invalid_mu(invalid_mu):
    with pytest.raises(AssertionError):
        logHingeLoss(mu=invalid_mu)

@pytest.mark.parametrize(
    'invalid_c', [-1, 0, -0.5],
)
def test_invalid_c(invalid_c):
    with pytest.raises(AssertionError):
        logHingeLoss(C=invalid_c)

@pytest.mark.parametrize(
    'invalid_tol', [-1, 0, -0.5],
)
def test_invalid_tol(invalid_tol):
    with pytest.raises(AssertionError):
        logHingeLoss(tol=invalid_tol)

@pytest.mark.parametrize(
    'invalid_max_iter', [-1, 0, -0.5, 1.5, 10.0],
)
def test_invalid_max_iter(invalid_max_iter):
    with pytest.raises(AssertionError):
        logHingeLoss(max_iter=invalid_max_iter)