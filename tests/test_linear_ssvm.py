import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from tdd.linear_ssvm import logHingeLoss

def test_compute_gradient():
    x = np.array([[-1, 0], [0, 1], [1, 1]])
    t = np.array([0, 1, 1])
    w = np.zeros(len(x[0])+1)
    X = np.ones((len(x),len(x[0])+1))
    X[:,:-1] = x.copy()
    t[t==0] = -1
    u, p, grad = logHingeLoss(mu=1e-2).gradient(w, X, t)
    assert u.shape == w.shape
    assert p.shape == w.shape
    assert grad.shape == w.shape
    assert_array_almost_equal(u, np.array([1,1,1]))
    assert_array_almost_equal(p, np.array([1,1,1]))
    assert_array_almost_equal(grad, np.array([-2/3,-2/3,-1/3]))
    
    u, p, grad = logHingeLoss(mu=1e-2).gradient(np.ones(len(x[0])+1), X, t)
    assert_array_almost_equal(u, np.array([1,-1,-2]))
    assert_array_almost_equal(p, np.array([1,0,0]))
    assert_array_almost_equal(grad, np.array([2/3,1,4/3]))

def test_simple_biclass_ssvm():
    X = np.array([[-1, 0], [0, 1], [1, 1]])
    Y = np.array([0, 1, 1])
    # Simple sanity check on a 2 classes dataset
    # Make sure it predicts the correct result on simple datasets.
    n_samples = len(Y)
    classes = np.unique(Y)
    n_classes = classes.shape[0]
    assert n_classes == 2
    clf = logHingeLoss()
    clf.fit(X, Y)
    predicted = clf.predict(X)
    assert predicted.shape == (n_samples,)
    assert set({0,1}) == set(clf.classes_.tolist())
    assert_array_equal(predicted, Y)

def test_biclass_ssvm():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    Y = np.array([1, 1, 1, 0, 0, 0])
    clf = logHingeLoss(1e-2)
    clf.fit(X, Y)
    assert_array_equal(clf.predict(X), Y)
    assert_array_almost_equal(clf._intercept, np.array(0.0))   
    T = np.array([[-1, -1], [2, 2], [3, 2]])
    true_result = np.array([1, -1, -1])
    assert_array_equal(clf.predict(T), true_result)