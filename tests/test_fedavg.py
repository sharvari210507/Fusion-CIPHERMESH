import numpy as np
from src.fedavg import fedavg


def test_weighted_average_known_values():
    u = [{"coef": np.array([[1.0, 2.0]]), "intercept": np.array([0.0]), "n": 1, "meta": {}},
         {"coef": np.array([[3.0, 4.0]]), "intercept": np.array([2.0]), "n": 3, "meta": {}}]
    g, acc, rej, w = fedavg(u, (1, 2), (1,))
    assert acc == [0, 1] and rej == []
    np.testing.assert_allclose(g["coef"], [[2.5, 3.5]])
    np.testing.assert_allclose(g["intercept"], [1.5])
    assert abs(w[0]["weight"] - 0.25) < 1e-9 and abs(w[1]["weight"] - 0.75) < 1e-9


def test_incompatible_shapes_rejected():
    u = [{"coef": np.array([[1.0, 2.0]]), "intercept": np.array([0.0]), "n": 5, "meta": {}},
         {"coef": np.array([[1.0, 2.0, 3.0]]), "intercept": np.array([0.0]), "n": 5, "meta": {}}]
    g, acc, rej, _ = fedavg(u, (1, 2), (1,))
    assert acc == [0] and len(rej) == 1


def test_nan_inf_rejected():
    bad1 = {"coef": np.array([[np.nan, 1.0]]), "intercept": np.array([0.0]), "n": 5, "meta": {}}
    bad2 = {"coef": np.array([[1.0, np.inf]]), "intercept": np.array([0.0]), "n": 5, "meta": {}}
    good = {"coef": np.array([[1.0, 1.0]]), "intercept": np.array([0.0]), "n": 5, "meta": {}}
    g, acc, rej, _ = fedavg([bad1, bad2, good], (1, 2), (1,))
    assert acc == [2] and len(rej) == 2
