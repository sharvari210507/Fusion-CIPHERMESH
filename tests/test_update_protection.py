import numpy as np
from src.update_protection import clip_update, add_noise


def _u(c):
    return {"coef": np.array([[c, 0.0]]), "intercept": np.array([0.0]), "n": 10, "meta": {}}


def _g():
    return {"coef": np.array([[0.0, 0.0]]), "intercept": np.array([0.0])}


def test_clipping_respects_threshold():
    out, info = clip_update(_u(10.0), _g(), 1.0)
    assert info["clipped"] is True
    assert info["norm_after"] <= 1.0 + 1e-9
    out2, info2 = clip_update(_u(0.3), _g(), 1.0)
    assert info2["clipped"] is False


def test_noise_changes_update_and_records_settings():
    rng = np.random.default_rng(0)
    out = add_noise(_u(1.0), 1.0, 0.5, rng)
    assert out["meta"]["noise_sigma"] == 0.5
    assert not np.allclose(out["coef"], [[1.0, 0.0]])
    out0 = add_noise(_u(1.0), 1.0, 0.0, rng)
    np.testing.assert_allclose(out0["coef"], [[1.0, 0.0]])
