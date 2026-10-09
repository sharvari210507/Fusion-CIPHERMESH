"""Regression tests: cache schema mismatch and feature-column consistency.

Uses isolated tmp dirs and the offline simulated generator only.
"""
import numpy as np


def _write_stale_cache(path):
    """Recreate the historical stale layout: 9 columns with a duplicated
    `type_encoded` column and no schema version (as in the old fixtures)."""
    import os
    os.makedirs(path, exist_ok=True)
    stale = ["step", "type_encoded", "amount", "oldbalanceOrg", "newbalanceOrig",
             "oldbalanceDest", "newbalanceDest", "isFlaggedFraud", "type_encoded"]
    rng = np.random.RandomState(0)
    for b in range(5):
        np.savez(os.path.join(path, f"bank_{b}_data.npz"),
                 X=rng.normal(size=(20, 9)), y=np.zeros(20, dtype=int),
                 feature_names=stale)
        import joblib
        from sklearn.preprocessing import StandardScaler
        joblib.dump(StandardScaler(), os.path.join(path, f"bank_{b}_scaler.save"))
    np.savez(os.path.join(path, "metadata.npz"), n_banks=5, feature_names=stale)


def test_stale_cache_rebuilt_not_reused(tmp_path):
    from src.dataset import load_and_preprocess_data, _cache_valid, get_bank_data
    from src.preprocessing import feature_names, validate_feature_matrix
    d = str(tmp_path / "cache")
    _write_stale_cache(d)
    assert _cache_valid(d) is False
    data = load_and_preprocess_data(d)  # must rebuild, never reuse stale schema
    assert _cache_valid(d) is True
    assert data[0]["feature_names"] == feature_names()
    X, _ = get_bank_data(0, data)
    assert X.shape[1] == 12
    validate_feature_matrix(X, "stale_rebuild")


def test_feature_columns_consistent_and_unique(tmp_path):
    from src.dataset import load_and_preprocess_data
    from src.preprocessing import feature_names
    data = load_and_preprocess_data(str(tmp_path / "fresh"), force_download=True)
    expected = feature_names()
    assert len(expected) == len(set(expected)) == 12
    for b in range(5):
        cols = data[b]["feature_names"]
        assert cols == expected, f"bank {b}: {cols}"
        assert data[b]["X"].shape[1] == 12


def test_valid_cache_reused(tmp_path, capsys):
    from src.dataset import load_and_preprocess_data
    d = str(tmp_path / "reuse")
    load_and_preprocess_data(d, force_download=True)
    load_and_preprocess_data(d)  # second load must hit the cache
    out = capsys.readouterr().out
    assert "Loading preprocessed data from cache" in out
