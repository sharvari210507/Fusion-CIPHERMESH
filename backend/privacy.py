import numpy as np


def clip_update(delta, clip_norm):
    delta = np.asarray(delta, dtype=np.float64)
    n = float(np.linalg.norm(delta))
    if clip_norm is None:
        return delta, n, n
    if n > clip_norm and n > 0:
        delta = delta * (clip_norm / n)
    return delta, n, float(np.linalg.norm(delta))


def add_noise(delta, clip_norm, noise_multiplier, rng):
    delta = np.asarray(delta, dtype=np.float64)
    if noise_multiplier and noise_multiplier > 0:
        if clip_norm is None:
            raise ValueError("noise requires clip_norm")
        std = float(noise_multiplier * clip_norm)
        return delta + rng.normal(0, std, size=delta.shape), std
    return delta, 0.0
