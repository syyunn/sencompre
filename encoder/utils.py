import numpy as np


def l2dist(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def l2dist(u, v):
    return np.linalg.norm(u-v)
