import numpy as np

def angle_between(u, v):
    num = np.dot(u, v)
    den = np.linalg.norm(u) * np.linalg.norm(v)
    cos_theta = num / den
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.rad2deg(np.arccos(cos_theta))

def normalize_cycle(signal, idx, n_points=101):
    sig_cycle = signal[idx]
    n = len(sig_cycle)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)

    sig_norm = np.interp(x_new, x_old, sig_cycle)
    return x_new, sig_norm