import numpy as np

def normalize_cycle(signal, start_frame, end_frame, n_points=101):
    cycle = signal[start_frame:end_frame+1]
    n = len(cycle)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)

    cycle_norm = np.interp(x_new, x_old, cycle)

    return x_new, cycle_norm