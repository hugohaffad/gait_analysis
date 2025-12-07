import numpy as np

def sort_events(events):
    for side in events:
        for evt in ["HS", "TO"]:
            frames = np.array(events[side][evt]["frame"])
            times  = np.array(events[side][evt]["time"])

            order = np.argsort(frames)

            events[side][evt]["frame"] = frames[order].tolist()
            events[side][evt]["time"]  = times[order].tolist()

    return events


def normalize_cycle(signal, start_frame, end_frame, n_points=101):
    cycle = signal[start_frame:end_frame+1]
    n = len(cycle)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)

    cycle_norm = np.interp(x_new, x_old, cycle)

    return x_new, cycle_norm

def derivee(y, dt, ordre):
    for _ in range(ordre):
        y = np.gradient(y, dt)
    return y

def angle_between(u, v):
    num = np.sum(u * v, axis=1)
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cos_theta = np.clip(num / den, -1, 1)
    return np.rad2deg(np.arccos(cos_theta))