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

def angle(v):
    theta = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 2]), period=2*np.pi))
    return theta