from config import DATA, REP

import moveck_bridge_btk as btk
import numpy as np
import matplotlib.pyplot as plt

c3d_file = DATA / "Hugo01.c3d"
filename = c3d_file.stem
output_dir = REP / filename
output_dir.mkdir(parents=True, exist_ok=True)

h = btk.btkReadAcquisition(str(c3d_file))

# Fonctions
def normalize_cycle(signal, idx, n_points=101): #Normalisation cycle de marche
    sig_cycle = signal[idx]
    n = len(sig_cycle)
    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)
    sig_norm = np.interp(x_new, x_old, sig_cycle)
    return x_new, sig_norm

def angle_between(u, v):
    num = np.dot(u, v)
    den = np.linalg.norm(u) * np.linalg.norm(v)
    cos_theta = num / den
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.rad2deg(np.arccos(cos_theta))

# Variables
markers, markersInfo = btk.btkGetMarkers(h)
frame = btk.btkGetPointFrameNumber(h) # frames
freq = btk.btkGetPointFrequency(h) # fréquence d'échantillonage
time = np.arange(frame) / freq # temps

forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
fpw = btk.btkGetForcePlatformWrenches(h) # force platform wrenches
fp_frame = btk.btkGetAnalogFrameNumber(h) # frames
fp_freq = btk.btkGetAnalogFrequency(h) # fréquence d'échantillonage
fp_time = np.arange(fp_frame) / fp_freq # temps
active_forceplates = []

mark = {
    "gauche": {
        "EIAS": "EIAS_G",
        "hanche": "HANCHE_G",
        "genou": "GENOU_G",
        "cheville": "CHEVILLE_G",
        "pied": "PIED_G"
    },
    "droite": {
        "EIAS": "EIAS_D",
        "hanche": "HANCHE_D",
        "genou": "GENOU_D",
        "cheville": "CHEVILLE_D",
        "pied": "PIED_D"
    }
}

events = {
    "gauche": {
        "HS": {"frame": [], "time": []},
        "TO": {"frame": [], "time": []}
    },
    "droite": {
        "HS": {"frame": [], "time": []},
        "TO": {"frame": [], "time": []},
    }
}

ST_parameters = {
    "gauche": {
        "step_length": [],
        "stride_length": [],
        "cycle_time": [],
        "cycle_frame": []
    },
    "droite": {
        "step_length": [],
        "stride_length": [],
        "cycle_time": [],
        "cycle_frame": []
    }
}

# Identification des plateformes de force actives
for i in range(len(forceplates)):
    if np.count_nonzero(fpw[i]["F"]) > 0 :
        active_forceplates.append(i)
    else:
        print(f"Plateforme {i} non utilisée")

# Détection des évènements de marche (HS, TO)
for i in active_forceplates:
    GRFz = fpw[i]["F"][:,2]

    HS = np.where(GRFz > 5)[0][0]  # Détection des heel strikes
    HS_time = HS/fp_freq
    HS_frame = int(np.round(HS_time * freq))

    TO = np.where(GRFz > 5)[0][-1] # Détection des toe off
    TO_time = TO/fp_freq
    TO_frame = int(np.round(TO_time * freq))

    if markers["CHEVILLE_D"][HS_frame, 2] < markers["CHEVILLE_G"][HS_frame, 2]:
        side = "droite"
    else:
        side = "gauche"

    events[side]["HS"]["frame"].append(HS_frame)
    events[side]["HS"]["time"].append(HS_time)
    events[side]["TO"]["frame"].append(TO_frame)
    events[side]["TO"]["time"].append(TO_time)

# Détermination d'un cycle de marche
plt.figure()
for side in ["gauche", "droite"]:
    start_frame, end_frame = np.sort(events[side]["HS"]["frame"])
    cycle_frame = end_frame + 1 - start_frame

    ST_parameters[side]["cycle_frame"].append(cycle_frame)
    ST_parameters[side]["cycle_time"].append(cycle_frame/freq)

    print(f"Durée du cycle de marche {side} : {cycle_frame/freq:.2f} s")

    print(start_frame, end_frame)