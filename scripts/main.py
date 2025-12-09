from config import DATA, REP
from functions import *

import moveck_bridge_btk as btk
import numpy as np
import matplotlib.pyplot as plt

#file import
c3d_file = DATA / "Hugo01.c3d"
h = btk.btkReadAcquisition(str(c3d_file))
filename = c3d_file.stem
output_dir = REP / filename
output_dir.mkdir(parents=True, exist_ok=True)

#variables
"""
recording sampling frequency = 100 Hz
markers : CHEVILLE_D, CHEVILLE_G, EIAS_D, EIAS_G, GENOU_D, GENOU_G, HANCHE_D, HANCHE_G, PIED_D, PIED_G
position = mm
analog sampling frequency = 1000 Hz
force = N
moment = Nmm
"""

markers, markersInfo = btk.btkGetMarkers(h)
frame = btk.btkGetPointFrameNumber(h)
freq = btk.btkGetPointFrequency(h)
time = np.arange(frame)/freq
dt = 1 / freq

mrk = {
    "ASIS": {"left": "EIAS_G", "right": "EIAS_D"},
    "Hip":  {"left": "HANCHE_G", "right": "HANCHE_D"},
    "Knee": {"left": "GENOU_G", "right": "GENOU_D"},
    "Ankle":{"left": "CHEVILLE_G", "right": "CHEVILLE_D"},
    "Foot": {"left": "PIED_G", "right": "PIED_D"}
}

forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
grw = btk.btkGetGroundReactionWrenches(h)
fp_frame = btk.btkGetAnalogFrameNumber(h)
fp_freq = btk.btkGetAnalogFrequency(h)
fp_time = np.arange(fp_frame)/fp_freq
active_forceplates = []

events = {
    "left": {
        "HS": {"frame": [], "time": []},
        "TO": {"frame": [], "time": []}
    },
    "right": {
        "HS": {"frame": [], "time": []},
        "TO": {"frame": [], "time": []},
    }
}

left_foot = markers["PIED_G"][:,1:] - markers["CHEVILLE_G"][:,1:]
right_foot = markers["PIED_D"][:,1:] - markers["CHEVILLE_D"][:,1:]
left_leg = markers["GENOU_G"][:,1:] - markers["CHEVILLE_G"][:,1:]
right_leg = markers["GENOU_D"][:,1:] - markers["CHEVILLE_D"][:,1:]
left_thigh = markers["GENOU_G"][:,1:] - markers["HANCHE_G"][:,1:]
right_thigh = markers["GENOU_D"][:,1:] - markers["HANCHE_D"][:,1:]
left_pelvis = markers["EIAS_G"][:,1:] - markers["HANCHE_G"][:,1:]
right_pelvis = markers["EIAS_D"][:,1:] - markers["HANCHE_D"][:,1:]

left_ankle = 90 - angle_between(left_foot, left_leg)
right_ankle = 90 - angle_between(right_foot, right_leg)
left_knee = 180 - angle_between(left_leg, left_thigh)
right_knee = 180 - angle_between(right_leg, right_thigh)
left_hip = 180 - angle_between(left_thigh, left_pelvis)
right_hip = 180 - angle_between(right_thigh, right_pelvis)

angles = {
    "Hip":   {"left": left_hip,   "right": right_hip},
    "Knee":  {"left": left_knee,      "right": right_knee},
    "Ankle": {"left": left_ankle,     "right": right_ankle}
}

# --- Table anthropométrique ---
len_frac = {
    "Foot": 0.152,
    "Leg":  0.246,
    "Thigh":0.18
}
mass_frac = {
    "Foot": 0.0145,
    "Leg":  0.0465,
    "Thigh":0.10
}
com_frac = {
    "Foot": 0.50,
    "Leg":  0.433,
    "Thigh":0.433
}
rg_frac = {
    "Foot": 0.475,
    "Leg":  0.302,
    "Thigh":0.323
}
# ------------------------------

body_mass = 70
body_height = 1.84
g  = 9.806

anthropo = {
    "Foot": {
        "mass": None,
        "len": None,
        "com": None,
        "radius_gyr": None
    },
    "Leg": {
        "mass": None,
        "len": None,
        "com": None,
        "radius_gyr": None
    },
    "Thigh": {
        "mass": None,
        "len": None,
        "com": None,
        "radius_gyr": None
    }
}

#gait cycle determination
for i in range(len(forceplates)):
    if np.count_nonzero(grw[i]["F"]) == 0:
        continue
    else:
        active_forceplates.append(i)

for i in active_forceplates:
    GRFz = grw[i]["F"][:,2]

    HS = np.where(GRFz > 5)[0][0]
    HS_time = HS/fp_freq
    HS_frame = int(np.round(HS_time * freq))

    TO = np.where(GRFz > 5)[0][-1]
    TO_time = TO/fp_freq
    TO_frame = int(np.round(TO_time * freq))

    if markers["CHEVILLE_D"][HS_frame, 2] < markers["CHEVILLE_G"][HS_frame, 2]:
        side = "right"
    else:
        side = "left"

    events[side]["HS"]["frame"].append(HS_frame)
    events[side]["HS"]["time"].append(HS_time)
    events[side]["TO"]["frame"].append(TO_frame)
    events[side]["TO"]["time"].append(TO_time)

events = sort_events(events)

if events["right"]["HS"]["frame"][0] < events["left"]["HS"]["frame"][0]:
    side = "right"
else:
    side = "left"
start_frame, end_frame = events[side]["HS"]["frame"][0], events[side]["HS"]["frame"][1]

#set ankle reference angles
window_R = right_leg[events["right"]["HS"]["frame"][0]:events["right"]["TO"]["frame"][0], 0]
idx_local_R = int(np.argmin(np.abs(window_R)))
idx_global_R = events["right"]["HS"]["frame"][0] + idx_local_R
ankle_R = np.asarray(angles["Ankle"]["right"])
angle_ref_R = ankle_R[idx_global_R]
angles["Ankle"]["right"] = ankle_R - angle_ref_R

window_L = left_leg[events["left"]["HS"]["frame"][0]:events["left"]["TO"]["frame"][0], 0]
idx_local_L = int(np.argmin(np.abs(window_L)))
idx_global_L = events["left"]["HS"]["frame"][0] + idx_local_L
ankle_L = np.asarray(angles["Ankle"]["left"])
angle_ref_L = ankle_L[idx_global_L]
angles["Ankle"]["left"] = ankle_L - angle_ref_L

#joints angles
labels = list(angles.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    right = angles[label]["right"]
    left = angles[label]["left"]

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right", linewidth=2)
    ax.plot(gc, left_norm,  label="Left", linewidth=2)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].legend(loc="upper right")

fig.supylabel("Angle (°)", fontsize=10, x=0.01)
fig.supxlabel("Gait cycle (%)",fontsize=10, y = 0.03)

plt.tight_layout()
plt.savefig(str(output_dir / "joint_angle.png"), dpi=300)
plt.close()

#angular velocity
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    right = derivee(angles[label]["right"], dt, 1)
    left = derivee(angles[label]["left"], dt, 1)

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right", linewidth=2)
    ax.plot(gc, left_norm,  label="Left", linewidth=2)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].legend(loc="upper right")
axes[0].set_ylabel("Angular velocity (°/s)")
axes[0].set_xlabel("Gait cycle (%)")

plt.tight_layout()
plt.savefig(str(output_dir / "angular_velocity.png"), dpi=300)
plt.close()

#angular acceleration
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    right = derivee(angles[label]["right"], dt, 2)
    left = derivee(angles[label]["left"], dt, 2)

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right", linewidth=2)
    ax.plot(gc, left_norm,  label="Left", linewidth=2)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].legend(loc="upper right")
axes[0].set_ylabel("Angular acceleration (°/s$^2$)")
axes[0].set_xlabel("Gait cycle (%)")

plt.tight_layout()
plt.savefig(str(output_dir / "angular_acceleration.png"), dpi=300)
plt.close()