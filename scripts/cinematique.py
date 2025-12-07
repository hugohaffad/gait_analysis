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

#position
labels = list(mrk.keys())

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    left  = markers[mrk[label]["left"]][:, 2]
    right = markers[mrk[label]["right"]][:, 2]

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right")
    ax.plot(gc, left_norm,  label="Left")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[-1].axis("off")

axes[0].legend(loc="upper right")
axes[0].set_ylabel("Position (mm)")
axes[3].set_ylabel("Position (mm)")
axes[3].set_xlabel("Gait cycle (%)")
axes[4].set_xlabel("Gait cycle (%)")

plt.tight_layout()
plt.savefig(str(output_dir / "position.png"), dpi=300)
plt.close()

#linear velocity
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    left  = derivee(markers[mrk[label]["left"]][:, 2], dt,1)
    right = derivee(markers[mrk[label]["right"]][:, 2], dt, 1)

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right")
    ax.plot(gc, left_norm,  label="Left")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[-1].axis("off")

axes[0].legend(loc="upper right")
axes[0].set_ylabel("Velocity (mm/s)")
axes[3].set_ylabel("Velocity (mm/s)")
axes[3].set_xlabel("Gait cycle (%)")
axes[4].set_xlabel("Gait cycle (%)")

plt.tight_layout()
plt.savefig(str(output_dir / "linear_velocity.png"), dpi=300)
plt.close()

#linear acceleration
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    left  = derivee(markers[mrk[label]["left"]][:, 2], dt,2)
    right = derivee(markers[mrk[label]["right"]][:, 2], dt, 2)

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right")
    ax.plot(gc, left_norm,  label="Left")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[-1].axis("off")

axes[0].legend(loc="upper right")
axes[0].set_ylabel("Acceleration (mm/s$^2$)")
axes[3].set_ylabel("Acceleration (mm/s$^2$)")
axes[3].set_xlabel("Gait cycle (%)")
axes[4].set_xlabel("Gait cycle (%)")

plt.tight_layout()
plt.savefig(str(output_dir / "linear_acceleration.png"), dpi=300)
plt.close()