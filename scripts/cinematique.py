from config import DATA, REP
from functions import *

import moveck_bridge_btk as btk
import numpy as np
import matplotlib.pyplot as plt

from scripts.functions import derivee

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

#joints angles
labels = list(angles.keys())

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    right = angles[label]["right"]
    left = angles[label]["left"]

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right")
    ax.plot(gc, left_norm,  label="Left")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].legend(loc="upper right")
axes[0].set_ylabel("Angle (°)")
axes[0].set_xlabel("Gait cycle (%)")

plt.tight_layout()
plt.savefig(str(output_dir / "joint_angle.png"), dpi=300)
plt.close()

#angular velocity
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    right = derivee(angles[label]["right"], dt, 1)
    left = derivee(angles[label]["left"], dt, 1)

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right")
    ax.plot(gc, left_norm,  label="Left")

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
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

for i, label in enumerate(labels):
    ax = axes[i]

    right = derivee(angles[label]["right"], dt, 2)
    left = derivee(angles[label]["left"], dt, 2)

    gc, right_norm = normalize_cycle(right, start_frame, end_frame)
    _,  left_norm  = normalize_cycle(left,  start_frame, end_frame)

    ax.plot(gc, right_norm, label="Right")
    ax.plot(gc, left_norm,  label="Left")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].legend(loc="upper right")
axes[0].set_ylabel("Angular acceleration (°/s$^2$)")
axes[0].set_xlabel("Gait cycle (%)")

plt.tight_layout()
plt.savefig(str(output_dir / "angular_acceleration.png"), dpi=300)
plt.close()