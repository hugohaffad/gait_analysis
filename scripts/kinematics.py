from config import DATA, REP
from functions import *

import moveck_bridge_btk as btk
import matplotlib.pyplot as plt

# ---- File import ---- #
c3d_file = DATA / "Hugo01.c3d"
h = btk.btkReadAcquisition(str(c3d_file))
filename = c3d_file.stem
output_dir = REP / filename
output_dir.mkdir(parents=True, exist_ok=True)

# ---- Variables ---- #

"""
recording sampling frequency = 100 Hz
markers : CHEVILLE_D, CHEVILLE_G, EIAS_D, EIAS_G, GENOU_D, GENOU_G, HANCHE_D, HANCHE_G, PIED_D, PIED_G
position = mm
analog sampling frequency = 1000 Hz
force = N
moment = Nmm
"""

## Markers
markers, markersInfo = btk.btkGetMarkers(h)
freq = btk.btkGetPointFrequency(h)
mrk = {
    "ASIS": {"left": "EIAS_G", "right": "EIAS_D"},
    "Hip":  {"left": "HANCHE_G", "right": "HANCHE_D"},
    "Knee": {"left": "GENOU_G", "right": "GENOU_D"},
    "Ankle":{"left": "CHEVILLE_G", "right": "CHEVILLE_D"},
    "Foot": {"left": "PIED_G", "right": "PIED_D"}
}

## Segments
left_pelvis = markers["EIAS_G"] - markers["HANCHE_G"]
right_pelvis = markers["EIAS_D"] - markers["HANCHE_D"]
left_thigh = markers["GENOU_G"] - markers["HANCHE_G"]
right_thigh = markers["GENOU_D"] - markers["HANCHE_D"]
left_leg = markers["GENOU_G"] - markers["CHEVILLE_G"]
right_leg = markers["GENOU_D"] - markers["CHEVILLE_D"]
left_foot = markers["PIED_G"] - markers["CHEVILLE_G"]
right_foot = markers["PIED_D"] - markers["CHEVILLE_D"]

## Angles
left_hip = 180 - (angle(left_thigh) - angle(left_pelvis))
right_hip = 180 - (angle(right_thigh) - angle(right_pelvis))
left_knee = 180 - (angle(left_thigh) - angle(left_leg))
right_knee = 180 - (angle(right_thigh) - angle(right_leg))
left_ankle = 90 - (angle(left_foot) - angle(left_leg))
right_ankle = 90 - (angle(right_foot) - angle(right_leg))

angles = {
    "Hip":   {"left": left_hip, "right": right_hip},
    "Knee":  {"left": left_knee, "right": right_knee},
    "Ankle": {"left": left_ankle, "right": right_ankle}
}

## Forceplates
forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
grw = btk.btkGetGroundReactionWrenches(h)
fp_freq = btk.btkGetAnalogFrequency(h)
active_forceplates = []

## Events
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

# ---- Gait cycle determination ---- #
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

# ---- Joints angles ---- #
"""
ok pour wrap-around
tjr un pb de offset
"""

## Reports
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

    ax.set_ylabel("Angle (°)", fontsize=10)
    ax.set_xlabel("Gait cycle (%)",fontsize=10)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].legend(loc="upper right")

plt.tight_layout()
plt.savefig(str(output_dir / "joint_angle.png"), dpi=300)
plt.close()