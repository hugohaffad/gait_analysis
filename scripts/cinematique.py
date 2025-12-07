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

marker = {
    "left": {"ASIS": "EIAS_G", "hip": "HANCHE_G", "knee": "GENOU_G", "ankle": "CHEVILLE_G", "foot": "PIED_G"},
    "right": {"ASIS": "EIAS_D", "hip": "HANCHE_D", "knee": "GENOU_D", "ankle": "CHEVILLE_D", "foot": "PIED_D"}
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

print(events)

#
# if events["right"]["HS"]["frame"][0] < events["left"]["HS"]["frame"][0]:
#     side = "right"
# else:
#     side = "left"
# start_frame, end_frame = events[side]["HS"]["frame"]
# gc_percent, l_ankle_norm = normalize_cycle(markers[marker["left"]["ankle"]][:, 2], start_frame, end_frame)
# _,          r_ankle_norm = normalize_cycle(markers[marker["right"]["ankle"]][:, 2], start_frame, end_frame)
#
# plt.figure()
# plt.plot(gc_percent, l_ankle_norm, label="Left")
# plt.plot(gc_percent, r_ankle_norm, label="Right")
# plt.xlim(0, 100)
# plt.legend(loc="best")
# plt.show()

#analyse cinématique = trajectoires, vitesses et accélérations