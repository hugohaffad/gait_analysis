# Packages
from pathlib import Path
from config import DATA, REP
from functions import angle_between

import moveck_bridge_btk as btk
import matplotlib.pyplot as plt
import numpy as np

# Directories
data_dir = Path(DATA)
c3d_file = data_dir / "Hugo01.c3d" # Insert filename
filename = c3d_file.stem
output_dir = Path(REP / filename)
output_dir.mkdir(parents=True, exist_ok=True)

h = btk.btkReadAcquisition(str(c3d_file))

# Variables
markers, markersInfo = btk.btkGetMarkers(h) # markers
fpw = btk.btkGetForcePlatformWrenches(h) # force platform wrenches
frames = btk.btkGetPointFrameNumber(h) # frames number

# Inverse kinematics
## Positions
mrk_pairs = [
    ("EIAS_G", "EIAS_D"),
    ("HANCHE_G", "HANCHE_D"),
    ("GENOU_G", "GENOU_D"),
    ("CHEVILLE_G", "CHEVILLE_D"),
    ("PIED_G", "PIED_D")
]

for i, (gauche, droite) in enumerate(mrk_pairs):
    plt.figure()

    plt.subplot(311) #X-axis
    plt.plot(markers[gauche][:,0], label="Left")
    plt.plot(markers[droite][:,0], label="Right")
    plt.ylabel("X axis (mm)")
    plt.legend(loc="upper right")

    plt.subplot(312) #Y-axis
    plt.plot(markers[gauche][:,1])
    plt.plot(markers[droite][:,1])
    plt.ylabel("Y axis (mm)")

    plt.subplot(313) #Z-axis
    plt.plot(markers[gauche][:,2])
    plt.plot(markers[droite][:,2])
    plt.ylabel("Z axis (mm)")
    plt.xlabel("Frame")

    plt.subplots_adjust(hspace=0.6)

    plt.savefig(str(output_dir / f"position_{i}.png"))

## Angles
left_hip = []
right_hip = []
left_knee = []
right_knee = []
left_ankle = []
right_ankle = []

for i in range(int(frames)):
    # segments
    r_PC = markers["CHEVILLE_D"][i,1:] - markers["PIED_D"][i,1:]
    r_GC = markers["CHEVILLE_D"][i,1:] - markers["GENOU_D"][i,1:]
    r_GH = markers["HANCHE_D"][i,1:] - markers["GENOU_D"][i,1:]
    r_EH = markers["HANCHE_D"][i,1:] - markers["EIAS_D"][i,1:]
    l_PC = markers["CHEVILLE_G"][i,1:] - markers["PIED_G"][i,1:]
    l_GC = markers["CHEVILLE_G"][i,1:] - markers["GENOU_G"][i,1:]
    l_GH = markers["HANCHE_G"][i,1:] - markers["GENOU_G"][i,1:]
    l_EH = markers["HANCHE_G"][i,1:] - markers["EIAS_G"][i,1:]

    # angles
    l_hip_agl = 180 - angle_between(l_GH, l_EH)
    r_hip_agl = 180 - angle_between(r_GH, r_EH)
    l_knee_agl = 180 - angle_between(l_GC, l_GH)
    r_knee_agl = 180 - angle_between(r_GC, r_GH)
    l_ankle_agl = 90 - angle_between(l_PC, l_GC)
    r_ankle_agl = 90 - angle_between(r_PC, r_GC)

    left_hip.append(l_hip_agl)
    right_hip.append(r_hip_agl)
    left_knee.append(l_knee_agl)
    right_knee.append(r_knee_agl)
    left_ankle.append(l_ankle_agl)
    right_ankle.append(r_ankle_agl)

    # plots
plt.figure()
plt.plot(left_hip, label="Left")
plt.plot(right_hip, label="Right")
plt.ylabel("Hip angle (°)")
plt.legend(loc="upper right")
plt.savefig(str(output_dir / f"angle_hip.png"))

plt.figure()
plt.plot(left_knee, label="Left")
plt.plot(right_knee, label="Right")
plt.ylabel("Knee angle (°)")
plt.legend(loc="upper right")
plt.savefig(str(output_dir / f"angle_knee.png"))

plt.figure()
plt.plot(left_ankle, label="Left")
plt.plot(right_ankle, label="Right")
plt.ylabel("Ankle angle (°)")
plt.legend(loc="upper right")
plt.savefig(str(output_dir / f"angle_ankle.png"))