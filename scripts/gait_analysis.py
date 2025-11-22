# Packages
from pathlib import Path

from fontTools.misc.roundTools import roundFunc

from config import DATA, REP
import moveck_bridge_btk as btk
import matplotlib.pyplot as plt
import numpy as np

# Directories
data_dir = Path(DATA)
c3d_file = data_dir / "Hugo01.c3d" # Insert filename
filename = c3d_file.stem
output_dir = Path(REP / filename)
output_dir.mkdir(parents=True, exist_ok=True)

# Functions
def angle_between(u, v):
    num = np.dot(u, v)
    den = np.linalg.norm(u) * np.linalg.norm(v)
    cos_theta = num / den
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.rad2deg(np.arccos(cos_theta))

def normalize_cycle(signal, idx, n_points=101):
    sig_cycle = signal[idx]
    n = len(sig_cycle)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)

    sig_norm = np.interp(x_new, x_old, sig_cycle)
    return x_new, sig_norm

# Opening file
h = btk.btkReadAcquisition(str(c3d_file))

# Variables
markers, markersInfo = btk.btkGetMarkers(h) # markers
frames = btk.btkGetPointFrameNumber(h) # frames number
freq = btk.btkGetPointFrequency(h) # sampling rate
time = np.arange(frames) / freq # sample time

forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
fpw = btk.btkGetForcePlatformWrenches(h) # force platform wrenches
fp_frames = btk.btkGetAnalogFrameNumber(h) # number of frames
fp_freq = btk.btkGetAnalogFrequency(h) # sampling rate
fp_time = np.arange(fp_frames) / fp_freq # sample time

mrk_pairs = [
    ("EIAS_G", "EIAS_D"),
    ("HANCHE_G", "HANCHE_D"),
    ("GENOU_G", "GENOU_D"),
    ("CHEVILLE_G", "CHEVILLE_D"),
    ("PIED_G", "PIED_D")
]

HS_time = []
L_step_length = []
R_step_length = []
stride_length = []

# Gait cycle identification
for i in range(len(forceplates)) :
    if np.count_nonzero(fpw[i]["F"][:,2]) > 0 :
        continue
    else:
        print(f" force platform {i} non-used")

plt.figure()
for i in range(len(forceplates)-1):
    grf_z = fpw[i]["F"][:,2]
    index = np.where(grf_z > 20)[0][0] # index of the 1st frame at which GRFz > 0
    HS = index/fp_freq
    HS_time.append(HS)
    plt.plot(fp_time, grf_z, label=f"Platform {i}")
    if i == 0:
        plt.axvline(HS_time[i], c="black", label="Heel strike")
    else:
        plt.axvline(HS_time[i], c="black")
plt.plot(time, markers["CHEVILLE_D"][:,2], label="CHEVILLE_D")
plt.plot(time, markers["CHEVILLE_G"][:,2], label="CHEVILLE_G")
plt.xlabel("Time (s)")
plt.ylabel("GRFz (N.mm)")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig(str(output_dir / "gait_cycle_identification.png"))
plt.close()

# Spatiotemporal parameters
## step_length
for i in range(0, len(forceplates)-1):
    grf_z = fpw[i]["F"][:,2]
    index = np.where(grf_z > 20)[0][0] # index of the 1st frame at which GRFz > 0
    HS = index/fp_freq
    HS_time.append(HS)
HS_frames = (np.array(HS_time) * freq).round().astype(int)

for i in range(len(forceplates)-1):
    step_length = abs(markers["CHEVILLE_D"][HS_frames[i], 1] - markers["CHEVILLE_G"][HS_frames[i], 1])
    if markers["CHEVILLE_D"][HS_frames[i], 1] > markers["CHEVILLE_G"][HS_frames[i], 1]:
        R_step_length.append(step_length)
    else:
        L_step_length.append(step_length)
print(f"Longueur moyenne du pas gauche : {np.mean(L_step_length)} mm")
print(f"Longueur moyenne du pas droit : {np.mean(R_step_length)} mm")

## stride_length
for i in range(min(len(R_step_length), len(L_step_length))):
    stride_length.append(R_step_length[i] + L_step_length[i])
print(f"Mean stride Length : {np.mean(stride_length)} mm")

## cadence
HS_time = []
for i in range(0, len(forceplates)-1):
    grf_z = fpw[i]["F"][:,2]
    index = np.where(grf_z > 20)[0][0]
    HS = index/fp_freq
    HS_time.append(HS)
HS_time = np.array(HS_time)
HS_time = np.sort(HS_time)
step_durations = np.diff(HS_time)
mean_step_time = step_durations.mean()
cadence = round(60/mean_step_time,0)
print(f"Cadence : {cadence} pas/min")

## vitesse
vitesse = round(((np.mean(stride_length)/1000)*cadence)/120,2)
print(f"vitesse de marche : {vitesse} m/s")

# Kinematics
## Positions
HS_time = []
grfz0 = fpw[0]["F"][:,2] # You have to choose the right platforms !!
#grfz1 = fpw[1]["F"][:,2]
grfz2 = fpw[2]["F"][:,2]
#grfz3 = fpw[3]["F"][:,2]
for plateforme in (grfz2, grfz0) : # And change names here !!
    index = np.where(plateforme > 20)[0][0] # index of the 1st frame at which GRFz > 0
    HS = index/fp_freq
    HS_time.append(HS)

HS_frames = (np.array(HS_time) * freq).round().astype(int)
start_frame, end_frame = HS_frames
cycle_index = np.arange(start_frame, end_frame + 1)

for i, (gauche, droite) in enumerate(mrk_pairs):
    x_perc, left_norm = normalize_cycle(markers[gauche][:, 2], cycle_index)
    x_perc, right_norm = normalize_cycle(markers[droite][:, 2],  cycle_index)

    plt.figure()
    plt.plot(x_perc, left_norm, label=f"{gauche}")
    plt.plot(x_perc, right_norm, label=f"{droite}")
    plt.xlim(0, 100)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.xlabel("Gait cycle (%)", fontsize=10)
    plt.ylabel("Vertical position (mm)", fontsize=10)
    plt.savefig(str(output_dir / f"position_{i}.png"))
    plt.close()

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

left_hip = np.array(left_hip)
right_hip = np.array(right_hip)
left_knee = np.array(left_knee)
right_knee = np.array(right_knee)
left_ankle = np.array(left_ankle)
right_ankle = np.array(right_ankle)

agl_pairs = [
    (left_hip, right_hip),
    (left_knee, right_knee),
    (left_ankle, right_ankle)
]

    # plots
for i, (gauche, droite) in enumerate(agl_pairs):
    x_perc, left_norm  = normalize_cycle(gauche, cycle_index)
    x_perc, right_norm = normalize_cycle(droite, cycle_index)

    plt.figure()
    plt.plot(x_perc, left_norm, label="Left")
    plt.plot(x_perc, right_norm, label="Right")
    plt.xlim(0, 100)
    plt.xlabel("Gait cycle (%)")
    plt.ylabel("Angle (°)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(str(output_dir / f"angle_{i}.png"))
    plt.close()

# Kinetics