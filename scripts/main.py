import moveck_bridge_btk as btk
import numpy as np
import matplotlib.pyplot as plt
from config import DATA, REP

c3d_file = DATA / "Hugo01.c3d"
filename = c3d_file.stem
output_dir = REP / filename
output_dir.mkdir(parents=True, exist_ok=True)

h = btk.btkReadAcquisition(str(c3d_file))

# Fonctions
def normalize_cycle(signal, idx, n_points=101):
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

def find_vertical_tibia_frame(markers, side):
    genou = markers[f"GENOU_{side}"]
    cheville = markers[f"CHEVILLE_{side}"]
    CGy = genou[:, 1] - cheville[:,1]
    idx_ref = np.argmin(np.abs(CGy))
    return idx_ref

# Variables
## Marqueurs
markers, markersInfo = btk.btkGetMarkers(h)
frames = btk.btkGetPointFrameNumber(h) # frames
freq = btk.btkGetPointFrequency(h) # fréquence d'échantillonage
time = np.arange(frames) / freq # temps
marqueurs = {
    "EIAS": {"G": "EIAS_G", "D" : "EIAS_D"},
    "Hanche": {"G": "HANCHE_G", "D": "HANCHE_D"},
    "Genou": {"G": "GENOU_G", "D": "GENOU_D"},
    "Cheville": {"G": "CHEVILLE_G", "D": "CHEVILLE_D"},
    "Pied": {"G": "PIED_G", "D": "PIED_D"}
}

## Plateformes de force
forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
fpw = btk.btkGetForcePlatformWrenches(h) # force platform wrenches
fp_frames = btk.btkGetAnalogFrameNumber(h) # frames
fp_freq = btk.btkGetAnalogFrequency(h) # fréquence d'échantillonage
fp_time = np.arange(fp_frames) / fp_freq # temps

## Segments
left_hip = []
right_hip = []
left_knee = []
right_knee = []
left_ankle = []
right_ankle = []

# Détermination d'un cycle de marche
for i in range(len(forceplates)): # Identification des plateformes de force actives
    if np.count_nonzero(fpw[i]["F"]) > 0 :
        continue
    else:
        print(f"Plateforme {i} non utilisée")

HS_time = []
L_step_length = []
R_step_length = []
stride_length = []

plt.figure()
for i in range(len(forceplates)-1):
    GRFz = fpw[i]["F"][:,2]
    index = np.where(GRFz > 20)[0][0] # index de la 1e frame où GRFz > 20 N
    HS = index/fp_freq
    HS_time.append(HS)
    plt.plot(fp_time, GRFz, label=f"Plateforme {i}")
    if i == 0:
        plt.axvline(HS_time[i], c="black", label="Heel strike")
    else:
        plt.axvline(HS_time[i], c="black")
plt.plot(time, markers["CHEVILLE_G"][:,2], label="Cheville gauche")
plt.plot(time, markers["CHEVILLE_D"][:,2], label="Cheville droite")
plt.xlabel("Temps (s)")
plt.ylabel("GRFz (N)")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig(str(output_dir / "Identification_cycle_marche.png"))
plt.close()

# Extraction des paramètres spatiotemporels
## Longueur du pas
for i in range(0, len(forceplates)-1):
    GRFz = fpw[i]["F"][:,2]
    index = np.where(GRFz > 20)[0][0]
    HS = index/fp_freq
    HS_time.append(HS)
HS_frames = (np.array(HS_time) * freq).round().astype(int)

for i in range(len(forceplates)-1):
    step_length = abs(markers["CHEVILLE_D"][HS_frames[i], 1] - markers["CHEVILLE_G"][HS_frames[i], 1])
    if markers["CHEVILLE_D"][HS_frames[i], 1] > markers["CHEVILLE_G"][HS_frames[i], 1]:
        R_step_length.append(step_length)
    else:
        L_step_length.append(step_length)
print(f"Longueur moyenne du pas gauche : {np.mean(L_step_length):.1f} mm")
print(f"Longueur moyenne du pas droit : {np.mean(R_step_length):.1f} mm")

## Longueur d'enjambée
for i in range(min(len(R_step_length), len(L_step_length))):
    stride_length.append(R_step_length[i] + L_step_length[i])
print(f"Longueur moyenne d'enjambée : {np.mean(stride_length):.1f} mm")

## Cadence de marche
HS_time = []
for i in range(len(forceplates)-1):
    GRFz = fpw[i]["F"][:,2]
    index = np.where(GRFz > 20)[0][0]
    HS = index/fp_freq
    HS_time.append(HS)
HS_time = np.array(HS_time)
HS_time = np.sort(HS_time)
step_duration = np.diff(HS_time)
mean_step_time = step_duration.mean()
cadence = 60/mean_step_time
print(f"Cadence : {cadence:.1f} pas/min")

## Vitesse
speed = ((np.mean(stride_length)/1000)*cadence)/120
print(f"vitesse de marche : {speed:.1f} m/s")

# Analyse cinématique
## Position des marqueurs (axe Z)
HS_time = []
GRFz0 = fpw[0]["F"][:,2] # You have to choose the right platforms !!
#GRFz1 = fpw[1]["F"][:,2]
GRFz2 = fpw[2]["F"][:,2]
#GRFz3 = fpw[3]["F"][:,2]
for plateforme in (GRFz2, GRFz0) : # And change names here !!
    index = np.where(plateforme > 20)[0][0] # index of the 1st frame at which GRFz > 0
    HS = index/fp_freq
    HS_time.append(HS)

HS_frames = (np.array(HS_time) * freq).round().astype(int)
start_frame, end_frame = HS_frames
cycle_index = np.arange(start_frame, end_frame + 1)

plt.figure()
for landmarks, sides in marqueurs.items():
    gauche = sides["G"]
    droite = sides["D"]

    x_perc, left_norm = normalize_cycle(markers[gauche][:, 2], cycle_index)
    x_perc, right_norm = normalize_cycle(markers[droite][:, 2],  cycle_index)

    plt.plot(x_perc, left_norm, label="Gauche")
    plt.plot(x_perc, right_norm, label="Droite")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.xlabel("Cycle de marche (%)")
    plt.ylabel("Position (mm)")
    plt.xlim(0, 100)
    plt.savefig(output_dir / f"Position_{landmarks}.png")
    plt.close()

## Angles articulaires
for i in range(int(frames)):
    PC_D = markers["CHEVILLE_D"][i, 1:] - markers["PIED_D"][i, 1:]
    GC_D = markers["CHEVILLE_D"][i, 1:] - markers["GENOU_D"][i, 1:]
    PC_G = markers["CHEVILLE_G"][i, 1:] - markers["PIED_G"][i, 1:]
    GC_G = markers["CHEVILLE_G"][i, 1:] - markers["GENOU_G"][i, 1:]
    GH_D = markers["HANCHE_D"][i,1:] - markers["GENOU_D"][i,1:]
    EH_D = markers["HANCHE_D"][i,1:] - markers["EIAS_D"][i,1:]
    GH_G = markers["HANCHE_G"][i,1:] - markers["GENOU_G"][i,1:]
    EH_G = markers["HANCHE_G"][i,1:] - markers["EIAS_G"][i,1:]

    l_hip_agl = 180 - angle_between(GH_G, EH_G)
    r_hip_agl = 180 - angle_between(GH_D, EH_D)
    l_knee_agl = 180 - angle_between(GC_G, GH_G)
    r_knee_agl = 180 - angle_between(GC_D, GH_D)
    l_ankle_agl = 90 - angle_between(PC_G, GC_G)
    r_ankle_agl = 90 - angle_between(PC_D, GC_D)

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

angles = {
    "Hanche": {"G": left_hip, "D": right_hip},
    "Genou": {"G": left_knee, "D": right_knee},
    "Cheville": {"G": left_ankle, "D": right_ankle}
}

for articulation, sides in angles.items():
    gauche = sides["G"]
    droite = sides["D"]

    x_perc, left_norm  = normalize_cycle(gauche, cycle_index)
    x_perc, right_norm = normalize_cycle(droite, cycle_index)

    plt.figure()
    plt.plot(x_perc, left_norm, label="Gauche")
    plt.plot(x_perc, right_norm, label="Droite")
    plt.xlim(0, 100)
    plt.xlabel("Cycle de marche (%)")
    plt.ylabel("Angle (°)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(str(output_dir / f"Angle_{articulation}.png"))
    plt.close()