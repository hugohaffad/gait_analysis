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

# Inverse dynamics
#
# # 1) Paramètres anthropométriques (pied, jambe, cuisse)
# for segment in anthropo.keys():
#     anthropo[segment]["len"] = body_height * len_frac[segment]
#     anthropo[segment]["mass"] = body_mass * mass_frac[segment]
#     anthropo[segment]["com"] = anthropo[segment]["len"] * com_frac[segment]
#     anthropo[segment]["radius_gyr"] = anthropo[segment]["len"] * rg_frac[segment]
#
# markers_m = {name: arr / 1000.0 for name, arr in markers.items()}
#
# # =========================================================
# # INVERSE DYNAMICS – LEFT SIDE
# # =========================================================
#
# # ---------- ANKLE ----------
#
# # Inertie du pied
# m_foot_l  = anthropo["Foot"]["mass"]
# k_foot_l  = anthropo["Foot"]["radius_gyr"]
# Ic_foot_l = m_foot_l * k_foot_l**2
#
# # Vecteur pied
# left_foot_m = markers_m["PIED_G"][:, 1:] - markers_m["CHEVILLE_G"][:, 1:]
#
# # Angle absolu
# theta_l_foot = np.arctan2(left_foot_m[:, 1], left_foot_m[:, 0])
# alpha_l_foot = derivee(theta_l_foot, dt, 2)
#
# # COM pied
# com_l_foot = markers_m["CHEVILLE_G"][:, 1:] + com_frac["Foot"] * left_foot_m
#
# r_yC_l = com_l_foot[:, 0] - markers_m["CHEVILLE_G"][:, 1]
# r_zC_l = com_l_foot[:, 1] - markers_m["CHEVILLE_G"][:, 2]
#
# acc_com_l_y = derivee(com_l_foot[:, 0], dt, 2)
# acc_com_l_z = derivee(com_l_foot[:, 1], dt, 2)
#
# # CoP / GRF
# cop_y_fp_l = grw[1]["P"][:, 1]
# cop_z_fp_l = grw[1]["P"][:, 2]
# Fy_fp_l = grw[1]["F"][:, 1]
# Fz_fp_l = grw[1]["F"][:, 2]
#
# cop_y_l = np.interp(time, fp_time, cop_y_fp_l) / 1000.0
# cop_z_l = np.interp(time, fp_time, cop_z_fp_l) / 1000.0
# Fy_l = np.interp(time, fp_time, Fy_fp_l)
# Fz_l = np.interp(time, fp_time, Fz_fp_l)
#
# r_yP_l = cop_y_l - markers_m["CHEVILLE_G"][:, 1]
# r_zP_l = cop_z_l - markers_m["CHEVILLE_G"][:, 2]
#
# term_rot_foot_l = Ic_foot_l * alpha_l_foot
# term_trans_foot_l = m_foot_l * (r_yC_l * acc_com_l_z - r_zC_l * acc_com_l_y)
# term_grav_foot_l = m_foot_l * g * r_yC_l
# M_grf_foot_l = r_yP_l * Fz_l - r_zP_l * Fy_l
#
# M_ankle_left_full = term_rot_foot_l + term_trans_foot_l + term_grav_foot_l + M_grf_foot_l
# M_ankle_left_full_norm = M_ankle_left_full / body_mass
#
# # ---------- Leg ----------
#
# m_leg_l  = anthropo["Leg"]["mass"]
# k_leg_l  = anthropo["Leg"]["radius_gyr"]
# Ic_leg_l = m_leg_l * k_leg_l**2
#
# # Réaction à la cheville (sur le pied)
# R_A_y_l = m_foot_l * acc_com_l_y - Fy_l
# R_A_z_l = m_foot_l * acc_com_l_z + m_foot_l * g - Fz_l
#
# # Force distale sur la jambe
# Fdist_y_l = -R_A_y_l
# Fdist_z_l = -R_A_z_l
#
# # Cinématique jambe (genou → cheville)
# left_leg_m = markers_m["CHEVILLE_G"][:, 1:] - markers_m["GENOU_G"][:, 1:]
# theta_l_leg = np.arctan2(left_leg_m[:, 1], left_leg_m[:, 0])
# alpha_l_leg = derivee(theta_l_leg, dt, 2)
#
# com_l_leg = markers_m["GENOU_G"][:, 1:] + com_frac["Leg"] * left_leg_m
#
# r_yC_leg_l = com_l_leg[:, 0] - markers_m["GENOU_G"][:, 1]
# r_zC_leg_l = com_l_leg[:, 1] - markers_m["GENOU_G"][:, 2]
#
# acc_com_leg_y_l = derivee(com_l_leg[:, 0], dt, 2)
# acc_com_leg_z_l = derivee(com_l_leg[:, 1], dt, 2)
#
# # Réaction proximale au genou (sur la jambe)
# R_K_y_l = m_leg_l * acc_com_leg_y_l - Fdist_y_l
# R_K_z_l = m_leg_l * acc_com_leg_z_l + m_leg_l * g - Fdist_z_l
#
# # Bras de levier genou → cheville
# r_yA_leg_l = markers_m["CHEVILLE_G"][:, 1] - markers_m["GENOU_G"][:, 1]
# r_zA_leg_l = markers_m["CHEVILLE_G"][:, 2] - markers_m["GENOU_G"][:, 2]
#
# term_rot_leg_l = Ic_leg_l * alpha_l_leg
# term_trans_leg_l = m_leg_l * (r_yC_leg_l * acc_com_leg_z_l - r_zC_leg_l * acc_com_leg_y_l)
# term_grav_leg_l = m_leg_l * g * r_yC_leg_l
#
# M_dist_leg_l = -M_ankle_left_full
# moment_distal_force_l = -(r_yA_leg_l * Fdist_z_l - r_zA_leg_l * Fdist_y_l)
#
# M_knee_left = term_rot_leg_l + term_trans_leg_l + term_grav_leg_l + moment_distal_force_l + M_dist_leg_l
# M_knee_left_norm = M_knee_left / body_mass
#
# # ---------- Thigh ----------
#
# m_thigh_l  = anthropo["Thigh"]["mass"]
# k_thigh_l  = anthropo["Thigh"]["radius_gyr"]
# Ic_thigh_l = m_thigh_l * k_thigh_l**2
#
# # Force distale au genou
# FdistK_y_l = -R_K_y_l
# FdistK_z_l = -R_K_z_l
#
# # Cinématique cuisse
# left_thigh_m = markers_m["GENOU_G"][:, 1:] - markers_m["HANCHE_G"][:, 1:]
# theta_l_thigh = np.arctan2(left_thigh_m[:, 1], left_thigh_m[:, 0])
# alpha_l_thigh = derivee(theta_l_thigh, dt, 2)
#
# com_l_thigh = markers_m["HANCHE_G"][:, 1:] + com_frac["Thigh"] * left_thigh_m
#
# r_yC_th_l = com_l_thigh[:, 0] - markers_m["HANCHE_G"][:, 1]
# r_zC_th_l = com_l_thigh[:, 1] - markers_m["HANCHE_G"][:, 2]
#
# acc_com_th_y_l = derivee(com_l_thigh[:, 0], dt, 2)
# acc_com_th_z_l = derivee(com_l_thigh[:, 1], dt, 2)
#
# # Bras de levier hanche -> genou
# r_yK_th_l = markers_m["GENOU_G"][:, 1] - markers_m["HANCHE_G"][:, 1]
# r_zK_th_l = markers_m["GENOU_G"][:, 2] - markers_m["HANCHE_G"][:, 2]
#
# term_rot_th_l   = Ic_thigh_l * alpha_l_thigh
# term_trans_th_l = m_thigh_l * (r_yC_th_l * acc_com_th_z_l - r_zC_th_l * acc_com_th_y_l)
# term_grav_th_l  = m_thigh_l * g * r_yC_th_l
#
# moment_distal_force_th_l = -(r_yK_th_l * FdistK_z_l - r_zK_th_l * FdistK_y_l)
# M_dist_thigh_l = -M_knee_left
#
# M_hip_left = term_rot_th_l + term_trans_th_l + term_grav_th_l + moment_distal_force_th_l + M_dist_thigh_l
# M_hip_left_norm = M_hip_left / body_mass
#
# # =========================================================
# # INVERSE DYNAMICS – RIGHT SIDE
# # =========================================================
#
# # ---------- Foot ----------
#
# # Inertie du pied
# m_foot_r = anthropo["Foot"]["mass"]
# k_foot_r = anthropo["Foot"]["radius_gyr"]
# Ic_foot_r = m_foot_r * k_foot_r**2
#
# # Vecteur pied droit (cheville → avant-pied) en m, YZ
# right_foot_m = markers_m["PIED_D"][:, 1:] - markers_m["CHEVILLE_D"][:, 1:]
#
# # Angle absolu du pied droit
# theta_r_foot = np.arctan2(right_foot_m[:, 1], right_foot_m[:, 0])
# alpha_r_foot = derivee(theta_r_foot, dt, 2)
#
# # COM pied droit
# com_r_foot = markers_m["CHEVILLE_D"][:, 1:] + com_frac["Foot"] * right_foot_m
#
# r_yC_r = com_r_foot[:, 0] - markers_m["CHEVILLE_D"][:, 1]
# r_zC_r = com_r_foot[:, 1] - markers_m["CHEVILLE_D"][:, 2]
#
# acc_com_r_y = derivee(com_r_foot[:, 0], dt, 2)
# acc_com_r_z = derivee(com_r_foot[:, 1], dt, 2)
#
# # CoP / GRF
# cop_y_fp_r = grw[0]["P"][:, 1]
# cop_z_fp_r = grw[0]["P"][:, 2]
# Fy_fp_r = grw[0]["F"][:, 1]
# Fz_fp_r = grw[0]["F"][:, 2]
#
# cop_y_r = np.interp(time, fp_time, cop_y_fp_r) / 1000.0
# cop_z_r = np.interp(time, fp_time, cop_z_fp_r) / 1000.0
# Fy_r = np.interp(time, fp_time, Fy_fp_r)
# Fz_r = np.interp(time, fp_time, Fz_fp_r)
#
# r_yP_r = cop_y_r - markers_m["CHEVILLE_D"][:, 1]
# r_zP_r = cop_z_r - markers_m["CHEVILLE_D"][:, 2]
#
# term_rot_foot_r = Ic_foot_r * alpha_r_foot
# term_trans_foot_r = m_foot_r * (r_yC_r * acc_com_r_z - r_zC_r * acc_com_r_y)
# term_grav_foot_r = m_foot_r * g * r_yC_r
# M_grf_foot_r = r_yP_r * Fz_r - r_zP_r * Fy_r
#
# M_ankle_right_full = term_rot_foot_r + term_trans_foot_r + term_grav_foot_r + M_grf_foot_r
# M_ankle_right_full_norm = M_ankle_right_full / body_mass
#
# # ---------- Leg ----------
#
# m_leg_r  = anthropo["Leg"]["mass"]
# k_leg_r  = anthropo["Leg"]["radius_gyr"]
# Ic_leg_r = m_leg_r * k_leg_r**2
#
# # Réaction à la cheville (sur le pied droit)
# R_A_y_r = m_foot_r * acc_com_r_y - Fy_r
# R_A_z_r = m_foot_r * acc_com_r_z + m_foot_r * g - Fz_r
#
# # Force distale sur la jambe droite = pied droit sur jambe droite
# Fdist_y_r = -R_A_y_r
# Fdist_z_r = -R_A_z_r
#
# # Cinématique jambe droite (genou → cheville)
# right_leg_m = markers_m["CHEVILLE_D"][:, 1:] - markers_m["GENOU_D"][:, 1:]
# theta_r_leg = np.arctan2(right_leg_m[:, 1], right_leg_m[:, 0])
# alpha_r_leg = derivee(theta_r_leg, dt, 2)
#
# com_r_leg = markers_m["GENOU_D"][:, 1:] + com_frac["Leg"] * right_leg_m
#
# r_yC_leg_r = com_r_leg[:, 0] - markers_m["GENOU_D"][:, 1]
# r_zC_leg_r = com_r_leg[:, 1] - markers_m["GENOU_D"][:, 2]
#
# acc_com_leg_y_r = derivee(com_r_leg[:, 0], dt, 2)
# acc_com_leg_z_r = derivee(com_r_leg[:, 1], dt, 2)
#
# # Réaction proximale au genou (sur la jambe droite)
# R_K_y_r = m_leg_r * acc_com_leg_y_r - Fdist_y_r
# R_K_z_r = m_leg_r * acc_com_leg_z_r + m_leg_r * g - Fdist_z_r
#
# # Bras de levier genou → cheville droit
# r_yA_leg_r = markers_m["CHEVILLE_D"][:, 1] - markers_m["GENOU_D"][:, 1]
# r_zA_leg_r = markers_m["CHEVILLE_D"][:, 2] - markers_m["GENOU_D"][:, 2]
#
# term_rot_leg_r   = Ic_leg_r * alpha_r_leg
# term_trans_leg_r = m_leg_r * (r_yC_leg_r * acc_com_leg_z_r - r_zC_leg_r * acc_com_leg_y_r)
# term_grav_leg_r  = m_leg_r * g * r_yC_leg_r
#
# M_dist_leg_r = -M_ankle_right_full
# moment_distal_force_r = -(r_yA_leg_r * Fdist_z_r - r_zA_leg_r * Fdist_y_r)
#
# M_knee_right = term_rot_leg_r + term_trans_leg_r + term_grav_leg_r + moment_distal_force_r + M_dist_leg_r
# M_knee_right_norm = M_knee_right / body_mass
#
# # ---------- Thigh ----------
#
# m_thigh_r  = anthropo["Thigh"]["mass"]
# k_thigh_r  = anthropo["Thigh"]["radius_gyr"]
# Ic_thigh_r = m_thigh_r * k_thigh_r**2
#
# # Force distale au genou (sur la cuisse droite)
# FdistK_y_r = -R_K_y_r
# FdistK_z_r = -R_K_z_r
#
# # Cinématique cuisse droite (hanche → genou)
# right_thigh_m = markers_m["GENOU_D"][:, 1:] - markers_m["HANCHE_D"][:, 1:]
# theta_r_thigh = np.arctan2(right_thigh_m[:, 1], right_thigh_m[:, 0])
# alpha_r_thigh = derivee(theta_r_thigh, dt, 2)
#
# com_r_thigh = markers_m["HANCHE_D"][:, 1:] + com_frac["Thigh"] * right_thigh_m
#
# r_yC_th_r = com_r_thigh[:, 0] - markers_m["HANCHE_D"][:, 1]
# r_zC_th_r = com_r_thigh[:, 1] - markers_m["HANCHE_D"][:, 2]
#
# acc_com_th_y_r = derivee(com_r_thigh[:, 0], dt, 2)
# acc_com_th_z_r = derivee(com_r_thigh[:, 1], dt, 2)
#
# # Bras de levier hanche -> genou droit
# r_yK_th_r = markers_m["GENOU_D"][:, 1] - markers_m["HANCHE_D"][:, 1]
# r_zK_th_r = markers_m["GENOU_D"][:, 2] - markers_m["HANCHE_D"][:, 2]
#
# term_rot_th_r   = Ic_thigh_r * alpha_r_thigh
# term_trans_th_r = m_thigh_r * (r_yC_th_r * acc_com_th_z_r - r_zC_th_r * acc_com_th_y_r)
# term_grav_th_r  = m_thigh_r * g * r_yC_th_r
#
# moment_distal_force_th_r = -(r_yK_th_r * FdistK_z_r - r_zK_th_r * FdistK_y_r)
# M_dist_thigh_r = -M_knee_right
#
# M_hip_right = term_rot_th_r + term_trans_th_r + term_grav_th_r + moment_distal_force_th_r + M_dist_thigh_r
# M_hip_right_norm = M_hip_right / body_mass
#
# # =========================================================
# # 1) Définir les cycles gauche et droit (en % du cycle)
# # =========================================================
#
# # Gauche
# start_L, end_L = events["left"]["HS"]["frame"][0], events["left"]["HS"]["frame"][1]
# M_cycle_ankle_L = M_ankle_left_full_norm[start_L:end_L+1]
# M_cycle_knee_L  = M_knee_left_norm[start_L:end_L+1]
# M_cycle_hip_L   = M_hip_left_norm[start_L:end_L+1]
# gc_L = np.linspace(0, 100, len(M_cycle_ankle_L))
#
# # Droite
# start_R, end_R = events["left"]["HS"]["frame"][0], events["left"]["HS"]["frame"][1]
# M_cycle_ankle_R = M_ankle_right_full_norm[start_R:end_R+1]
# M_cycle_knee_R  = M_knee_right_norm[start_R:end_R+1]
# M_cycle_hip_R   = M_hip_right_norm[start_R:end_R+1]
# gc_R = np.linspace(0, 100, len(M_cycle_ankle_R))
#
# # =========================================================
# # PLOT
# # =========================================================
#
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#
# titles       = ["Ankle", "Knee", "Hip"]
# left_moments  = [M_cycle_ankle_L, M_cycle_knee_L, M_cycle_hip_L]
# right_moments = [M_cycle_ankle_R, M_cycle_knee_R, M_cycle_hip_R]
#
# for ax, title, M_L, M_R in zip(axes, titles, left_moments, right_moments):
#     ax.plot(gc_L, M_L, label="Left",  linewidth=2)
#     ax.plot(gc_R, M_R, label="Right", linewidth=2)
#     ax.axhline(0, color="black", linewidth=1)
#     ax.set_title(title, fontsize=10, fontweight="bold")
#     ax.set_xlabel("Gait cycle (%)")
#     ax.set_ylabel("Moment (Nm/kg)")
#     ax.set_xlim(0, 100)
#     ax.grid(True)
#
# axes[0].legend(loc="upper right")
#
# plt.tight_layout()
# plt.savefig(str(output_dir / "joint_moments.png"), dpi=300)
# plt.close()
