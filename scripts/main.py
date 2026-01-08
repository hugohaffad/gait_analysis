import numpy as np
import matplotlib.pyplot as plt
import moveck_bridge_btk as btk
from pathlib import Path
from config import HEA, IMP, REP

output_dir = REP
output_dir.mkdir(parents=True, exist_ok=True)

JOINTS = ["Hip", "Knee", "Ankle"]

# ─────────────────────────────────────────────────────────────────────────
# Paramètres anthropométriques
# ─────────────────────────────────────────────────────────────────────────
BODY_MASS = 70.0
BODY_HEIGHT = 1.84
GRAVITY = 9.806

# Fractions de la longueur corporelle
LENGTH_FRAC = {
    "Foot": 0.152,
    "Leg": 0.246,
    "Thigh": 0.18
}

# Fractions de la masse corporelle
MASS_FRAC = {
    "Foot": 0.0145,
    "Leg": 0.0465,
    "Thigh": 0.10
}

# Position du centre de masse (fraction de la longueur du segment)
COM_FRAC = {
    "Foot": 0.50,
    "Leg": 0.433,
    "Thigh": 0.433
}

# Rayon de giration (fraction de la longueur du segment)
RADIUS_GYR_FRAC = {
    "Foot": 0.475,
    "Leg": 0.302,
    "Thigh": 0.323
}

# Calcul des paramètres anthropométriques
ANTHROPO = {}
for segment in ["Foot", "Leg", "Thigh"]:
    length = BODY_HEIGHT * LENGTH_FRAC[segment]
    mass = BODY_MASS * MASS_FRAC[segment]
    com_pos = length * COM_FRAC[segment]
    radius_gyr = length * RADIUS_GYR_FRAC[segment]

    ANTHROPO[segment] = {
        "length": length,
        "mass": mass,
        "com": com_pos,
        "radius_gyr": radius_gyr,
        "inertia": mass * radius_gyr ** 2  # Moment d'inertie
    }

# ═══════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════

def compute_derivative(signal, dt, order=1):
    """
    Calcule la dérivée numérique d'un signal

    Paramètres :
        signal : array - Signal à dériver
        dt : float - Pas de temps
        order : int - Ordre de la dérivée (1=vitesse, 2=accélération)
    """
    result = signal.copy()
    for _ in range(order):
        result = np.gradient(result, dt)
    return result

def normalize_cycle(signal, start_frame, end_frame, n_points=101):
    """Normalise un cycle sur 0-100% avec n_points échantillons"""
    cycle = signal[start_frame:end_frame + 1]
    x_old = np.linspace(0, 100, len(cycle))
    x_new = np.linspace(0, 100, n_points)
    return np.linspace(0, 100, n_points), np.interp(x_new, x_old, cycle)

def sort_events(events):
    """Trie chronologiquement les événements par frame"""
    for side in ["left", "right"]:
        for evt in ["HS", "TO"]:
            if len(events[side][evt]["frame"]) > 0:
                frames = events[side][evt]["frame"]
                times = events[side][evt]["time"]
                order = np.argsort(frames)
                events[side][evt]["frame"] = [frames[i] for i in order]
                events[side][evt]["time"] = [times[i] for i in order]
    return events

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSE CINÉMATIQUE
# ═══════════════════════════════════════════════════════════════════════════

def kinematics_analysis(file, joints=None):
    if joints is None:
        joints = JOINTS

    h = btk.btkReadAcquisition(str(file))
    markers, _ = btk.btkGetMarkers(h)
    freq = btk.btkGetPointFrequency(h)

    forceplates, _ = btk.btkGetForcePlatforms(h)
    grw = btk.btkGetGroundReactionWrenches(h)
    fp_freq = btk.btkGetAnalogFrequency(h)

    # Vecteurs segmentaires
    segments = {
        "left": {
            "pelvis": markers["EIAS_G"] - markers["HANCHE_G"],
            "thigh": markers["GENOU_G"] - markers["HANCHE_G"],
            "leg": markers["GENOU_G"] - markers["CHEVILLE_G"],
            "foot": markers["PIED_G"] - markers["CHEVILLE_G"]
        },
        "right": {
            "pelvis": markers["EIAS_D"] - markers["HANCHE_D"],
            "thigh": markers["GENOU_D"] - markers["HANCHE_D"],
            "leg": markers["GENOU_D"] - markers["CHEVILLE_D"],
            "foot": markers["PIED_D"] - markers["CHEVILLE_D"]
        }
    }

    # Calcul des angles
    def segment_angle(v):
        """Angle d'un vecteur dans le plan sagittal (Y, Z)"""
        theta = np.arctan2(v[:, 1], v[:, 2])
        theta = np.unwrap(theta, period=2 * np.pi)
        return np.degrees(theta)

    angles_raw = {}
    for side in ["left", "right"]:
        hip = (180 - (segment_angle(segments[side]["thigh"]) -
                      segment_angle(segments[side]["pelvis"]))) % 360
        hip = np.where(hip > 180, hip - 360, hip)

        knee = (180 - (segment_angle(segments[side]["thigh"]) -
                       segment_angle(segments[side]["leg"]))) % 360
        knee = np.where(knee > 180, knee - 360, knee)

        ankle = 90 - (segment_angle(segments[side]["foot"]) -
                      segment_angle(segments[side]["leg"]))
        ankle = np.where(ankle > 180, ankle - 360, ankle)

        angles_raw[side] = {"Hip": hip, "Knee": knee, "Ankle": ankle}

    # Détection des événements
    events = {
        "left": {"HS": {"frame": [], "time": []}, "TO": {"frame": [], "time": []}},
        "right": {"HS": {"frame": [], "time": []}, "TO": {"frame": [], "time": []}}
    }

    for i, _ in enumerate(forceplates):
        GRFz = grw[i]["F"][:, 2]

        if np.count_nonzero(GRFz) == 0:
            continue

        contact_idx = np.where(GRFz > 5)[0]
        if len(contact_idx) == 0:
            continue

        HS_idx = contact_idx[0]
        TO_idx = contact_idx[-1]

        HS_time = HS_idx / fp_freq
        TO_time = TO_idx / fp_freq

        HS_frame = int(np.round(HS_time * freq))
        TO_frame = int(np.round(TO_time * freq))

        side = "right" if markers["CHEVILLE_D"][HS_frame, 2] < markers["CHEVILLE_G"][HS_frame, 2] else "left"

        events[side]["HS"]["frame"].append(HS_frame)
        events[side]["HS"]["time"].append(HS_time)
        events[side]["TO"]["frame"].append(TO_frame)
        events[side]["TO"]["time"].append(TO_time)

    events = sort_events(events)

    if len(events["left"]["HS"]["frame"]) < 2 or len(events["right"]["HS"]["frame"]) < 2:
        raise ValueError(f"Pas assez de HS dans {Path(file).name}")

    # Normalisation des cycles
    angles_normalized = {"left": {}, "right": {}}
    gc = None

    for side in ["left", "right"]:
        start = events[side]["HS"]["frame"][0]
        end = events[side]["HS"]["frame"][1]

        for joint in joints:
            gc_tmp, normalized = normalize_cycle(angles_raw[side][joint], start, end)
            angles_normalized[side][joint] = normalized
            if gc is None:
                gc = gc_tmp

    return {"gc": gc, "left": angles_normalized["left"], "right": angles_normalized["right"]}

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSE CINÉTIQUE
# ═══════════════════════════════════════════════════════════════════════════
def inverse_dynamics(file, joints=None):
    """
    Dynamique inverse : calcul des moments articulaires (Nm/kg)
    Retourne :
        dict avec "gc" (0-100%), "left" et "right" (moments normalisés)
    """
    if joints is None:
        joints = JOINTS

    h = btk.btkReadAcquisition(str(file))
    markers, _ = btk.btkGetMarkers(h)
    freq = btk.btkGetPointFrequency(h)
    dt = 1.0 / freq

    # Conversion en mètres
    markers_m = {name: arr / 1000.0 for name, arr in markers.items()}

    forceplates, _ = btk.btkGetForcePlatforms(h)
    grw = btk.btkGetGroundReactionWrenches(h)
    fp_freq = btk.btkGetAnalogFrequency(h)

    n_frames = btk.btkGetPointFrameNumber(h)
    time = np.arange(n_frames) / freq
    fp_time = np.arange(grw[0]["F"].shape[0]) / fp_freq

    # Détection des événements
    events = {
        "left": {"HS": {"frame": [], "time": []}, "TO": {"frame": [], "time": []}},
        "right": {"HS": {"frame": [], "time": []}, "TO": {"frame": [], "time": []}}
    }

    for i, _ in enumerate(forceplates):
        GRFz = grw[i]["F"][:, 2]

        if np.count_nonzero(GRFz) == 0:
            continue

        contact_idx = np.where(GRFz > 5)[0]
        if len(contact_idx) == 0:
            continue

        HS_idx = contact_idx[0]
        HS_time = HS_idx / fp_freq
        HS_frame = int(np.round(HS_time * freq))

        side = "right" if markers["CHEVILLE_D"][HS_frame, 2] < markers["CHEVILLE_G"][HS_frame, 2] else "left"

        events[side]["HS"]["frame"].append(HS_frame)
        events[side]["HS"]["time"].append(HS_time)

    events = sort_events(events)

    if len(events["left"]["HS"]["frame"]) < 2 or len(events["right"]["HS"]["frame"]) < 2:
        raise ValueError(f"Pas assez de HS dans {Path(file).name}")

    # Calcul des moments
    moments = {}

    for side_idx, (side, fp_idx) in enumerate([("right", 0), ("left", 1)]):

        # Sélection des marqueurs
        if side == "left":
            m_ankle = markers_m["CHEVILLE_G"]
            m_knee = markers_m["GENOU_G"]
            m_hip = markers_m["HANCHE_G"]
            m_foot = markers_m["PIED_G"]
        else:
            m_ankle = markers_m["CHEVILLE_D"]
            m_knee = markers_m["GENOU_D"]
            m_hip = markers_m["HANCHE_D"]
            m_foot = markers_m["PIED_D"]

        # Interpolation des GRF et CoP (indices 1=Y, 2=Z)
        cop_y = np.interp(time, fp_time, grw[fp_idx]["P"][:, 1]) / 1000.0
        cop_z = np.interp(time, fp_time, grw[fp_idx]["P"][:, 2]) / 1000.0
        Fy = np.interp(time, fp_time, grw[fp_idx]["F"][:, 1])
        Fz = np.interp(time, fp_time, grw[fp_idx]["F"][:, 2])

        # ========== PIED (calcul à la cheville) ==========
        # Vecteur du segment pied (cheville -> marqueur pied) dans le plan YZ
        foot_vec = m_foot[:, 1:] - m_ankle[:, 1:]  # [Y, Z]
        theta_foot = np.arctan2(foot_vec[:, 1], foot_vec[:, 0])
        alpha_foot = compute_derivative(theta_foot, dt, order=2)

        # Centre de masse du pied
        com_foot = m_ankle[:, 1:] + COM_FRAC["Foot"] * foot_vec

        # Accélérations du CoM du pied
        acc_com_foot_y = compute_derivative(com_foot[:, 0], dt, order=2)
        acc_com_foot_z = compute_derivative(com_foot[:, 1], dt, order=2)

        # Bras de levier cheville -> CoM du pied
        r_y_CoM = com_foot[:, 0] - m_ankle[:, 1]
        r_z_CoM = com_foot[:, 1] - m_ankle[:, 2]

        # Bras de levier cheville -> CoP
        r_y_CoP = cop_y - m_ankle[:, 1]
        r_z_CoP = cop_z - m_ankle[:, 2]

        # Équation de Newton : R_ankle + GRF - m*g = m*a
        # En Z : R_z + Fz - m*g = m*a_z  =>  R_z = m*a_z + m*g - Fz
        R_ankle_y = ANTHROPO["Foot"]["mass"] * acc_com_foot_y - Fy
        R_ankle_z = ANTHROPO["Foot"]["mass"] * acc_com_foot_z + ANTHROPO["Foot"]["mass"] * GRAVITY - Fz

        # Moment à la cheville autour de X : M_x = r_y * F_z - r_z * F_y
        # ΣM = I*α
        # M_ankle + M_GRF + M_weight + M_inertia = I*α
        M_ankle = (
            ANTHROPO["Foot"]["inertia"] * alpha_foot -
            ANTHROPO["Foot"]["mass"] * (r_y_CoM * acc_com_foot_z - r_z_CoM * acc_com_foot_y) -
            ANTHROPO["Foot"]["mass"] * GRAVITY * r_y_CoM +
            (r_y_CoP * Fz - r_z_CoP * Fy)
        ) / BODY_MASS

        # ========== JAMBE (calcul au genou) ==========
        # Vecteur du segment jambe (genou -> cheville)
        leg_vec = m_ankle[:, 1:] - m_knee[:, 1:]
        theta_leg = np.arctan2(leg_vec[:, 1], leg_vec[:, 0])
        alpha_leg = compute_derivative(theta_leg, dt, order=2)

        # Centre de masse de la jambe
        com_leg = m_knee[:, 1:] + COM_FRAC["Leg"] * leg_vec

        # Accélérations du CoM de la jambe
        acc_com_leg_y = compute_derivative(com_leg[:, 0], dt, order=2)
        acc_com_leg_z = compute_derivative(com_leg[:, 1], dt, order=2)

        # Bras de levier genou -> CoM de la jambe
        r_y_CoM_leg = com_leg[:, 0] - m_knee[:, 1]
        r_z_CoM_leg = com_leg[:, 1] - m_knee[:, 2]

        # Bras de levier genou -> cheville
        r_y_ankle = m_ankle[:, 1] - m_knee[:, 1]
        r_z_ankle = m_ankle[:, 2] - m_knee[:, 2]

        # Forces distales (action-réaction)
        F_dist_y = -R_ankle_y
        F_dist_z = -R_ankle_z

        # Forces de réaction au genou
        R_knee_y = ANTHROPO["Leg"]["mass"] * acc_com_leg_y - F_dist_y
        R_knee_z = ANTHROPO["Leg"]["mass"] * acc_com_leg_z + ANTHROPO["Leg"]["mass"] * GRAVITY - F_dist_z

        # Moment au genou autour de X
        M_knee = (
            ANTHROPO["Leg"]["inertia"] * alpha_leg -
            ANTHROPO["Leg"]["mass"] * (r_y_CoM_leg * acc_com_leg_z - r_z_CoM_leg * acc_com_leg_y) -
            ANTHROPO["Leg"]["mass"] * GRAVITY * r_y_CoM_leg +
            (r_y_ankle * F_dist_z - r_z_ankle * F_dist_y) +
            M_ankle * BODY_MASS
        ) / BODY_MASS

        # ========== CUISSE (calcul à la hanche) ==========
        # Vecteur du segment cuisse (hanche -> genou)
        thigh_vec = m_knee[:, 1:] - m_hip[:, 1:]
        theta_thigh = np.arctan2(thigh_vec[:, 1], thigh_vec[:, 0])
        alpha_thigh = compute_derivative(theta_thigh, dt, order=2)

        # Centre de masse de la cuisse
        com_thigh = m_hip[:, 1:] + COM_FRAC["Thigh"] * thigh_vec

        # Accélérations du CoM de la cuisse
        acc_com_thigh_y = compute_derivative(com_thigh[:, 0], dt, order=2)
        acc_com_thigh_z = compute_derivative(com_thigh[:, 1], dt, order=2)

        # Bras de levier hanche -> CoM de la cuisse
        r_y_CoM_thigh = com_thigh[:, 0] - m_hip[:, 1]
        r_z_CoM_thigh = com_thigh[:, 1] - m_hip[:, 2]

        # Bras de levier hanche -> genou
        r_y_knee = m_knee[:, 1] - m_hip[:, 1]
        r_z_knee = m_knee[:, 2] - m_hip[:, 2]

        # Forces distales (action-réaction)
        F_dist_knee_y = -R_knee_y
        F_dist_knee_z = -R_knee_z

        # Forces de réaction à la hanche
        R_hip_y = ANTHROPO["Thigh"]["mass"] * acc_com_thigh_y - F_dist_knee_y
        R_hip_z = ANTHROPO["Thigh"]["mass"] * acc_com_thigh_z + ANTHROPO["Thigh"]["mass"] * GRAVITY - F_dist_knee_z

        # Moment à la hanche autour de X
        M_hip = (
            ANTHROPO["Thigh"]["inertia"] * alpha_thigh -
            ANTHROPO["Thigh"]["mass"] * (r_y_CoM_thigh * acc_com_thigh_z - r_z_CoM_thigh * acc_com_thigh_y) -
            ANTHROPO["Thigh"]["mass"] * GRAVITY * r_y_CoM_thigh +
            (r_y_knee * F_dist_knee_z - r_z_knee * F_dist_knee_y) +
            M_knee * BODY_MASS
        ) / BODY_MASS

        moments[side] = {"Ankle": M_ankle, "Knee": M_knee, "Hip": M_hip}

    # Normalisation des cycles
    moments_normalized = {"left": {}, "right": {}}
    gc = None

    for side in ["left", "right"]:
        start = events[side]["HS"]["frame"][0]
        end = events[side]["HS"]["frame"][1]

        for joint in joints:
            gc_tmp, normalized = normalize_cycle(moments[side][joint], start, end)
            moments_normalized[side][joint] = normalized
            if gc is None:
                gc = gc_tmp

    return {"gc": gc, "left": moments_normalized["left"], "right": moments_normalized["right"]}

# ═══════════════════════════════════════════════════════════════════════════
# TRAITEMENT DES GROUPES
# ═══════════════════════════════════════════════════════════════════════════

def analyze_group(folder_path, analysis_func, joints=None):
    """
    Applique une fonction d'analyse (cinématique ou cinétique) à tous les C3D

    Paramètres :
        folder_path : Path - Dossier contenant les fichiers C3D
        analysis_func : function - kinematics_analysis ou inverse_dynamics
        joints : list - Articulations à analyser
    """
    if joints is None:
        joints = JOINTS

    curves = {joint: {"left": [], "right": []} for joint in joints}
    gc = None

    c3d_files = sorted(folder_path.glob("*.c3d"))

    for filepath in c3d_files:
        result = analysis_func(filepath, joints=joints)

        if gc is None:
            gc = result["gc"]

        for joint in joints:
            curves[joint]["left"].append(result["left"][joint])
            curves[joint]["right"].append(result["right"][joint])

    return gc, curves

# ═══════════════════════════════════════════════════════════════════════════
# EXÉCUTION DES ANALYSES
# ═══════════════════════════════════════════════════════════════════════════

# Cinématique
gc_kin, angles_healthy = analyze_group(HEA, kinematics_analysis, JOINTS)
_, angles_impaired = analyze_group(IMP, kinematics_analysis, JOINTS)

# Calcul des asymétries
asymmetry_healthy = {}
asymmetry_impaired = {}

for joint in JOINTS:
    data_R = np.array(angles_healthy[joint]["right"])
    data_L = np.array(angles_healthy[joint]["left"])
    asymmetry_healthy[joint] = data_R - data_L

    data_R = np.array(angles_impaired[joint]["right"])
    data_L = np.array(angles_impaired[joint]["left"])
    asymmetry_impaired[joint] = data_R - data_L

# Cinétique
gc_kin_moments, moments_healthy = analyze_group(HEA, inverse_dynamics, JOINTS)
_, moments_impaired = analyze_group(IMP, inverse_dynamics, JOINTS)

# Asymétrie
diff_imp_vs_hea = {}
for joint in JOINTS:
    hea_R = np.array(angles_healthy[joint]["right"]).mean(axis=0)
    hea_L = np.array(angles_healthy[joint]["left"]).mean(axis=0)
    imp_R = np.array(angles_impaired[joint]["right"]).mean(axis=0)
    imp_L = np.array(angles_impaired[joint]["left"]).mean(axis=0)

    diff_imp_vs_hea[joint] = {
        "right": imp_R - hea_R,
        "left": imp_L - hea_L
    }

# ═══════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Angles articulaires
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, joint in enumerate(JOINTS):
    ax = axes[i]

    for side, color, label in [("right", "tab:blue", "Right"),
                               ("left", "tab:orange", "Left")]:
        data = np.array(angles_healthy[joint][side])
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        ax.plot(gc_kin, mean, color=color, linewidth=2, label=label)
        ax.fill_between(gc_kin, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

        data_imp = np.array(angles_impaired[joint][side])
        ax.plot(gc_kin, data_imp.T, color=color, linestyle="--", linewidth=1, alpha=0.8)

    ax.set_title(f"{joint}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.7)

    OT = 13
    TO = 62

    ax.axvline(OT, color="k", linewidth=1)
    ax.axvline(TO, color="k", linewidth=1)

    ax.text(OT, ax.get_ylim()[1]*0.95, "OT",
            ha="center", va="top", fontsize=9)
    ax.text(TO, ax.get_ylim()[1]*0.95, "TO",
            ha="center", va="top", fontsize=9)

axes[0].set_ylabel("Angle (°)", fontsize=11)
axes[0].set_ylim(-20, 40)
axes[1].set_ylim(0, 80)
axes[2].set_ylim(-50, 0)
axes[0].legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "joint_angles.png", dpi=300, bbox_inches='tight')
plt.close()

# Asymétrie (angles)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, joint in enumerate(JOINTS):
    ax = axes[i]

    # Sain (moyenne ± SD)
    data_hea = asymmetry_healthy[joint]
    mean_hea = data_hea.mean(axis=0)
    std_hea = data_hea.std(axis=0, ddof=1) if len(data_hea) > 1 else np.zeros(101)

    ax.plot(gc_kin, mean_hea, color="tab:blue", linewidth=2, label="Healthy")
    ax.fill_between(gc_kin, mean_hea - std_hea, mean_hea + std_hea, color="tab:blue", alpha=0.2, linewidth=0)

    # Altéré (essais individuels)
    data_imp = asymmetry_impaired[joint]
    ax.plot(gc_kin, data_imp.T, color="k", linestyle="--", linewidth=1, alpha=0.8, label="Impaired")

    ax.axhline(0, color="k", linestyle=":", linewidth=1.5)
    ax.set_title(f"{joint}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)", fontsize=11)
    ax.set_ylabel("Δ angle (Right - Left) (°)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(-30, 30)
    ax.grid(True, alpha=0.7)

    OT = 13
    TO = 62

    ax.axvline(OT, color="k", linewidth=1)
    ax.axvline(TO, color="k", linewidth=1)

    ax.text(OT, ax.get_ylim()[1]*0.95, "OT",
            ha="center", va="top", fontsize=9)
    ax.text(TO, ax.get_ylim()[1]*0.95, "TO",
            ha="center", va="top", fontsize=9)

axes[0].legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(output_dir / "intra_group_asymmetry.png", dpi=300, bbox_inches='tight')
plt.close()

# Différence Altéré - Sain
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, joint in enumerate(JOINTS):
    ax = axes[i]

    diff_R = diff_imp_vs_hea[joint]["right"]
    diff_L = diff_imp_vs_hea[joint]["left"]

    ax.plot(gc_kin, diff_R, color="tab:blue", linewidth=2, label="Right")
    ax.plot(gc_kin, diff_L, color="tab:orange", linewidth=2, label="Left")

    ax.axhline(0, color="k", linestyle=":", linewidth=1.5)
    ax.set_title(f"{joint}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)", fontsize=11)
    ax.set_ylabel("Δ angle (Impaired - Healthy) (°)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.7)

    OT = 13
    TO = 62

    ax.axvline(OT, color="k", linewidth=1)
    ax.axvline(TO, color="k", linewidth=1)

    ax.text(OT, ax.get_ylim()[1]*0.95, "OT",
            ha="center", va="top", fontsize=9)
    ax.text(TO, ax.get_ylim()[1]*0.95, "TO",
            ha="center", va="top", fontsize=9)

axes[0].legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(output_dir / "delta_angles_imp-hea.png", dpi=300, bbox_inches='tight')
plt.close()

# Moments articulaires
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, joint in enumerate(JOINTS):
    ax = axes[i]

    for side, color, label in [("right", "tab:blue", "Right"),
                               ("left", "tab:orange", "Left")]:

        data_hea = np.array(moments_healthy[joint][side])
        mean_hea = data_hea.mean(axis=0)
        std_hea = data_hea.std(axis=0)

        ax.plot(gc_kin_moments, mean_hea, color=color, linewidth=2, label=label)
        ax.fill_between(gc_kin_moments, mean_hea - std_hea, mean_hea + std_hea,
                        color=color, alpha=0.2, linewidth=0)

        data_imp = np.array(moments_impaired[joint][side])
        ax.plot(gc_kin_moments, data_imp.T, color=color, linestyle="--",
                linewidth=1, alpha=0.8)

    ax.set_title(f"{joint}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.7)

    OT = 13
    TO = 62

    ax.axvline(OT, color="k", linewidth=1)
    ax.axvline(TO, color="k", linewidth=1)

    ax.text(OT, ax.get_ylim()[1]*0.95, "OT",
            ha="center", va="top", fontsize=9)
    ax.text(TO, ax.get_ylim()[1]*0.95, "TO",
            ha="center", va="top", fontsize=9)

axes[0].set_ylabel("Moment (Nm/kg)", fontsize=11)
axes[0].legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "joint_moments.png", dpi=300, bbox_inches='tight')
plt.close()

# # ═══════════════════════════════════════════════════════════════════════════
# # STATISTIQUES
# # ═══════════════════════════════════════════════════════════════════════════
#
# print("\n" + "=" * 70)
# print("📊 STATISTIQUES")
# print("=" * 70)
#
# print("\n1. ASYMÉTRIE ANGULAIRE (amplitude moyenne |R-L|)")
# print("-" * 70)
# print(f"{'Articulation':<12} {'Sain (M±SD)':<20} {'Altéré (M±SD)':<20} {'Δ':<10}")
# print("-" * 70)
#
# for joint in JOINTS:
#     amp_hea = np.abs(asymmetry_healthy[joint]).mean(axis=1)
#     mean_hea = amp_hea.mean()
#     sd_hea = amp_hea.std(ddof=1) if len(amp_hea) > 1 else 0.0
#
#     amp_imp = np.abs(asymmetry_impaired[joint]).mean(axis=1)
#     mean_imp = amp_imp.mean()
#     sd_imp = amp_imp.std(ddof=1) if len(amp_imp) > 1 else 0.0
#
#     delta = mean_imp - mean_hea
#
#     print(f"{joint:<12} {mean_hea:5.2f} ± {sd_hea:4.2f}°      "
#           f"{mean_imp:5.2f} ± {sd_imp:4.2f}°      "
#           f"{delta:+5.2f}°")
#
# print("\n2. MOMENTS ARTICULAIRES (pic moyen)")
# print("-" * 70)
# print(f"{'Articulation':<12} {'Sain (M±SD)':<25} {'Altéré (M±SD)':<25}")
# print("-" * 70)
#
# for joint in JOINTS:
#     # Pic moyen pour chaque essai (moyenne des deux côtés)
#     data_hea_r = np.array(moments_healthy[joint]["right"])
#     data_hea_l = np.array(moments_healthy[joint]["left"])
#     peak_hea = (np.abs(data_hea_r).max(axis=1) + np.abs(data_hea_l).max(axis=1)) / 2
#
#     data_imp_r = np.array(moments_impaired[joint]["right"])
#     data_imp_l = np.array(moments_impaired[joint]["left"])
#     peak_imp = (np.abs(data_imp_r).max(axis=1) + np.abs(data_imp_l).max(axis=1)) / 2
#
#     mean_hea = peak_hea.mean()
#     sd_hea = peak_hea.std(ddof=1) if len(peak_hea) > 1 else 0.0
#
#     mean_imp = peak_imp.mean()
#     sd_imp = peak_imp.std(ddof=1) if len(peak_imp) > 1 else 0.0
#
#     print(f"{joint:<12} {mean_hea:5.2f} ± {sd_hea:4.2f} Nm/kg     "
#           f"{mean_imp:5.2f} ± {sd_imp:4.2f} Nm/kg")