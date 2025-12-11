from config import *

import numpy as np
import matplotlib.pyplot as plt
import moveck_bridge_btk as btk
from scipy.stats import wilcoxon, shapiro

output_dir = REP
output_dir.mkdir(parents=True, exist_ok=True)

joints = ["Hip", "Knee", "Ankle"]

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

body_mass   = 70.0   # à adapter si tu as la masse réelle
body_height = 1.84   # à adapter si tu as la taille réelle
g  = 9.806

anthropo = {}
for seg in ["Foot", "Leg", "Thigh"]:
    L = body_height * len_frac[seg]
    m = body_mass   * mass_frac[seg]
    com = L * com_frac[seg]
    rg  = L * rg_frac[seg]
    anthropo[seg] = {
        "len": L,
        "mass": m,
        "com": com,
        "radius_gyr": rg
    }

# =====================================================================
# 1. FONCTIONS DE BAS NIVEAU : TRAITEMENT DES DONNÉES BRUTES
# =====================================================================

def compute_derivative(y, dt, ordre):
    """
    Calcule la dérivée numérique d'ordre 'ordre'
    en supposant un pas de temps constant dt.
    """
    for _ in range(ordre):
        y = np.gradient(y, dt)
    return y

def sort_events(events):
    """
    Trie chronologiquement les évènements HS et TO pour chaque côté.
    """
    for side in events:
        for evt in ["HS", "TO"]:
            frames = np.array(events[side][evt]["frame"])
            times  = np.array(events[side][evt]["time"])
            order = np.argsort(frames)
            events[side][evt]["frame"] = frames[order].tolist()
            events[side][evt]["time"]  = times[order].tolist()
    return events


def normalize_cycle(signal, start_frame, end_frame, n_points=101):
    """
    Extrait un cycle entre start_frame et end_frame (inclus)
    et le normalise sur 0–100 % avec n_points échantillons.
    """
    cycle = signal[start_frame:end_frame+1]
    n = len(cycle)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)

    cycle_norm = np.interp(x_new, x_old, cycle)
    return x_new, cycle_norm


def angle(v):
    """
    Calcule l'angle du vecteur v dans le plan (Y, Z) en degrés,
    avec déroulement de phase (unwrap).
    v : array (n_frames, 3).
    """
    theta = np.degrees(
        np.unwrap(
            np.arctan2(v[:, 1], v[:, 2]),
            period=2 * np.pi
        )
    )
    return theta


def compute_integral(curves_by_joint):
    """
    curves_by_joint[joint] = courbe(t)
    Retourne, pour chaque articulation, l’aire sous |courbe(t)|
    (mesure globale de magnitude sur le cycle).
    """
    integral = {}
    for joint, curve in curves_by_joint.items():
        integral[joint] = np.sum(np.abs(curve))
    return integral

# =====================================================================
# 2. CINÉMATIQUE : LECTURE .C3D ET NORMALISATION DES CYCLES
# =====================================================================

def kinematics_analysis(file, joints):
    """
    Lit un fichier c3d, calcule les angles Hip/Knee/Ankle (Right & Left),
    détecte un cycle de marche par côté (HS – HS), normalise les courbes
    sur 0–100 % du cycle et renvoie les angles normalisés.
    """

    h = btk.btkReadAcquisition(str(file))

    # ---- Markers ----
    markers, markersInfo = btk.btkGetMarkers(h)
    freq = btk.btkGetPointFrequency(h)

    # Segments (vecteurs articulaires)
    left_pelvis  = markers["EIAS_G"] - markers["HANCHE_G"]
    right_pelvis = markers["EIAS_D"] - markers["HANCHE_D"]
    left_thigh   = markers["GENOU_G"] - markers["HANCHE_G"]
    right_thigh  = markers["GENOU_D"] - markers["HANCHE_D"]
    left_leg     = markers["GENOU_G"] - markers["CHEVILLE_G"]
    right_leg    = markers["GENOU_D"] - markers["CHEVILLE_D"]
    left_foot    = markers["PIED_G"]  - markers["CHEVILLE_G"]
    right_foot   = markers["PIED_D"]  - markers["CHEVILLE_D"]

    # Angles de hanche
    left_hip = (180 - (angle(left_thigh) - angle(left_pelvis))) % 360
    left_hip = np.where(left_hip > 180, left_hip - 360, left_hip)
    right_hip = (180 - (angle(right_thigh) - angle(right_pelvis))) % 360
    right_hip = np.where(right_hip > 180, right_hip - 360, right_hip)

    # Angles de genou
    left_knee = (180 - (angle(left_thigh) - angle(left_leg))) % 360
    left_knee = np.where(left_knee > 180, left_knee - 360, left_knee)
    right_knee = (180 - (angle(right_thigh) - angle(right_leg))) % 360
    right_knee = np.where(right_knee > 180, right_knee - 360, right_knee)

    # Angles de cheville
    left_ankle = 90 - (angle(left_foot) - angle(left_leg))
    left_ankle = np.where(left_ankle > 180, left_ankle - 360, left_ankle)
    right_ankle = 90 - (angle(right_foot) - angle(right_leg))
    right_ankle = np.where(right_ankle > 180, right_ankle - 360, right_ankle)

    angles = {
        "Hip":   {"left": left_hip,   "right": right_hip},
        "Knee":  {"left": left_knee,  "right": right_knee},
        "Ankle": {"left": left_ankle, "right": right_ankle},
    }

    # ---- Forceplates et événements ----
    forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
    grw = btk.btkGetGroundReactionWrenches(h)
    fp_freq = btk.btkGetAnalogFrequency(h)

    events = {
        "left":  {"HS": {"frame": [], "time": []},
                  "TO": {"frame": [], "time": []}},
        "right": {"HS": {"frame": [], "time": []},
                  "TO": {"frame": [], "time": []}},
    }

    # 1) plateformes actives
    active_forceplates = []
    for i in range(len(forceplates)):
        if np.count_nonzero(grw[i]["F"]) != 0:
            active_forceplates.append(i)

    # 2) détection HS / TO sur chaque plateforme
    for i in active_forceplates:
        GRFz = grw[i]["F"][:, 2]

        HS_idx = np.where(GRFz > 5)[0][0]
        TO_idx = np.where(GRFz > 5)[0][-1]

        HS_time = HS_idx / fp_freq
        TO_time = TO_idx / fp_freq

        HS_frame = int(np.round(HS_time * freq))
        TO_frame = int(np.round(TO_time * freq))

        # 3) attribution à la jambe droite ou gauche
        if markers["CHEVILLE_D"][HS_frame, 2] < markers["CHEVILLE_G"][HS_frame, 2]:
            side = "right"
        else:
            side = "left"

        events[side]["HS"]["frame"].append(HS_frame)
        events[side]["HS"]["time"].append(HS_time)
        events[side]["TO"]["frame"].append(TO_frame)
        events[side]["TO"]["time"].append(TO_time)

    events = sort_events(events)

    # besoin d’au moins 2 HS par côté pour définir un cycle
    if len(events["right"]["HS"]["frame"]) < 2 or len(events["left"]["HS"]["frame"]) < 2:
        raise ValueError(
            f"Pas assez de HS dans {file} : "
            f"right={len(events['right']['HS']['frame'])}, "
            f"left={len(events['left']['HS']['frame'])}"
        )

    start_R, end_R = events["right"]["HS"]["frame"][0], events["right"]["HS"]["frame"][1]
    start_L, end_L = events["left"]["HS"]["frame"][0],  events["left"]["HS"]["frame"][1]

    # ---- Normalisation des cycles (0–100 %) ----
    angles_norm = {"right": {}, "left": {}}
    gc = None  # vecteur 0–100 %

    for joint in joints:
        sig_R = angles[joint]["right"]
        gc_tmp, sig_R_norm = normalize_cycle(sig_R, start_R, end_R)

        sig_L = angles[joint]["left"]
        _, sig_L_norm = normalize_cycle(sig_L, start_L, end_L)

        angles_norm["right"][joint] = sig_R_norm
        angles_norm["left"][joint]  = sig_L_norm

        if gc is None:
            gc = gc_tmp

    return {"gc": gc, "right": angles_norm["right"], "left": angles_norm["left"]}


def compute_group_curves(dir_path, joints):
    """
    Parcourt tous les fichiers .c3d d’un dossier (HEA ou IMP),
    applique kinematics_analysis et stocke les courbes normalisées
    dans all_curves[joint][side] (liste des essais).
    """
    all_curves = {joint: {"right": [], "left": []} for joint in joints}
    gc = None

    for file in dir_path.iterdir():
        if file.suffix.lower() != ".c3d":
            continue

        res = kinematics_analysis(file, joints=joints)

        if gc is None:
            gc = res["gc"]

        for joint in joints:
            all_curves[joint]["right"].append(res["right"][joint])
            all_curves[joint]["left"].append(res["left"][joint])

    return gc, all_curves

# =====================================================================
# 3. FONCTIONS DE PLOT : FIGURE 1 (angles) ET FIGURE 2 (asymétrie)
# =====================================================================

def plot_group_curves(gc, all_curves, joints, filename, title=None):
    """
    Figure 1 : courbes moyennes ± 1 SD pour Right et Left
    (profil angulaire) dans une condition donnée.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    for i, joint in enumerate(joints):
        ax = axes[i]
        for side, color, label in [("right", "tab:blue", "Right"),
                                   ("left",  "tab:orange", "Left")]:

            data = np.vstack(all_curves[joint][side])
            mean_curve = data.mean(axis=0)
            std_curve  = data.std(axis=0)

            ax.plot(gc, mean_curve, label=label, linewidth=2)
            ax.fill_between(gc,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            alpha=0.2)

        ax.set_title(joint, fontsize=10, fontweight="bold")
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Angle (°)")
        ax.set_xlim(0, 100)
        ax.grid(True)

    axes[0].set_ylim(-20, 40)
    axes[0].legend(loc="upper right")
    axes[1].set_ylim(0, 80)
    axes[2].set_ylim(-50, 0)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(str(output_dir / filename), dpi=300)
    plt.close()


def plot_asymmetry(gc, asym, joints, filename):
    """
    Figure 2 : asymétrie Right-Left pour une condition (Healthy ou Impaired),
    avec ligne 0° (symétrie parfaite) et moyenne de l’asymétrie (ligne rouge).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    for i, joint in enumerate(joints):
        ax = axes[i]
        curve = asym[joint]
        mean_curve = curve.mean()
        std_curve = curve.std(axis=0)
        ax.fill_between(gc,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2)

        ax.plot(gc, curve, linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(mean_curve, color="k", linestyle="-", linewidth=1.5)

        ax.set_title(joint, fontsize=10, fontweight="bold")
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Asymmetry (°)")
        ax.set_xlim(0, 100)
        ax.set_ylim(-30, 30)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(str(output_dir / filename), dpi=300)
    plt.close()

# =====================================================================
# 4. FONCTIONS « REPORT » : ROM, ASYMÉTRIE, DIFFÉRENCES IMP–HEA
# =====================================================================

def compute_intra_group_asymmetry(curves, joints):
    """
    Asymétrie intra-groupe :
      Asym(joint, t) = Right_mean(t) - Left_mean(t)
    (utilisé pour Figure 2 et les tests d’asymétrie ≠ 0).
    """
    asym = {}
    for joint in joints:
        left_mean  = np.vstack(curves[joint]["left"]).mean(axis=0)
        right_mean = np.vstack(curves[joint]["right"]).mean(axis=0)
        asym[joint] = right_mean - left_mean
    return asym


def compute_inter_group_asymmetry(curves_hea, curves_imp, joints):
    """
    Asymétrie inter-groupe :
      Δ_asym(joint, t) = (R-L)_IMP - (R-L)_HEA
    (utilisé pour la figure Δ_asym et les tests Healthy vs Impaired).
    """
    delta_asym = {}
    for joint in joints:
        left_hea  = np.vstack(curves_hea[joint]["left"]).mean(axis=0)
        right_hea = np.vstack(curves_hea[joint]["right"]).mean(axis=0)
        diff_hea  = right_hea - left_hea

        left_imp  = np.vstack(curves_imp[joint]["left"]).mean(axis=0)
        right_imp = np.vstack(curves_imp[joint]["right"]).mean(axis=0)
        diff_imp  = right_imp - left_imp

        delta_asym[joint] = diff_imp - diff_hea
    return delta_asym


def compute_rom_from_group(curves, joints, label):
    """
    Calcule min, max et ROM de la COURBE MOYENNE (Right et Left)
    pour chaque articulation dans un groupe (Healthy / Impaired).
    Sert à décrire Figure 1.
    """
    print(f"\n--- {label} ---")
    for joint in joints:
        for side in ["right", "left"]:
            data = np.vstack(curves[joint][side])
            mean_curve = data.mean(axis=0)

            angle_min = mean_curve.min()
            angle_max = mean_curve.max()
            rom = angle_max - angle_min

            side_str = "Right" if side == "right" else "Left"
            print(f"{joint:5s} – {side_str:5s} | "
                  f"min = {angle_min:6.2f}°, "
                  f"max = {angle_max:6.2f}°, "
                  f"ROM = {rom:6.2f}°")


def compute_imp_minus_hea(curves_hea, curves_imp, joints):
    """
    Calcule, pour chaque articulation et chaque côté,
    la différence Impaired – Healthy sur 0–100 % du cycle.
    diff[joint]["right"] = IMP_right_mean(t) - HEA_right_mean(t)
    diff[joint]["left"]  = IMP_left_mean(t)  - HEA_left_mean(t)
    Sert à répondre aux questions sur l’effet de la condition
    pour chaque jambe (Q8–Q9).
    """
    diff = {}
    for joint in joints:
        hea_R = np.vstack(curves_hea[joint]["right"]).mean(axis=0)
        hea_L = np.vstack(curves_hea[joint]["left"]).mean(axis=0)
        imp_R = np.vstack(curves_imp[joint]["right"]).mean(axis=0)
        imp_L = np.vstack(curves_imp[joint]["left"]).mean(axis=0)

        diff[joint] = {
            "right": imp_R - hea_R,
            "left":  imp_L - hea_L
        }
    return diff

# =====================================================================
# 5. SCRIPT PRINCIPAL : APPELS, FIGURES ET TESTS STATISTIQUES
# =====================================================================

# 5.1 Courbes par groupe (cinématique normalisée)
gc_hea, curves_hea = compute_group_curves(HEA, joints)
gc_imp, curves_imp = compute_group_curves(IMP, joints)

# 5.2 ROM pour Figure 1
compute_rom_from_group(curves_hea, joints, label="Panel A – Healthy")
compute_rom_from_group(curves_imp, joints, label="Panel B – Impaired")

# 5.3 Asymétrie intra-condition (Figure 2)
asym_hea = compute_intra_group_asymmetry(curves_hea, joints)
asym_imp = compute_intra_group_asymmetry(curves_imp, joints)

plot_group_curves(gc_hea, curves_hea, joints, filename="joint_angles_healthy.png")
plot_group_curves(gc_imp, curves_imp, joints, filename="joint_angles_impaired.png")

plot_asymmetry(gc_hea, asym_hea, joints, filename="asymmetry_healthy.png")
plot_asymmetry(gc_imp, asym_imp, joints, filename="asymmetry_impaired.png")

# 5.4 Δ asymétrie entre conditions (Figure Δ_asym)
delta_asym = compute_inter_group_asymmetry(curves_hea, curves_imp, joints)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()
for i, joint in enumerate(joints):
    ax = axes[i]
    da = delta_asym[joint]
    mean_delta = da.mean()
    ax.plot(gc_hea, da, linewidth=2)
    ax.axhline(mean_delta, color="k", linestyle="-", linewidth=1.5)
    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_title(joint, fontsize=10, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Δ asym (°)")
    ax.set_xlim(0, 100)
    ax.grid(True)
plt.tight_layout()
plt.savefig(str(output_dir / "delta_asymmetry.png"), dpi=300)
plt.close()

# 5.5 Intégrales d’asymétrie (magnitude globale)
integral_asym = compute_integral(delta_asym)
integral_hea  = compute_integral(asym_hea)
integral_imp  = compute_integral(asym_imp)

print("\nIntégrale d’asymétrie (aire sous |Δ_asym|) :")
for joint in joints:
    print(f"{joint}: {integral_asym[joint]:.2f}")

print("\nIntégrale d’asymétrie (Healthy) :")
for joint in joints:
    print(f"{joint}: {integral_hea[joint]:.2f}")

print("\nIntégrale d’asymétrie (Impaired) :")
for joint in joints:
    print(f"{joint}: {integral_imp[joint]:.2f}\n")

# 5.6 Différence d’angle Impaired – Healthy pour chaque jambe
diff_angles = compute_imp_minus_hea(curves_hea, curves_imp, joints)

# 5.6.1 Normalité des Δ angles (Shapiro–Wilk)
normality_results = {}
print("=== Test de Shapiro–Wilk sur Δ angles (Imp - Hea) ===")
for joint in joints:
    normality_results[joint] = {}
    print(f"\n{joint}:")
    for side in ["right", "left"]:
        data = diff_angles[joint][side]
        stat, p_val = shapiro(data)
        normality_results[joint][side] = {"statistic": stat, "p_value": p_val}
        print(f"  {side} : W = {stat:.3f}, p = {p_val:.4f}")

# 5.6.2 Médiane, IQR et test de Wilcoxon vs 0 pour Δ angles
summary_stats = {}
print("\n=== Médiane et IQR des différences d’angle (Imp - Hea) ===\n")
for joint in joints:
    summary_stats[joint] = {}
    print(joint + ":")
    for side in ["right", "left"]:
        data = diff_angles[joint][side]
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        summary_stats[joint][side] = {"median": median, "q1": q1, "q3": q3, "IQR": iqr}
        print(f"  {side}: median = {median:.2f}°, IQR = [{q1:.2f} ; {q3:.2f}]")
    print()

wilcoxon_results = {}
print("=== Test de Wilcoxon sur Δ angles (H0 : médiane = 0) ===\n")
for joint in joints:
    wilcoxon_results[joint] = {}
    print(joint + ":")
    for side in ["right", "left"]:
        data = diff_angles[joint][side]
        stat, p_value = wilcoxon(data)
        wilcoxon_results[joint][side] = {"statistic": stat, "p_value": p_value}
        print(f"  {side}: W = {stat:.1f}, p = {p_value:.5f}")
    print()

# 5.7 Asymétrie ≠ 0 ? (Q4) pour chaque condition
conditions = {"Healthy": asym_hea, "Impaired": asym_imp}
results_asym_vs_0 = {}

print("Pour chaque condition de marche, l’asymétrie droite-gauche est-elle significativement différente de 0 ?")
for cond_name, cond_data in conditions.items():
    results_asym_vs_0[cond_name] = {}
    print(f"\n=== {cond_name} ===")
    for joint in joints:
        asym_curve = cond_data[joint]
        stat, p = wilcoxon(asym_curve)  # H0 : median = 0
        median = np.median(asym_curve)
        iqr = np.percentile(asym_curve, 75) - np.percentile(asym_curve, 25)
        results_asym_vs_0[cond_name][joint] = {"median": median, "IQR": iqr, "W": stat, "p": p}
        print(f"{joint}: median={median:.2f}°, IQR={iqr:.2f}, W={stat:.1f}, p={p:.5f}")

# 5.8 Comparaison d’asymétrie Healthy vs Impaired (Q6)
wilcox_between = {}
print("\n=== Wilcoxon: Healthy vs Impaired asymmetry (H0: median Δ = 0) ===\n")
for joint in joints:
    a_hea = asym_hea[joint]
    a_imp = asym_imp[joint]
    diff = a_imp - a_hea
    W, p = wilcoxon(diff)
    wilcox_between[joint] = {
        "median_diff": np.median(diff),
        "IQR": np.percentile(diff, [25, 75]),
        "W": W,
        "p_value": p
    }
    d = wilcox_between[joint]
    print(f"{joint}: median Δ = {d['median_diff']:.2f}°, "
          f"IQR = [{d['IQR'][0]:.2f} ; {d['IQR'][1]:.2f}], "
          f"W = {d['W']:.1f}, p = {d['p_value']:.5f}")

# =====================================================================
# Inverse dynamics
# =====================================================================

def inverse_dynamics(file, joints):
    """
    Calcule, pour un fichier .c3d :
      - les moments articulaires de cheville, genou, hanche
        pour la JAMBE GAUCHE et la JAMBE DROITE
      - normalisés sur un cycle HS–HS par côté (0–100 %)
      - normalisés par la masse corporelle (Nm/kg)

    Retourne un dict :
    {
      "gc"   : vecteur 0–100 % (101 points),
      "left" : {"Ankle": M_L_ankle_norm, "Knee": ..., "Hip": ...},
      "right": {idem}
    }
    """
    # --- Lecture de l’acquisition ---
    h = btk.btkReadAcquisition(str(file))
    markers, markersInfo = btk.btkGetMarkers(h)
    frame = btk.btkGetPointFrameNumber(h)
    freq  = btk.btkGetPointFrequency(h)
    time  = np.arange(frame) / freq
    dt    = 1.0 / freq

    # Marqueurs en metres
    markers_m = {name: arr / 1000.0 for name, arr in markers.items()}

    # Force plates
    forceplates, fpInfo = btk.btkGetForcePlatforms(h)
    grw = btk.btkGetGroundReactionWrenches(h)
    fp_freq = btk.btkGetAnalogFrequency(h)
    fp_time = np.arange(grw[0]["F"].shape[0]) / fp_freq

    # ------------------------------------------------------------------
    #      DÉTECTION DES ÉVÉNEMENTS (identique à la cinématique)
    # ------------------------------------------------------------------
    events = {
        "left":  {"HS": {"frame": [], "time": []},
                  "TO": {"frame": [], "time": []}},
        "right": {"HS": {"frame": [], "time": []},
                  "TO": {"frame": [], "time": []}},
    }

    active_forceplates = []
    for i in range(len(forceplates)):
        if np.count_nonzero(grw[i]["F"]) == 0:
            continue
        active_forceplates.append(i)

    for i in active_forceplates:
        GRFz = grw[i]["F"][:, 2]

        # Seuil simple pour détection HS / TO
        HS_idx = np.where(GRFz > 5)[0][0]
        TO_idx = np.where(GRFz > 5)[0][-1]

        HS_time = HS_idx / fp_freq
        TO_time = TO_idx / fp_freq

        HS_frame = int(np.round(HS_time * freq))
        TO_frame = int(np.round(TO_time * freq))

        # Attribution de la plate-forme à gauche/droite
        if markers["CHEVILLE_D"][HS_frame, 2] < markers["CHEVILLE_G"][HS_frame, 2]:
            side = "right"
        else:
            side = "left"

        events[side]["HS"]["frame"].append(HS_frame)
        events[side]["HS"]["time"].append(HS_time)
        events[side]["TO"]["frame"].append(TO_frame)
        events[side]["TO"]["time"].append(TO_time)

    events = sort_events(events)

    if len(events["right"]["HS"]["frame"]) < 2 or len(events["left"]["HS"]["frame"]) < 2:
        raise ValueError(
            f"Pas assez de HS dans {file} : "
            f"right={len(events['right']['HS']['frame'])}, "
            f"left={len(events['left']['HS']['frame'])}"
        )

    # ------------------------------------------------------------------
    #      DYNAMIQUE INVERSE – JAMBE GAUCHE
    # ------------------------------------------------------------------
    m_foot_l  = anthropo["Foot"]["mass"]
    k_foot_l  = anthropo["Foot"]["radius_gyr"]
    Ic_foot_l = m_foot_l * k_foot_l**2

    m_leg_l   = anthropo["Leg"]["mass"]
    k_leg_l   = anthropo["Leg"]["radius_gyr"]
    Ic_leg_l  = m_leg_l * k_leg_l**2

    m_thigh_l   = anthropo["Thigh"]["mass"]
    k_thigh_l   = anthropo["Thigh"]["radius_gyr"]
    Ic_thigh_l  = m_thigh_l * k_thigh_l**2

    # --- Pied gauche (cheville) ---
    left_foot_m = markers_m["PIED_G"][:, 1:] - markers_m["CHEVILLE_G"][:, 1:]
    theta_l_foot = np.arctan2(left_foot_m[:, 1], left_foot_m[:, 0])
    alpha_l_foot = compute_derivative(theta_l_foot, dt, 2)

    com_l_foot = markers_m["CHEVILLE_G"][:, 1:] + com_frac["Foot"] * left_foot_m
    r_yC_l = com_l_foot[:, 0] - markers_m["CHEVILLE_G"][:, 1]
    r_zC_l = com_l_foot[:, 1] - markers_m["CHEVILLE_G"][:, 2]

    acc_com_l_y = compute_derivative(com_l_foot[:, 0], dt, 2)
    acc_com_l_z = compute_derivative(com_l_foot[:, 1], dt, 2)

    # GRF / CoP sur la plate-forme gauche (on suppose index 1)
    cop_y_fp_l = grw[1]["P"][:, 1]
    cop_z_fp_l = grw[1]["P"][:, 2]
    Fy_fp_l    = grw[1]["F"][:, 1]
    Fz_fp_l    = grw[1]["F"][:, 2]

    cop_y_l = np.interp(time, fp_time, cop_y_fp_l) / 1000.0
    cop_z_l = np.interp(time, fp_time, cop_z_fp_l) / 1000.0
    Fy_l    = np.interp(time, fp_time, Fy_fp_l)
    Fz_l    = np.interp(time, fp_time, Fz_fp_l)

    r_yP_l = cop_y_l - markers_m["CHEVILLE_G"][:, 1]
    r_zP_l = cop_z_l - markers_m["CHEVILLE_G"][:, 2]

    term_rot_foot_l   = Ic_foot_l * alpha_l_foot
    term_trans_foot_l = m_foot_l * (r_yC_l * acc_com_l_z - r_zC_l * acc_com_l_y)
    term_grav_foot_l  = m_foot_l * g * r_yC_l
    M_grf_foot_l      = r_yP_l * Fz_l - r_zP_l * Fy_l

    M_ankle_left_full      = term_rot_foot_l + term_trans_foot_l + term_grav_foot_l + M_grf_foot_l
    M_ankle_left_full_norm = M_ankle_left_full / body_mass

    # --- Jambe gauche (genou) ---
    # Réaction à la cheville (équation de Newton pied)
    R_A_y_l = m_foot_l * acc_com_l_y - Fy_l
    R_A_z_l = m_foot_l * acc_com_l_z + m_foot_l * g - Fz_l

    # Force distale sur la jambe = pied sur jambe
    Fdist_y_l = -R_A_y_l
    Fdist_z_l = -R_A_z_l

    left_leg_m = markers_m["CHEVILLE_G"][:, 1:] - markers_m["GENOU_G"][:, 1:]
    theta_l_leg = np.arctan2(left_leg_m[:, 1], left_leg_m[:, 0])
    alpha_l_leg = compute_derivative(theta_l_leg, dt, 2)

    com_l_leg = markers_m["GENOU_G"][:, 1:] + com_frac["Leg"] * left_leg_m
    r_yC_leg_l = com_l_leg[:, 0] - markers_m["GENOU_G"][:, 1]
    r_zC_leg_l = com_l_leg[:, 1] - markers_m["GENOU_G"][:, 2]

    acc_com_leg_y_l = compute_derivative(com_l_leg[:, 0], dt, 2)
    acc_com_leg_z_l = compute_derivative(com_l_leg[:, 1], dt, 2)

    # Réaction proximale au genou (sur la jambe)
    R_K_y_l = m_leg_l * acc_com_leg_y_l + Fdist_y_l
    R_K_z_l = m_leg_l * acc_com_leg_z_l + m_leg_l * g + Fdist_z_l

    r_yA_leg_l = markers_m["CHEVILLE_G"][:, 1] - markers_m["GENOU_G"][:, 1]
    r_zA_leg_l = markers_m["CHEVILLE_G"][:, 2] - markers_m["GENOU_G"][:, 2]

    term_rot_leg_l   = Ic_leg_l * alpha_l_leg
    term_trans_leg_l = m_leg_l * (r_yC_leg_l * acc_com_leg_z_l - r_zC_leg_l * acc_com_leg_y_l)
    term_grav_leg_l  = m_leg_l * g * r_yC_leg_l

    M_dist_leg_l         = -M_ankle_left_full
    moment_distal_force_l = -(r_yA_leg_l * Fdist_z_l - r_zA_leg_l * Fdist_y_l)

    M_knee_left      = term_rot_leg_l + term_trans_leg_l + term_grav_leg_l + moment_distal_force_l + M_dist_leg_l
    M_knee_left_norm = M_knee_left / body_mass

    # --- Cuisse gauche (hanche) ---
    # Force distale au genou (sur la cuisse)
    FdistK_y_l = -R_K_y_l
    FdistK_z_l = -R_K_z_l

    left_thigh_m = markers_m["GENOU_G"][:, 1:] - markers_m["HANCHE_G"][:, 1:]
    theta_l_thigh = np.arctan2(left_thigh_m[:, 1], left_thigh_m[:, 0])
    alpha_l_thigh = compute_derivative(theta_l_thigh, dt, 2)

    com_l_thigh = markers_m["HANCHE_G"][:, 1:] + com_frac["Thigh"] * left_thigh_m
    r_yC_th_l = com_l_thigh[:, 0] - markers_m["HANCHE_G"][:, 1]
    r_zC_th_l = com_l_thigh[:, 1] - markers_m["HANCHE_G"][:, 2]

    acc_com_th_y_l = compute_derivative(com_l_thigh[:, 0], dt, 2)
    acc_com_th_z_l = compute_derivative(com_l_thigh[:, 1], dt, 2)

    r_yK_th_l = markers_m["GENOU_G"][:, 1] - markers_m["HANCHE_G"][:, 1]
    r_zK_th_l = markers_m["GENOU_G"][:, 2] - markers_m["HANCHE_G"][:, 2]

    term_rot_th_l   = Ic_thigh_l * alpha_l_thigh
    term_trans_th_l = m_thigh_l * (r_yC_th_l * acc_com_th_z_l - r_zC_th_l * acc_com_th_y_l)
    term_grav_th_l  = m_thigh_l * g * r_yC_th_l

    moment_distal_force_th_l = -(r_yK_th_l * FdistK_z_l - r_zK_th_l * FdistK_y_l)
    M_dist_thigh_l           = -M_knee_left

    M_hip_left      = term_rot_th_l + term_trans_th_l + term_grav_th_l + moment_distal_force_th_l + M_dist_thigh_l
    M_hip_left_norm = M_hip_left / body_mass

    # ------------------------------------------------------------------
    #      DYNAMIQUE INVERSE – JAMBE DROITE (symétrique)
    # ------------------------------------------------------------------
    m_foot_r  = anthropo["Foot"]["mass"]
    k_foot_r  = anthropo["Foot"]["radius_gyr"]
    Ic_foot_r = m_foot_r * k_foot_r**2

    m_leg_r  = anthropo["Leg"]["mass"]
    k_leg_r  = anthropo["Leg"]["radius_gyr"]
    Ic_leg_r = m_leg_r * k_leg_r**2

    m_thigh_r  = anthropo["Thigh"]["mass"]
    k_thigh_r  = anthropo["Thigh"]["radius_gyr"]
    Ic_thigh_r = m_thigh_r * k_thigh_r**2

    # Pied droit
    right_foot_m = markers_m["PIED_D"][:, 1:] - markers_m["CHEVILLE_D"][:, 1:]
    theta_r_foot = np.arctan2(right_foot_m[:, 1], right_foot_m[:, 0])
    alpha_r_foot = compute_derivative(theta_r_foot, dt, 2)

    com_r_foot = markers_m["CHEVILLE_D"][:, 1:] + com_frac["Foot"] * right_foot_m
    r_yC_r = com_r_foot[:, 0] - markers_m["CHEVILLE_D"][:, 1]
    r_zC_r = com_r_foot[:, 1] - markers_m["CHEVILLE_D"][:, 2]

    acc_com_r_y = compute_derivative(com_r_foot[:, 0], dt, 2)
    acc_com_r_z = compute_derivative(com_r_foot[:, 1], dt, 2)

    cop_y_fp_r = grw[0]["P"][:, 1]
    cop_z_fp_r = grw[0]["P"][:, 2]
    Fy_fp_r    = grw[0]["F"][:, 1]
    Fz_fp_r    = grw[0]["F"][:, 2]

    cop_y_r = np.interp(time, fp_time, cop_y_fp_r) / 1000.0
    cop_z_r = np.interp(time, fp_time, cop_z_fp_r) / 1000.0
    Fy_r    = np.interp(time, fp_time, Fy_fp_r)
    Fz_r    = np.interp(time, fp_time, Fz_fp_r)

    r_yP_r = cop_y_r - markers_m["CHEVILLE_D"][:, 1]
    r_zP_r = cop_z_r - markers_m["CHEVILLE_D"][:, 2]

    term_rot_foot_r   = Ic_foot_r * alpha_r_foot
    term_trans_foot_r = m_foot_r * (r_yC_r * acc_com_r_z - r_zC_r * acc_com_r_y)
    term_grav_foot_r  = m_foot_r * g * r_yC_r
    M_grf_foot_r      = r_yP_r * Fz_r - r_zP_r * Fy_r

    M_ankle_right_full      = term_rot_foot_r + term_trans_foot_r + term_grav_foot_r + M_grf_foot_r
    M_ankle_right_full_norm = M_ankle_right_full / body_mass

    # Jambe droite
    R_A_y_r = m_foot_r * acc_com_r_y - Fy_r
    R_A_z_r = m_foot_r * acc_com_r_z + m_foot_r * g - Fz_r

    Fdist_y_r = -R_A_y_r
    Fdist_z_r = -R_A_z_r

    right_leg_m = markers_m["CHEVILLE_D"][:, 1:] - markers_m["GENOU_D"][:, 1:]
    theta_r_leg = np.arctan2(right_leg_m[:, 1], right_leg_m[:, 0])
    alpha_r_leg = compute_derivative(theta_r_leg, dt, 2)

    com_r_leg = markers_m["GENOU_D"][:, 1:] + com_frac["Leg"] * right_leg_m
    r_yC_leg_r = com_r_leg[:, 0] - markers_m["GENOU_D"][:, 1]
    r_zC_leg_r = com_r_leg[:, 1] - markers_m["GENOU_D"][:, 2]

    acc_com_leg_y_r = compute_derivative(com_r_leg[:, 0], dt, 2)
    acc_com_leg_z_r = compute_derivative(com_r_leg[:, 1], dt, 2)

    R_K_y_r = m_leg_r * acc_com_leg_y_r + Fdist_y_r
    R_K_z_r = m_leg_r * acc_com_leg_z_r + m_leg_r * g + Fdist_z_r

    r_yA_leg_r = markers_m["CHEVILLE_D"][:, 1] - markers_m["GENOU_D"][:, 1]
    r_zA_leg_r = markers_m["CHEVILLE_D"][:, 2] - markers_m["GENOU_D"][:, 2]

    term_rot_leg_r   = Ic_leg_r * alpha_r_leg
    term_trans_leg_r = m_leg_r * (r_yC_leg_r * acc_com_leg_z_r - r_zC_leg_r * acc_com_leg_y_r)
    term_grav_leg_r  = m_leg_r * g * r_yC_leg_r

    M_dist_leg_r         = -M_ankle_right_full
    moment_distal_force_r = -(r_yA_leg_r * Fdist_z_r - r_zA_leg_r * Fdist_y_r)

    M_knee_right      = term_rot_leg_r + term_trans_leg_r + term_grav_leg_r + moment_distal_force_r + M_dist_leg_r
    M_knee_right_norm = M_knee_right / body_mass

    # Cuisse droite
    FdistK_y_r = -R_K_y_r
    FdistK_z_r = -R_K_z_r

    right_thigh_m = markers_m["GENOU_D"][:, 1:] - markers_m["HANCHE_D"][:, 1:]
    theta_r_thigh = np.arctan2(right_thigh_m[:, 1], right_thigh_m[:, 0])
    alpha_r_thigh = compute_derivative(theta_r_thigh, dt, 2)

    com_r_thigh = markers_m["HANCHE_D"][:, 1:] + com_frac["Thigh"] * right_thigh_m
    r_yC_th_r = com_r_thigh[:, 0] - markers_m["HANCHE_D"][:, 1]
    r_zC_th_r = com_r_thigh[:, 1] - markers_m["HANCHE_D"][:, 2]

    acc_com_th_y_r = compute_derivative(com_r_thigh[:, 0], dt, 2)
    acc_com_th_z_r = compute_derivative(com_r_thigh[:, 1], dt, 2)

    r_yK_th_r = markers_m["GENOU_D"][:, 1] - markers_m["HANCHE_D"][:, 1]
    r_zK_th_r = markers_m["GENOU_D"][:, 2] - markers_m["HANCHE_D"][:, 2]

    term_rot_th_r   = Ic_thigh_r * alpha_r_thigh
    term_trans_th_r = m_thigh_r * (r_yC_th_r * acc_com_th_z_r - r_zC_th_r * acc_com_th_y_r)
    term_grav_th_r  = m_thigh_r * g * r_yC_th_r

    moment_distal_force_th_r = -(r_yK_th_r * FdistK_z_r - r_zK_th_r * FdistK_y_r)
    M_dist_thigh_r           = -M_knee_right

    M_hip_right      = term_rot_th_r + term_trans_th_r + term_grav_th_r + moment_distal_force_th_r + M_dist_thigh_r
    M_hip_right_norm = M_hip_right / body_mass

    # ------------------------------------------------------------------
    #      EXTRACTION ET NORMALISATION DES CYCLES HS–HS
    # ------------------------------------------------------------------
    start_L, end_L = events["left"]["HS"]["frame"][0],  events["left"]["HS"]["frame"][1]
    start_R, end_R = events["right"]["HS"]["frame"][0], events["right"]["HS"]["frame"][1]

    gc, M_ankle_L = normalize_cycle(M_ankle_left_full_norm,  start_L, end_L)
    _,  M_knee_L  = normalize_cycle(M_knee_left_norm,       start_L, end_L)
    _,  M_hip_L   = normalize_cycle(M_hip_left_norm,        start_L, end_L)

    _,  M_ankle_R = normalize_cycle(M_ankle_right_full_norm, start_R, end_R)
    _,  M_knee_R  = normalize_cycle(M_knee_right_norm,       start_R, end_R)
    _,  M_hip_R   = normalize_cycle(M_hip_right_norm,        start_R, end_R)

    results = {
        "gc": gc,
        "left": {
            "Ankle": M_ankle_L,
            "Knee":  M_knee_L,
            "Hip":   M_hip_L,
        },
        "right": {
            "Ankle": M_ankle_R,
            "Knee":  M_knee_R,
            "Hip":   M_hip_R,
        }
    }
    return results

# ----------------------------------------------------------------------
#      AGRÉGATION PAR GROUPE ET PLOTS
# ----------------------------------------------------------------------
def compute_group_moments(dir_path, joints):
    """
    Parcourt tous les fichiers c3d d'un dossier (HEA ou IMP),
    applique inverse_dynamics et stocke toutes les courbes
    normalisées dans all_moments[joint][side].
    """
    all_moments = {j: {"right": [], "left": []} for j in joints}
    gc = None

    for file in dir_path.iterdir():
        if file.suffix.lower() != ".c3d":
            continue

        res = inverse_dynamics(file, joints=joints)

        if gc is None:
            gc = res["gc"]

        for joint in joints:
            all_moments[joint]["right"].append(res["right"][joint])
            all_moments[joint]["left"].append(res["left"][joint])

    return gc, all_moments

def plot_group_moments(gc, all_moments, joints, filename):
    """
    Trace, pour chaque articulation, la moyenne ± 1 SD
    des moments articulaires (Nm/kg) des membres droits et gauches.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    for i, joint in enumerate(joints):
        ax = axes[i]

        for side, color, label in [("right", "tab:blue", "Right"),
                                   ("left",  "tab:orange", "Left")]:

            data = np.vstack(all_moments[joint][side])
            mean_curve = data.mean(axis=0)
            std_curve  = data.std(axis=0)

            ax.plot(gc, mean_curve, label=label, linewidth=2, color=color)
            ax.fill_between(gc,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            alpha=0.2)

        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(joint, fontsize=10, fontweight="bold")
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Joint moment (Nm/kg)")
        ax.set_xlim(0, 100)
        ax.grid(True)

    axes[0].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(str(output_dir / filename), dpi=300)
    plt.close()

# ----------------------------------------------------------------------
#      SCRIPT PRINCIPAL (GROUPES HEA / IMP)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Moments articulaires pour le groupe healthy
    gc_hea, moments_hea = compute_group_moments(HEA, joints)
    plot_group_moments(gc_hea, moments_hea, joints,
                       filename="joint_moments_healthy.png"
                       )

    # Moments articulaires pour le groupe impaired
    gc_imp, moments_imp = compute_group_moments(IMP, joints)
    plot_group_moments(gc_imp, moments_imp, joints,
                       filename="joint_moments_impaired.png"
                       )

