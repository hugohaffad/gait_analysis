from config import *

import numpy as np
import matplotlib.pyplot as plt
import moveck_bridge_btk as btk

output_dir = REP
output_dir.mkdir(parents=True, exist_ok=True)

joints = ["Hip", "Knee", "Ankle"]

def sort_events(events):
    for side in events:
        for evt in ["HS", "TO"]:
            frames = np.array(events[side][evt]["frame"])
            times  = np.array(events[side][evt]["time"])

            order = np.argsort(frames)

            events[side][evt]["frame"] = frames[order].tolist()
            events[side][evt]["time"]  = times[order].tolist()

    return events

def normalize_cycle(signal, start_frame, end_frame, n_points=101):
    cycle = signal[start_frame:end_frame+1]
    n = len(cycle)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)

    cycle_norm = np.interp(x_new, x_old, cycle)

    return x_new, cycle_norm

def angle(v):
    theta = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 2]), period=2*np.pi))
    return theta

def kinematics_analysis(file, joints):

    h = btk.btkReadAcquisition(str(file))

    # ---- Markers ---- #
    markers, markersInfo = btk.btkGetMarkers(h)
    freq = btk.btkGetPointFrequency(h)

    # ---- Segments ---- #
    left_pelvis = markers["EIAS_G"] - markers["HANCHE_G"]
    right_pelvis = markers["EIAS_D"] - markers["HANCHE_D"]
    left_thigh = markers["GENOU_G"] - markers["HANCHE_G"]
    right_thigh = markers["GENOU_D"] - markers["HANCHE_D"]
    left_leg = markers["GENOU_G"] - markers["CHEVILLE_G"]
    right_leg = markers["GENOU_D"] - markers["CHEVILLE_D"]
    left_foot = markers["PIED_G"] - markers["CHEVILLE_G"]
    right_foot = markers["PIED_D"] - markers["CHEVILLE_D"]

    # ---- Angles ---- #
    left_hip = (180 - (angle(left_thigh) - angle(left_pelvis))) % 360
    left_hip = np.where(left_hip > 180, left_hip - 360, left_hip)
    right_hip = (180 - (angle(right_thigh) - angle(right_pelvis))) % 360
    right_hip = np.where(right_hip > 180, right_hip - 360, right_hip)

    left_knee = (180 - (angle(left_thigh) - angle(left_leg))) % 360
    left_knee = np.where(left_knee > 180, left_knee - 360, left_knee)
    right_knee = (180 - (angle(right_thigh) - angle(right_leg))) % 360
    right_knee = np.where(right_knee > 180, right_knee - 360, right_knee)

    left_ankle = 90 - (angle(left_foot) - angle(left_leg))
    left_ankle = np.where(left_ankle > 180, left_ankle - 360, left_ankle)
    right_ankle = 90 - (angle(right_foot) - angle(right_leg))
    right_ankle = np.where(right_ankle > 180, right_ankle - 360, right_ankle)

    angles = {
        "Hip": {"left": left_hip, "right": right_hip},
        "Knee": {"left": left_knee, "right": right_knee},
        "Ankle": {"left": left_ankle, "right": right_ankle},
    }

    # ---- Forceplates ---- #
    forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
    grw = btk.btkGetGroundReactionWrenches(h)
    fp_freq = btk.btkGetAnalogFrequency(h)
    active_forceplates = []

    # ---- Events ---- #
    events = {
        "left": {"HS": {"frame": [], "time": []},
                 "TO": {"frame": [], "time": []}},
        "right": {"HS": {"frame": [], "time": []},
                  "TO": {"frame": [], "time": []}},
    }

    for i in range(len(forceplates)):
        if np.count_nonzero(grw[i]["F"]) == 0:
            continue
        else:
            active_forceplates.append(i)

    for i in active_forceplates:
        GRFz = grw[i]["F"][:, 2]

        HS_idx = np.where(GRFz > 5)[0][0]
        TO_idx = np.where(GRFz > 5)[0][-1]

        HS_time = HS_idx / fp_freq
        TO_time = TO_idx / fp_freq

        HS_frame = int(np.round(HS_time * freq))
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

    if len(events["right"]["HS"]["frame"]) < 2 or len(events["left"]["HS"]["frame"]) < 2:
        raise ValueError(
            f"Pas assez de HS dans {file} : "
            f"right={len(events['right']['HS']['frame'])}, "
            f"left={len(events['left']['HS']['frame'])}"
        )

    start_R, end_R = events["right"]["HS"]["frame"][0], events["right"]["HS"]["frame"][1]
    start_L, end_L = events["left"]["HS"]["frame"][0], events["left"]["HS"]["frame"][1]

    # ---- Normalisation des cycles ---- #
    angles_norm = {"right": {}, "left": {}}
    gc = None

    for joint in joints:
        # droite
        sig_R = angles[joint]["right"]
        gc_tmp, sig_R_norm = normalize_cycle(sig_R, start_R, end_R)

        # gauche
        sig_L = angles[joint]["left"]
        _, sig_L_norm = normalize_cycle(sig_L, start_L, end_L)

        angles_norm["right"][joint] = sig_R_norm
        angles_norm["left"][joint]  = sig_L_norm

        if gc is None:
            gc = gc_tmp   # on fixe le vecteur 0–100 % une fois pour toutes

    results = {
        "gc": gc,
        "right": angles_norm["right"],
        "left":  angles_norm["left"],
    }

    return results

def compute_group_curves(dir_path, joints):
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

def plot_group_curves(gc, all_curves, joints, title, filename):
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

    axes[0].set_ylim(-10, 40)
    axes[0].legend(loc="upper right")
    axes[1].set_ylim(0, 80)
    axes[2].set_ylim(0, -50)
    plt.tight_layout()
    plt.savefig(str(output_dir / filename), dpi=300)
    plt.close()

#---- Reports ----#
def compute_asym_from_means(curves, joints):
    """
    Calcule, pour chaque articulation, l'asymétrie droite-gauche
    à partir des COURBES MOYENNES dans un groupe donné.

    Asym(t) = Right_mean(t) - Left_mean(t)
    """
    asym = {}

    for joint in joints:
        left_mean  = np.vstack(curves[joint]["left"]).mean(axis=0)
        right_mean = np.vstack(curves[joint]["right"]).mean(axis=0)

        asym[joint] = right_mean - left_mean   # R - L

    return asym

gc_hea, curves_hea = compute_group_curves(HEA, joints)
gc_imp, curves_imp = compute_group_curves(IMP, joints)
asym_hea = compute_asym_from_means(curves_hea, joints)
asym_imp = compute_asym_from_means(curves_imp, joints)

plot_group_curves(gc_hea, curves_hea, joints,
                  title="Healthy gait",
                  filename="joint_angles_healthy.png")

plot_group_curves(gc_imp, curves_imp, joints,
                  title="Impaired gait",
                  filename="joint_angles_impaired.png")

def compute_delta_asym_from_means(curves_hea, curves_imp, joints):
    """
    Calcule, pour chaque articulation, la différence d'asymétrie G-D
    entre condition altérée (IMP) et condition saine (HEA),
    à partir des COURBES MOYENNES.

    Delta_asym(t) = [ (Left - Right)_IMP ] - [ (Left - Right)_HEA ]
    """

    delta_asym = {}

    for joint in joints:
        # Healthy : moyenne des courbes G et D
        left_hea  = np.vstack(curves_hea[joint]["left"]).mean(axis=0)
        right_hea = np.vstack(curves_hea[joint]["right"]).mean(axis=0)
        diff_hea  =  right_hea - left_hea

        # Impaired : moyenne des courbes G et D
        left_imp  = np.vstack(curves_imp[joint]["left"]).mean(axis=0)
        right_imp = np.vstack(curves_imp[joint]["right"]).mean(axis=0)
        diff_imp  =  right_imp - left_imp   # B(t)

        # Delta = B - A, point par point sur le cycle
        delta_asym[joint] = diff_imp - diff_hea

    return delta_asym

# delta_asym à partir des MOYENNES (comme ton analyse visuelle)
delta_asym = compute_delta_asym_from_means(curves_hea, curves_imp, joints)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, joint in enumerate(joints):
    ax = axes[i]
    da = delta_asym[joint]            # Δ_asym(t)

    mean_delta = da.mean()            # moyenne sur 0–100%

    # Courbe Δ_asym
    ax.plot(gc_hea, da, linewidth=2)

    # Ligne horizontale = moyenne
    ax.axhline(mean_delta, color="k", linestyle="-", linewidth=1.5)

    ax.axhline(0, linestyle="--", linewidth=1, color="gray")  # ligne zéro

    ax.set_title(f"{joint}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Δ asym (°)")
    ax.set_xlim(0, 100)
    ax.grid(True)

plt.tight_layout()
plt.savefig(str(output_dir / "delta_asymmetry.png"), dpi=300)
plt.close()

def compute_integral_asym(delta_asym):
    """
    delta_asym = dict(joint -> array of Δ_asym(t))
    Retourne un dict avec l'intégrale d'asymétrie pour chaque articulation.
    """
    integral = {}

    for joint, curve in delta_asym.items():
        integral[joint] = np.sum(np.abs(curve))  # aire sous la courbe |Δ_asym|

    return integral

integral_asym = compute_integral_asym(delta_asym)

print("Intégrale d’asymétrie (aire sous |Δ_asym|) :")
for joint in joints:
    print(f"{joint}: {integral_asym[joint]:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, joint in enumerate(joints):
    ax = axes[i]
    curve = asym_hea[joint]

    mean_asym = curve.mean()

    ax.plot(gc_hea, curve, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(mean_asym, color="red", linestyle="-", linewidth=1.5)

    ax.set_title(f"{joint}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Asymmetry (°)")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].set_ylim(-30, 30)
axes[1].set_ylim(-30, 30)
axes[2].set_ylim(-30, 30)

plt.tight_layout()
plt.savefig(str(output_dir / "asymmetry_healthy.png"), dpi=300)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, joint in enumerate(joints):
    ax = axes[i]
    curve = asym_imp[joint]

    mean_asym = curve.mean()

    ax.plot(gc_imp, curve, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(mean_asym, color="red", linestyle="-", linewidth=1.5)

    ax.set_title(f"{joint}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Asymmetry (°)")
    ax.set_xlim(0, 100)
    ax.grid(True)

axes[0].set_ylim(-30, 30)
axes[1].set_ylim(-30, 30)
axes[2].set_ylim(-30, 30)

plt.tight_layout()
plt.savefig(str(output_dir / "asymmetry_impaired.png"), dpi=300)
plt.close()

def compute_integral_asym(asym):
    """
    asym[joint] = courbe R-L(t)
    Retourne l'aire sous |R-L| pour chaque articulation.
    """
    integral = {}
    for joint, curve in asym.items():
        integral[joint] = np.sum(np.abs(curve))
    return integral

integral_hea = compute_integral_asym(asym_hea)
integral_imp = compute_integral_asym(asym_imp)

print("Intégrale d’asymétrie (Healthy) :")
for joint in joints:
    print(f"{joint}: {integral_hea[joint]:.2f}")

print("Intégrale d’asymétrie (Impaired) :")
for joint in joints:
    print(f"{joint}: {integral_imp[joint]:.2f}")
